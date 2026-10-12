"""ROS1 (Noetic) adapter for the setup used in the paper: xArm7 (xarm_ros driver),
LEAP Hand (LEAP ROS node), and an upstream perception stack that publishes segmented,
fused object point clouds in the arm base frame.

For other setups, implement ``dexmulti.interfaces.RobotInterface`` and ``Perception``.

Topics / services used
  xArm   /xarm/xarm_states (xarm_msgs/RobotMsg), /xarm/joint_states,
         /xarm/move_line, /xarm/move_joint, /xarm/set_mode, /xarm/set_state
  LEAP   /leaphand_node/cmd_leap (sensor_msgs/JointState, 40 Hz), /leap_hand_state
  Clouds /perception/object_fused_segmented_point_cloud,
         /perception/drawer_fused_segmented_point_cloud (sensor_msgs/PointCloud2)
"""

import threading
from typing import Dict, Optional

import numpy as np
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import JointState, PointCloud2
from xarm_msgs.msg import RobotMsg
from xarm_msgs.srv import Move, MoveRequest, SetInt16, SetInt16Request

from dexmulti.config import TrackerConfig
from dexmulti.perception.state import ObjectState, ObjectStateEstimator

XARM_MAX_VELO_MM, XARM_MAX_ACC_MM = 200.0, 2000.0
XARM_JOINT_MAX_VELO, XARM_JOINT_MAX_ACC = 0.35, 7.0
LEAP_PUBLISH_HZ, CTRL_HZ = 40.0, 10.0
XARM_MODE = {"joint": 6, "pose": 7}  # xArm online trajectory-planning modes (joint / Cartesian)


class XArmLeapRobot:
    def __init__(self):
        self._arm_state: Optional[RobotMsg] = None
        self._arm_joints: Optional[JointState] = None
        self._hand_state: Optional[JointState] = None
        rospy.Subscriber("/xarm/xarm_states", RobotMsg, lambda m: setattr(self, "_arm_state", m))
        rospy.Subscriber("/xarm/joint_states", JointState, lambda m: setattr(self, "_arm_joints", m))
        rospy.Subscriber("/leap_hand_state", JointState, lambda m: setattr(self, "_hand_state", m))
        self._move_line = rospy.ServiceProxy("/xarm/move_line", Move)
        self._move_joint = rospy.ServiceProxy("/xarm/move_joint", Move)

        # LEAP: 10 Hz commands linearly upsampled to 40 Hz.
        self._hand_pub = rospy.Publisher("/leaphand_node/cmd_leap", JointState, queue_size=10)
        self._up = int(LEAP_PUBLISH_HZ / CTRL_HZ)
        self._prev = self._next = None
        self._phase = 0
        self._lock = threading.Lock()
        rospy.Timer(rospy.Duration(1.0 / LEAP_PUBLISH_HZ), self._hand_tick)

        for name in ("_arm_state", "_arm_joints", "_hand_state"):
            while getattr(self, name) is None and not rospy.is_shutdown():
                rospy.sleep(0.05)

    # ---- state ----
    def get_eef_pose(self) -> np.ndarray:
        pose = np.array(self._arm_state.pose, dtype=np.float32)
        return np.concatenate([pose[:3] * 1e-3, pose[3:6]])

    def get_arm_joints(self) -> np.ndarray:
        return np.asarray(self._arm_joints.position[:7], dtype=np.float32)

    def get_hand_joints(self) -> np.ndarray:
        return np.asarray(self._hand_state.position, dtype=np.float32)

    # ---- modes ----
    def _srv(self, name: str, value: int):
        rospy.wait_for_service(name, timeout=5.0)
        resp = rospy.ServiceProxy(name, SetInt16)(SetInt16Request(data=int(value)))
        if resp.ret != 0:
            raise RuntimeError(f"{name}({value}) returned {resp.ret}")

    def set_control_mode(self, mode: str) -> None:
        self._srv("/xarm/set_mode", XARM_MODE[mode])
        self._srv("/xarm/set_state", 0)

    # ---- commands ----
    def send_eef_pose(self, pos, rpy) -> None:
        req = MoveRequest()
        req.pose = [*(np.asarray(pos, float) * 1000.0), *map(float, rpy)]
        req.mvvelo, req.mvacc, req.mvtime = XARM_MAX_VELO_MM, XARM_MAX_ACC_MM, 0.0
        resp = self._move_line(req)
        if resp.ret != 0:  # controller error: abort the trial, as the original interface did
            raise RuntimeError(f"/xarm/move_line returned {resp.ret}")

    def send_arm_joints(self, q) -> None:
        req = MoveRequest()
        req.pose = [float(v) for v in q]
        req.mvvelo, req.mvacc, req.mvtime = XARM_JOINT_MAX_VELO, XARM_JOINT_MAX_ACC, 0.0
        resp = self._move_joint(req)
        if resp.ret != 0:
            raise RuntimeError(f"/xarm/move_joint returned {resp.ret}")

    def send_hand(self, q) -> None:
        q = np.asarray(q, dtype=np.float32)
        with self._lock:
            self._prev = q.copy() if self._next is None else self._next.copy()
            self._next = q.copy()
            self._phase = 0

    def _hand_tick(self, _event):
        with self._lock:
            if self._prev is None or self._next is None:
                return
            if self._phase >= self._up:
                self._prev, self._next, self._phase = self._next, None, 0
                return
            a = min(1.0, self._phase / float(self._up - 1))
            q = (1 - a) * self._prev + a * self._next
            self._phase += 1
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = [f"joint_{i}" for i in range(len(q))]
        msg.position = list(map(float, q))
        self._hand_pub.publish(msg)


class RosPerception:
    """Subscribes to segmented clouds and runs one ObjectStateEstimator per object at
    30 Hz (this replaces the separate tracker node used in the experiments)."""

    TOPICS = {
        "object": "/perception/object_fused_segmented_point_cloud",
        "drawer": "/perception/drawer_fused_segmented_point_cloud",
    }

    def __init__(self, cfg: TrackerConfig = TrackerConfig(), baseline_tracker: bool = False):
        self._clouds: Dict[str, Optional[np.ndarray]] = {k: None for k in self.TOPICS}
        self._est = {k: ObjectStateEstimator(cfg, baseline=baseline_tracker) for k in self.TOPICS}
        self._states: Dict[str, Optional[ObjectState]] = {k: None for k in self.TOPICS}
        self._lock = threading.Lock()
        for name, topic in self.TOPICS.items():
            rospy.Subscriber(topic, PointCloud2, self._cb, callback_args=name, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / cfg.rate_hz), self._tick)
        while any(s is None for s in self._states.values()) and not rospy.is_shutdown():
            rospy.sleep(0.05)

    def _cb(self, msg: PointCloud2, name: str):
        pts = np.array(list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
                       dtype=np.float32)
        with self._lock:
            self._clouds[name] = pts

    def _tick(self, _event):
        for name in self.TOPICS:
            with self._lock:
                cloud = self._clouds[name]
            if cloud is None or len(cloud) == 0:
                continue
            st = self._est[name].update(cloud)
            with self._lock:
                self._states[name] = st

    def get_cloud(self, name: str) -> np.ndarray:
        with self._lock:
            return self._clouds[name].copy()

    def get_state(self, name: str) -> Optional[ObjectState]:
        with self._lock:
            return self._states[name]
