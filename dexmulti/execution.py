"""Bridge + open-loop replay of a retrieved canonical skill, and Cartesian safety limits."""

import time
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from dexmulti.config import ArmLimits
from dexmulti.geometry import canon_action_to_world, wrap_to_pi


def clip_to_workspace(pos: np.ndarray, limits: ArmLimits = ArmLimits()) -> np.ndarray:
    ws = limits.workspace
    return np.array([min(max(float(pos[i]), ws[ax][0]), ws[ax][1]) for i, ax in enumerate("xyz")],
                    dtype=np.float32)


def clip_and_clamp(target_pos: np.ndarray, target_rpy: np.ndarray, current_pose: np.ndarray,
                   limits: ArmLimits = ArmLimits()) -> Tuple[np.ndarray, np.ndarray]:
    """Workspace box clip, then limit the step from the *current* EE pose to
    ``max_step_pos_m`` / ``max_step_rot_deg``. Applied to every streamed Cartesian command
    in the experiments, so the replayed motion is rate-limited to 0.1 m/s and 25 deg/s."""
    target_pos = clip_to_workspace(target_pos, limits).astype(float)
    target_rpy = np.asarray(target_rpy, dtype=float)
    cur_pos, cur_rpy = np.asarray(current_pose[:3], dtype=float), np.asarray(current_pose[3:6], dtype=float)

    delta = target_pos - cur_pos
    dist = np.linalg.norm(delta)
    if dist > limits.max_step_pos_m and dist > 1e-6:
        delta *= limits.max_step_pos_m / dist
    pos = cur_pos + delta

    cur_R = R.from_euler("xyz", cur_rpy)
    dR = cur_R.inv() * R.from_euler("xyz", target_rpy)
    step_deg = np.degrees(np.linalg.norm(dR.as_rotvec()))
    if step_deg > limits.max_step_rot_deg and step_deg > 1e-6:
        dR = R.from_rotvec(dR.as_rotvec() * (limits.max_step_rot_deg / step_deg))
    rpy = (cur_R * dR).as_euler("xyz")
    return pos, rpy


class Rate:
    """Fixed-rate loop helper (like rospy.Rate) on the monotonic clock."""

    def __init__(self, hz: float):
        self.period = 1.0 / hz
        self._next = time.monotonic() + self.period

    def sleep(self) -> None:
        now = time.monotonic()
        if self._next > now:
            time.sleep(self._next - now)
            self._next += self.period
        else:  # overrun: re-anchor
            self._next = now + self.period


def bridge_waypoint(alpha: float, pos_start, R_start: R, pos_target, R_target: R):
    """Linear position + SLERP orientation at fraction alpha in [0, 1]."""
    slerp = Slerp([0.0, 1.0], R.from_quat([R_start.as_quat(), R_target.as_quat()]))
    pos = (1.0 - alpha) * np.asarray(pos_start) + alpha * np.asarray(pos_target)
    rpy = slerp(alpha).as_euler("xyz").astype(np.float32)
    rpy[2] = wrap_to_pi(rpy[2])
    return pos, rpy


def run_bridge(robot, send_pose: Callable, c_fixed: np.ndarray, yaw_fixed: float,
               pos_canon: np.ndarray, rot_canon: np.ndarray, hand_cmd,
               seconds: float = 3.0, ctrl_hz: float = 10.0,
               early_stop_fn: Callable[[], bool] = lambda: False) -> bool:
    """Time-based interpolation from the current EE pose / hand joints to the first
    action of the skill (mapped with the frozen object pose)."""
    pos_target, rpy_target = canon_action_to_world(c_fixed, yaw_fixed, pos_canon, rot_canon)
    R_target = R.from_euler("xyz", rpy_target)

    ee = robot.get_eef_pose()
    pos_start, R_start = ee[:3], R.from_euler("xyz", ee[3:6])

    hand_target = None
    if hand_cmd is not None:
        hand_start = np.asarray(robot.get_hand_joints(), dtype=np.float32)
        hand_target = np.asarray(hand_cmd, dtype=np.float32)
        assert hand_start.shape == hand_target.shape

    rate = Rate(ctrl_hz)
    t0 = time.monotonic()
    while True:
        if early_stop_fn():
            return False
        t = time.monotonic() - t0
        if t >= seconds:
            break
        alpha = float(np.clip(t / seconds, 0.0, 1.0))
        pos, rpy = bridge_waypoint(alpha, pos_start, R_start, pos_target, R_target)
        send_pose(pos, rpy)
        if hand_target is not None:
            robot.send_hand((1.0 - alpha) * hand_start + alpha * hand_target)
        rate.sleep()

    send_pose(pos_target, rpy_target)
    if hand_target is not None:
        robot.send_hand(hand_target)
    return True


def replay_waypoints(c_fixed: np.ndarray, yaw_fixed: float, pos_seq_canon: Iterable,
                     rot_seq_canon: Iterable, hand_seq: Iterable):
    """Yields (pos, rpy, hand_cmd or None) world-frame commands of an open-loop replay."""
    for pos_canon, rot_canon, hcmd in zip(pos_seq_canon, rot_seq_canon, hand_seq):
        pos, rpy = canon_action_to_world(c_fixed, yaw_fixed, np.asarray(pos_canon), np.asarray(rot_canon))
        yield pos, rpy, (None if hcmd is None else np.asarray(hcmd, dtype=np.float32))


def run_replay(robot, send_pose: Callable, c_fixed, yaw_fixed, pos_seq_canon, rot_seq_canon, hand_seq,
               ctrl_hz: float = 10.0, early_stop_fn: Callable[[], bool] = lambda: False) -> bool:
    """Open-loop replay of the whole skill with a single frozen object pose."""
    rate = Rate(ctrl_hz)
    for pos, rpy, hcmd in replay_waypoints(c_fixed, yaw_fixed, pos_seq_canon, rot_seq_canon, hand_seq):
        if early_stop_fn():
            return False
        send_pose(pos, rpy)
        if hcmd is not None:
            robot.send_hand(hcmd)
        rate.sleep()
    return True
