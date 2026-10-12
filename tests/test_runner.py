"""End-to-end deployment loop against a simulated robot (perfect tracking)."""

import numpy as np

from dexmulti.alignment.targets import alignment_yaw
from dexmulti.config import DexMultiConfig
from dexmulti.execution import replay_waypoints
from dexmulti.geometry import Rz
from dexmulti.perception.state import ObjectStateEstimator
from dexmulti.retrieval import DualTaskPolicyChamfer
from dexmulti.runner import DexMultiRunner


class SimRobot:
    def __init__(self):
        self.pose = np.array([0.35, 0.0, 0.45, 3.14, 0.0, 0.0], np.float32)
        self.q = np.zeros(7, np.float32)
        self.hand = np.zeros(16, np.float32)
        self.mode = None
        self.log = []

    def get_eef_pose(self): return self.pose.copy()
    def get_arm_joints(self): return self.q.copy()
    def get_hand_joints(self): return self.hand.copy()
    def set_control_mode(self, mode): self.mode = mode
    def send_arm_joints(self, q): self.q = np.asarray(q, np.float32); self.log.append(("joint", q))
    def send_hand(self, q): self.hand = np.asarray(q, np.float32)

    def send_eef_pose(self, pos, rpy):
        assert self.mode == "pose"
        self.pose = np.concatenate([pos, rpy]).astype(np.float32)
        self.log.append(("pose", self.pose.copy()))


class SimPerception:
    def __init__(self, clouds):
        self.clouds, self.states = clouds, {}
        for name, c in clouds.items():
            est = ObjectStateEstimator()
            for _ in range(60):
                est.update(c)
            self.states[name] = est.state

    def get_cloud(self, name): return self.clouds[name]
    def get_state(self, name): return self.states[name]


class HoldPlanner:
    def step(self, q, pos, rpy, hand, world):
        assert len(world) >= 2
        return np.concatenate([q[:7], hand])


def test_run_once(grasp_pull_library):
    i, yaw = 4, 0.4
    obj = np.asarray(grasp_pull_library["obj_pc_canon"][i], np.float32) @ Rz(yaw).T + [0.4, 0.05, 0.06]
    dr = np.asarray(grasp_pull_library["drawer_pc_canon"][i], np.float32) + [0.65, 0.0, 0.12]
    robot, perception = SimRobot(), SimPerception({"object": obj, "drawer": dr})
    cfg = DexMultiConfig()
    cfg.deploy.align_seconds, cfg.deploy.bridge_seconds, cfg.deploy.ctrl_hz = 0.2, 0.2, 200.0
    cfg.arm.max_step_pos_m, cfg.arm.max_step_rot_deg = 10.0, 360.0  # perfect tracking, no rate limit
    policy = DualTaskPolicyChamfer(grasp_pull_library)

    assert DexMultiRunner(robot, perception, policy, HoldPlanner(), cfg, log=lambda s: None).run_once()
    assert policy.demo_idx == i
    # The last command is the last waypoint of skill 2 mapped with the frozen drawer pose.
    st = perception.get_state("drawer")
    yaw_dr = alignment_yaw(st.yaw, policy.drawer_theta_star, grasp_pull_library["drawer_rot"][i][0])
    *_, (pos, _, _) = replay_waypoints(st.center, yaw_dr, grasp_pull_library["drawer_pos"][i],
                                       grasp_pull_library["drawer_rot"][i], grasp_pull_library["drawer_hand"][i])
    np.testing.assert_allclose(robot.log[-1][1][:3], np.clip(pos, [0.1, -0.5, 0.15], [0.7, 0.5, 0.5]), atol=1e-4)
