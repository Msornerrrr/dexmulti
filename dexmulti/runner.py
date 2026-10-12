"""DexMulti deployment loop (paper Alg. 1): retrieve once, then for each skill
align (closed loop) -> bridge -> replay (open loop)."""

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from dexmulti.alignment.targets import alignment_hand_target, alignment_target
from dexmulti.alignment.world import build_world
from dexmulti.config import DexMultiConfig
from dexmulti.execution import Rate, clip_and_clamp, run_bridge, run_replay
from dexmulti.interfaces import Perception, RobotInterface
from dexmulti.retrieval import DualTaskPolicyChamfer

OBJECT, DRAWER = "object", "drawer"


@dataclass
class AlignResult:
    ok: bool
    center: Optional[np.ndarray] = None
    yaw: Optional[float] = None


class DexMultiRunner:
    def __init__(self, robot: RobotInterface, perception: Perception, policy: DualTaskPolicyChamfer,
                 planner, cfg: DexMultiConfig = DexMultiConfig(),
                 early_stop_fn: Callable[[], bool] = lambda: False, log: Callable[[str], None] = print):
        self.robot = robot
        self.perception = perception
        self.policy = policy
        self.planner = planner  # e.g. alignment.pyroki_planner.PyrokiAlignmentPlanner
        self.cfg = cfg
        self.early_stop_fn = early_stop_fn
        self.log = log

    # ------------------------------------------------------------------
    def send_pose(self, pos, rpy):
        pos, rpy = clip_and_clamp(pos, rpy, self.robot.get_eef_pose(), self.cfg.arm)
        self.robot.send_eef_pose(pos, rpy)

    def _world(self, exclude_object: bool):
        obj, dr = self.perception.get_state(OBJECT), self.perception.get_state(DRAWER)
        return build_world(
            obj.sphere_centers if obj else None, obj.sphere_radii if obj else None,
            dr.sphere_centers if dr else None, dr.sphere_radii if dr else None,
            exclude_object=exclude_object, cfg=self.cfg.world,
        )

    def align(self, name: str, pos_canon, rot_canon, hand_cmd, theta_star: Optional[float],
              grasp_task: bool) -> AlignResult:
        """Closed-loop joint-space servo to the skill's first canonical action, re-mapped
        through the latest tracked object pose at every step."""
        d = self.cfg.deploy
        open_hand = np.zeros_like(self.robot.get_hand_joints(), dtype=np.float32)
        rate = Rate(d.ctrl_hz)
        t0 = time.monotonic()
        last = AlignResult(ok=False)

        while (time.monotonic() - t0) < d.align_seconds:
            if self.early_stop_fn():
                return AlignResult(ok=False)
            state = self.perception.get_state(name)
            if state is None:
                rate.sleep()
                continue
            yaw, pos, rpy = alignment_target(state.center, state.yaw, theta_star, pos_canon, rot_canon,
                                             d.max_abs_world_yaw_rad)
            last = AlignResult(ok=True, center=np.asarray(state.center).copy(), yaw=yaw)

            hand_target = alignment_hand_target(time.monotonic() - t0, d.align_seconds, grasp_task,
                                                hand_cmd, open_hand, d.grasp_open_fraction)
            q = np.concatenate([self.robot.get_arm_joints(), self.robot.get_hand_joints()])
            try:
                q_cmd = self.planner.step(q, pos, rpy, hand_target, self._world(exclude_object=not grasp_task))
            except Exception as e:  # keep servoing on transient solver failures
                self.log(f"[align] planner error: {e}")
                rate.sleep()
                continue
            self.robot.send_arm_joints(q_cmd[:7])
            self.robot.send_hand(q_cmd[7:])
            rate.sleep()
        return last

    def run_skill(self, name: str, pos_seq, rot_seq, hand_seq, theta_star, grasp_task: bool) -> bool:
        d = self.cfg.deploy
        hand0 = hand_seq[0] if len(hand_seq) > 0 else None

        self.robot.set_control_mode("joint")
        res = self.align(name, pos_seq[0], rot_seq[0], hand0, theta_star, grasp_task)
        if not res.ok:
            return False

        # Freeze the object pose for the rest of the skill.
        self.robot.set_control_mode("pose")
        if not run_bridge(self.robot, self.send_pose, res.center, res.yaw, pos_seq[0], rot_seq[0], hand0,
                          seconds=d.bridge_seconds, ctrl_hz=d.ctrl_hz, early_stop_fn=self.early_stop_fn):
            return False
        return run_replay(self.robot, self.send_pose, res.center, res.yaw, pos_seq, rot_seq, hand_seq,
                          ctrl_hz=d.ctrl_hz, early_stop_fn=self.early_stop_fn)

    # ------------------------------------------------------------------
    def run_once(self) -> bool:
        rc = self.cfg.retrieval
        obj_state = self.perception.get_state(OBJECT)
        dr_state = self.perception.get_state(DRAWER)
        if obj_state is None or dr_state is None:
            raise RuntimeError("Object states are not available yet.")

        # ---- skill 1: retrieve once over both objects ----
        pos1, rot1, hand1, idx, score = self.policy.retrieve_object(
            query_pc_world=self.perception.get_cloud(OBJECT),
            drawer_pc_world=self.perception.get_cloud(DRAWER),
            max_points=rc.max_points,
            obj_yaw_observable=obj_state.yaw_observable,
            drawer_yaw_observable=dr_state.yaw_observable,
            yaw_task=obj_state.yaw,
            lambda_drawer=rc.lambda_drawer,
        )
        self.log(f"[dexmulti] retrieved demo {idx} ({self.policy.episodes[idx]}), score={score:.6f}, "
                 f"object yaw observable={obj_state.yaw_observable}")
        if not self.run_skill(OBJECT, pos1, rot1, hand1, self.policy.obj_theta_star, grasp_task=True):
            return False

        # ---- skill 2: same demo; re-estimate theta* if needed ----
        dr_state = self.perception.get_state(DRAWER)
        pos2, rot2, hand2 = self.policy.retrieve_drawer(
            self.perception.get_cloud(DRAWER), max_points=rc.max_points,
            yaw_observable=dr_state.yaw_observable,
        )
        ok = self.run_skill(DRAWER, pos2, rot2, hand2, self.policy.drawer_theta_star, grasp_task=False)
        self.log("[dexmulti] done." if ok else "[dexmulti] stopped.")
        return ok
