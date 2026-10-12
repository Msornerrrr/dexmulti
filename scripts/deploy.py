#!/usr/bin/env python3
"""Run DexMulti on the robot (one trial).

  python scripts/deploy.py --library data/lerobot/dexmulti_grasp_pull/dexmulti/library/dual_task_policy_chamfer.npz

Uses the ROS1 adapter in robots/xarm_leap_ros1 (the setup used in the paper).
For other robots, implement dexmulti.interfaces.RobotInterface / Perception and
construct DexMultiRunner the same way.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from dexmulti.alignment.pyroki_planner import PlannerWeights, PyrokiAlignmentPlanner
from dexmulti.alignment.world import build_world
from dexmulti.config import DexMultiConfig
from dexmulti.execution import Rate
from dexmulti.retrieval import DualTaskPolicyChamfer
from dexmulti.runner import DexMultiRunner


def move_to_start(robot, cfg: DexMultiConfig, seconds: float = 3.0):
    """Stream the start pose (unclamped, as in the experiments) for a few seconds."""
    robot.set_control_mode("pose")
    rate, t0 = Rate(20.0), time.monotonic()
    while time.monotonic() - t0 < seconds:
        robot.send_eef_pose(np.array(cfg.deploy.start_pos), np.array(cfg.deploy.start_rpy))
        robot.send_hand(np.array(cfg.deploy.start_hand))
        rate.sleep()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--planner-weights", choices=["jan2026", "dec2025"], default="jan2026",
                   help="alignment planner cost preset, see PlannerWeights")
    p.add_argument("--baseline-tracker", action="store_true", help="raw PCA yaw ablation (Table II)")
    p.add_argument("--no-start-pose", action="store_true")
    a = p.parse_args()

    if not a.library.exists():
        p.error(f"--library {a.library} does not exist")
    import rospy

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from robots.xarm_leap_ros1.adapter import RosPerception, XArmLeapRobot

    rospy.init_node("dexmulti_deploy")
    cfg = DexMultiConfig()
    robot = XArmLeapRobot()
    perception = RosPerception(cfg.tracker, baseline_tracker=a.baseline_tracker)
    planner = PyrokiAlignmentPlanner(weights=getattr(PlannerWeights, a.planner_weights)())
    policy = DualTaskPolicyChamfer(a.library, num_yaw_bins=cfg.retrieval.num_yaw_bins)

    # Compile the planner for the run-time world size (2 planes + K spheres per object).
    obj, dr = perception.get_state("object"), perception.get_state("drawer")
    planner.warmup(build_world(obj.sphere_centers, obj.sphere_radii, dr.sphere_centers, dr.sphere_radii,
                               exclude_object=False, cfg=cfg.world))

    if not a.no_start_pose:
        move_to_start(robot, cfg)
    runner = DexMultiRunner(robot, perception, policy, planner, cfg,
                            early_stop_fn=rospy.is_shutdown, log=rospy.loginfo)
    runner.run_once()


if __name__ == "__main__":
    main()
