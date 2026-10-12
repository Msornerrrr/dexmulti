# xArm7 + LEAP Hand, ROS1

Interface for the setup used in the paper's experiments. It assumes these drivers and topics:

| | |
|---|---|
| arm | UFACTORY xArm7, `xarm_ros` driver (ROS Noetic). Alignment streams joint targets in mode 6 (joint online trajectory planning, `/xarm/move_joint`); bridge and replay stream Cartesian targets in mode 7 (Cartesian online trajectory planning, `/xarm/move_line`) |
| hand | LEAP Hand ROS node: commands on `/leaphand_node/cmd_leap` (10 Hz from DexMulti, interpolated to 40 Hz here), state on `/leap_hand_state` |
| perception | any stack publishing segmented, fused object clouds in the arm base frame on `/perception/object_fused_segmented_point_cloud` and `/perception/drawer_fused_segmented_point_cloud`. We used Grounding DINO + SAM2 masks on two RealSense L515 cameras, cleaned and fused with the chain in `dexmulti/perception/denoise.py` (GPU version), at ~20 Hz |

`RosPerception` runs the DexMulti object tracker (`ObjectStateEstimator`) on those clouds at
30 Hz; during the experiments this ran as a separate node with identical logic.

**Environment.** `deploy.py` runs the planner in the same process as `rospy`. The planner
pins jax 0.8.1, which needs Python ≥ 3.11, so a stock Noetic install (Python 3.8) does
not work. Use a conda ROS1 Noetic build for Python 3.11 (e.g. RoboStack), source the catkin
workspace that provides `xarm_msgs` (built from `xarm_ros`), and install `dexmulti[planner]`
into that environment.

```bash
python scripts/deploy.py --library data/lerobot/dexmulti_grasp_pull/dexmulti/library/dual_task_policy_chamfer.npz
```

`scripts/deploy.py` waits for both object states, compiles the planner (~18 s on CPU), moves to the start pose, and runs one trial.
Add `--baseline-tracker` for the raw-PCA-yaw ablation.
