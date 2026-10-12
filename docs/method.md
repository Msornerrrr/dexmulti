# Method: paper to code

This page shows where each part of the paper lives in the code. [design.md](design.md) explains
why each part is designed the way it is and what to change when porting. The section, equation,
figure and table numbers below follow the paper (arXiv:2603.11655). Deployment constants are in
[dexmulti/config.py](../dexmulti/config.py). Keyframe, planner-weight, denoising, tracker and
envelope constants are in [keyframes.py](../dexmulti/data/keyframes.py) (`KeyframeConfig`),
[pyroki_planner.py](../dexmulti/alignment/pyroki_planner.py) (`PlannerWeights`),
[denoise.py](../dexmulti/perception/denoise.py) (`CleanConfig`, `SEGMENTED`),
[tracker.py](../dexmulti/perception/tracker.py) (EKF noise and hysteresis defaults) and
[envelope.py](../dexmulti/perception/envelope.py) (`MultiSphereEnvelope`).

Terminology: the paper's objects $o_1, o_2$ are the **first object** (grasped in skill 1) and the
**second object** (drawer, container lid or second grasped object, manipulated in skill 2). In code
and in the data they are `object` and `drawer`, matching the recorded data keys.

## Lookup table

| Paper | Code |
|---|---|
| Overall pipeline (Fig. 2, Alg. 1) | [dexmulti/runner.py](../dexmulti/runner.py) `DexMultiRunner.run_once`; entry point [scripts/deploy.py](../scripts/deploy.py) |
| Perception (Sec. IV-A, Appendix A "Perception Pipeline"): masks, back-projection, crop, random downsample, SOR, ROR, DBSCAN, FPS to 512 points | [dexmulti/perception/denoise.py](../dexmulti/perception/denoise.py), a CPU reference of the GPU ROS nodes. Grounding DINO + SAM2 are not included, and the released data already contains the cleaned clouds |
| Hand point cloud from forward kinematics (Sec. IV-B, Appendix A "Skill Segmentation") | provided in the data (`observation.pointcloud.hand`). The sampler is not part of the release |
| Soft hand-object contact signal (Sec. IV-B) | [dexmulti/perception/contact.py](../dexmulti/perception/contact.py) `contact_map`. The data also includes the contact maps |
| Skill segmentation at sustained contact (Sec. IV-B) | [dexmulti/data/keyframes.py](../dexmulti/data/keyframes.py) `segment_episode`; [scripts/segment_keyframes.py](../scripts/segment_keyframes.py) |
| Canonical skill representation, Eq. (1) | [dexmulti/library.py](../dexmulti/library.py) `build_library`; [dexmulti/geometry.py](../dexmulti/geometry.py) `estimate_yaw`, `Rz`; [scripts/build_library.py](../scripts/build_library.py) |
| Object-centric state estimation, EKF and yaw observability with hysteresis (Sec. IV-C, Appendix A "EKF State Representation") | [dexmulti/perception/tracker.py](../dexmulti/perception/tracker.py) `ObjectFrameTracker`; 30 Hz loop with collision spheres in [dexmulti/perception/state.py](../dexmulti/perception/state.py) `ObjectStateEstimator` |
| Raw PCA yaw ablation (Table II) | `ObjectFrameTrackerBaseline` in the same file; `scripts/deploy.py --baseline-tracker` |
| Retrieval, Eq. (2), joint over both objects (Alg. 1 line 4) | [dexmulti/retrieval.py](../dexmulti/retrieval.py) `DualTaskPolicyChamfer.retrieve_object`; θ* for skill 2: `retrieve_drawer` |
| Canonical-to-world mapping, Eq. (3) | [dexmulti/geometry.py](../dexmulti/geometry.py) `canon_action_to_world` |
| Yaw selection rule during alignment, Eq. (4) (Appendix A) | [dexmulti/alignment/targets.py](../dexmulti/alignment/targets.py) `alignment_yaw` (the tracker holds the task yaw while yaw is unobservable); PCA sign: `geometry.resolve_yaw_sign_by_world_action` |
| Closed-loop alignment with Pyroki (Sec. IV-D) | [dexmulti/alignment/pyroki_planner.py](../dexmulti/alignment/pyroki_planner.py) `PyrokiAlignmentPlanner`; loop in `DexMultiRunner.align` |
| Multi-sphere collision envelope (Appendix A "Collision Representation") | [dexmulti/perception/envelope.py](../dexmulti/perception/envelope.py) `MultiSphereEnvelope`; planner world in [dexmulti/alignment/world.py](../dexmulti/alignment/world.py) `build_world` |
| Pose interpolation, then open-loop execution in the frozen object frame (Sec. IV-D "Execution") | [dexmulti/execution.py](../dexmulti/execution.py) `run_bridge`, `run_replay` |
| Arm command limits: 0.1 m/s, 25°/s, 10 Hz (Sec. V-A, Appendix C) | `execution.clip_and_clamp`, `config.ArmLimits`, `config.DeployConfig.ctrl_hz` |
| Robot and perception interface | [dexmulti/interfaces.py](../dexmulti/interfaces.py); paper setup in [robots/xarm_leap_ros1](../robots/xarm_leap_ros1) ([adapting.md](adapting.md#5-robot-interface)) |
| Demonstrations (Sec. V-A, Appendix C "Demonstration Recording") | released LeRobot datasets ([dataset.md](dataset.md)); reader and npz export in [dexmulti/data/lerobot.py](../dexmulti/data/lerobot.py) `LeRobotEpisodes`, `materialize`; npz episode format in [dexmulti/data/episode.py](../dexmulti/data/episode.py); scripts [download_data.py](../scripts/download_data.py), [lerobot_to_npz.py](../scripts/lerobot_to_npz.py), [visualize_demo.py](../scripts/visualize_demo.py) |
| Object-centric DP3 baseline (Sec. V-A, Appendix B) | [baselines/object_centric_dp3](../baselines/object_centric_dp3) |
| Table I, Table II, Fig. 6 settings | [reproducing.md](reproducing.md) |

## Details that are easy to miss

[design.md](design.md) gives the reasons for each of these.

- **One retrieval per episode.** Skill 2 reuses the demonstration retrieved before skill 1.
  Only its yaw anchor θ* is recomputed right before skill 2, and only if the second
  object's yaw is unobservable. The joint score is $d_\text{first} + 0.3\, d_\text{second}$
  (`RetrievalConfig.lambda_drawer`).
- **Skills start at contact.** A skill begins at the first frame of the longest run of sustained hand
  contact. Skill 1 ends when the object has been lifted, and skill 2 runs to the end of the demo. The
  demo's approach and the motion between the two skills are never replayed. Alignment replaces them.
- **What is replayed.** The commanded end-effector poses (`arm_cmd`) and hand commands
  (`hand_cmd`) of the skill are replayed, not the measured states. The first object stays in the hand during
  skill 2 because the replayed hand commands keep the grasp.
- **Alignment is time-boxed.** Each skill aligns for a fixed 10 s, then bridges for 3 s, then
  replays. The frozen object pose is the last tracked pose at the end of the alignment window.
- **Joint vs. Cartesian commands.** Alignment streams planner joint targets. The bridge and replay stream
  Cartesian poses, which pass through `clip_and_clamp` (workspace box, 0.1 m/s, 25°/s).
  Because of the clamp, a replay can fall behind a demo that moved faster than these limits.
- **Yaw sign.** PCA yaw is defined up to π. The alignment picks the sign under which the
  commanded end-effector yaw is closest to 0 (hand facing forward) among those within 120°,
  falling back to the smaller one if both exceed it.
- **Collision world during alignment.** The world has a ground plane at z = 0.03 m and a wall at
  x = 0.68 m (robot base frame), plus 6 spheres per object. The first object is an obstacle
  while the hand approaches it. Once it is held, it is replaced by far-away placeholder spheres, which
  keeps the size of the JIT-compiled problem fixed.
- **Hand during alignment.** In skill 1 the hand is open for the first 75 % of the 10 s window,
  then it closes to the demo's first hand command. In skill 2 the hand tracks the first skill-2 hand
  command throughout the window.
