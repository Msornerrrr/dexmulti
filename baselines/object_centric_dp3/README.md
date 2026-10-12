# Object-centric DP3 baseline

This is the "object-centric DP3" baseline from Table I of the paper. It is a 3D diffusion
policy (iDP3-style UNet, DDIM, 10 inference steps) conditioned on:

- the two segmented object point clouds `object_pc` and `drawer_pc`, 512 points each, in the robot base frame;
- the end-effector position and rotation matrix (`eef_pos`, `eef_rot9d`);
- the 16 LEAP hand joints (`hand_joint`).

The policy uses `n_obs_steps=3` observations. It predicts `horizon=16` absolute actions
`[pos(3), rot9(9, row-major), hand(16)]` (28-D) and executes the first 15 at 10 Hz. The
predicted `rot9` is projected onto SO(3) with an SVD.

```
convert.py        released episode .npz  ->  zarr ReplayBuffer (training data)
idp3/train.py     train (hydra)
idp3/serve.py     serve a checkpoint over websocket (msgpack_numpy)
client.py         ROS-free client, obs packing / action unpacking, async action-chunk runner
idp3/             vendored subset of the Improved-3D-Diffusion-Policy fork (MIT, see idp3/LICENSE)
```

## Attribution and modifications

`idp3/` is a trimmed copy of a lab-internal fork of
[Improved-3D-Diffusion-Policy](https://github.com/YanjieZe/Improved-3D-Diffusion-Policy)
(Copyright (c) 2024 Yanjie Ze, MIT License, `idp3/LICENSE`). The fork added multi-point-cloud
encoding, per-key or shared encoders, the `multi_stage_pointnet` and `better_attention_pointnet`
encoders, SE(3) point cloud augmentation, the 9-D rotation action with SO(3) projection, and the
websocket server.

This release makes the following changes to the fork. None of them change the model, the
losses or the hyper-parameters.

- **Removed** files the baseline does not use: GR1 datasets, configs and action utils;
  `deploy.py`; the image policy (`dp_workspace`, timm/R3M encoders, crop randomizer, model getter);
  the egocentric dataset; the RealSense wrapper; the attention, seg and attention-seg PointNet++
  encoders (`third_party/Pointnet2_PyTorch` is therefore not needed); `third_party/r3m` and the
  visualizer; `vis_dataset.py`; the experiment and GPU-scheduler scripts; the other task configs.
- **Paths**: `task.dataset.zarr_path` is absolute or relative to the launch directory. The fork
  used `"../../../" + zarr_path`. `train.py` no longer `chdir`s to the repo root.
- **Configs**: `multi_pc_segmented.yaml` is now `task/object_centric_xyz.yaml` and
  `multi_pc_segmented_v3.yaml` is now `task/object_centric_xyzc.yaml`. Their content is unchanged.
  The `idp3.yaml` defaults changed: the task now defaults to `object_centric_xyzc` (the fork's
  default pointed to a missing task), `logging.mode=disabled` (wandb is optional),
  `checkpoint.save_ckpt=True` (as in all paper runs), and outputs go to `outputs/`.
- **Serving**: the hydra-based `eval.py` is replaced by `serve.py --ckpt`, which reads the
  config from the checkpoint. The server protocol is unchanged, except that the metadata
  `device` reports the device actually used.
- **Small fixes**: `torch.load(..., weights_only=False)` for torch >= 2.6; no `pdb` call in the
  sampler; an explicit hydra `version_base="1.1"`, which keeps the run-dir behaviour the paper
  runs had.

Module paths are unchanged (`diffusion_policy_3d.*`), so checkpoints from the original fork load
directly. We checked that the vendored model's state-dict layout matches the paper checkpoints
for all four encoder/sharing combinations.

## Install

```bash
conda create -n dexmulti_dp3 python=3.10 -y && conda activate dexmulti_dp3
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128   # or the build for your CUDA
pip install -r requirements.txt
```

`requirements.txt` pins the versions of the environment that trained the paper models
(python 3.8, torch 2.4.1+cu121). The code was also tested with python 3.10 and torch 2.7.1+cu128.

## Convert

The converter reads episodes in the npz layout. First convert a released LeRobot dataset from the repo root
(see [docs/dataset.md](../../docs/dataset.md)), e.g.

```bash
python scripts/lerobot_to_npz.py --dataset data/lerobot/dexmulti_grasp_pull --out data/npz/grasp_pull
python scripts/lerobot_to_npz.py --dataset data/lerobot/dexmulti_grasp_open_sample_efficiency --split quantity_5 --out data/npz/quantity_5
```

Then run all commands from this directory with `DATA=../../data/npz`.

```bash
# v3 / xyzc: xyz + hand-contact channel (512x4)
python convert.py --variant xyzc --episodes-dir $DATA/grasp_pull --out data/grasp_pull_xyzc.zarr
# v1 / xyz: xyz only (512x3)
python convert.py --variant xyz  --episodes-dir $DATA/grasp_pull --out data/grasp_pull_xyz.zarr
```

`--episode-list file.txt` (one episode name per line) restricts the conversion to a subset.
Episodes are always added in sorted filename order. Several `--episodes-dir` values are
concatenated in the given order.

The converter does in one step what the original pipeline did in two
(`dex_controls/converters/dp3_convert_multi.py` -> pkl -> fork `convert.py` -> zarr). For all 13
paper training sets (the 3 main splits in xyz and xyzc, and the 7 sample-efficiency splits in
xyzc), it produces arrays identical to the zarrs the paper models were trained on.

| zarr key | source npz key | |
|---|---|---|
| `object_pc` | `fused_segmented_point_cloud_base_object` (xyz) or `..._object_to_hand_point_cloud_base_xyzc_contact_cloud` (xyzc) | float32 |
| `drawer_pc` | `fused_segmented_point_cloud_base_drawer` (xyz) or `..._drawer_to_hand_point_cloud_base_xyzc_contact_cloud` (xyzc) | float32 |
| `eef_pose` | `eef_pose` | obs: `eef_pos` = [:3], `eef_rot9d` = euler-xyz -> 3x3, flattened row-major |
| `eef_pose_target` | `arm_cmd` | action [0:12] = pos + rot9 |
| `hand_joint` | `hand_state` | obs |
| `hand_joint_target` | `hand_cmd` | action [12:28] |

## Train

```bash
python idp3/train.py task=object_centric_xyzc data_name=grasp_pull_xyzc \
    hydra.run.dir=outputs/grasp_pull_xyzc_multiStage_shared exp_name=grasp_pull_xyzc_multiStage_shared \
    training.seed=42 training.device=cuda:0 \
    pc_encoder=multi_stage_pointnet policy.pc_encoder_sharing=shared \
    horizon=16 n_action_steps=15 n_obs_steps=3 \
    policy.se3_augmentation_cfg.use_aug=true policy.use_fourier_basis=false
```

`data_name=X` reads `data/X.zarr`. You can instead pass `task.dataset.zarr_path=/abs/path.zarr`.
Training runs 301 epochs, batch size 64, and saves `checkpoints/latest.ckpt` every 100 epochs.
`val_ratio=0.02` holds out `max(1, round(0.02 N))` episodes, i.e. **exactly one episode** for
every released split (seed 42, sorted filename order). That episode is not trained on and is
never scored: the logged "val" losses are computed on training batches, as in the original
fork. The paper's baseline runs therefore trained on N-1 demos (e.g. 4 for `quantity_5`, 43 for
the 44-demo diversity splits). Pass `task.dataset.val_ratio=0` to use all episodes (this
differs from the paper runs). Add `logging.mode=online` for wandb.

### Configurations

| this release | original fork / data name | point clouds |
|---|---|---|
| `task=object_centric_xyz` + `convert.py --variant xyz` | `task=multi_pc_segmented`, data `*_v1` | `object_pc`, `drawer_pc`: 512x3 xyz |
| `task=object_centric_xyzc` + `convert.py --variant xyzc` | `task=multi_pc_segmented_v3`, data `*_v3` | `object_pc`, `drawer_pc`: 512x4 xyz + contact |

- `pc_encoder`: `multi_stage_pointnet` (iDP3 multi-stage PointNet with max-pooling) or
  `better_attention_pointnet` (per-point MLP with learned attention pooling). Both output 256 features per cloud.
- `policy.pc_encoder_sharing`: `shared` (one encoder for both clouds) or `per_key` (one encoder per cloud).
  The two cloud features are concatenated with a 64-D state feature, giving 576 dims per observation step.

The fork's `multi_pc_segmented_v2` variant (which added the hand cloud) is not included.

### Paper runs

Every object-centric DP3 run used this command (fork names; from the runs' `.hydra/overrides.yaml`), with `<task>`, `<data>`, `<enc>` and `<sharing>` filled in:

```
python -m train --config-name=idp3.yaml task=<task> hydra.run.dir=output/<date>/<exp>_seed42
  training.debug=False training.seed=42 training.device=cuda:0 exp_name=<exp>
  logging.mode=online checkpoint.save_ckpt=True pc_encoder=<enc>
  horizon=16 n_action_steps=15 n_obs_steps=3 policy.se3_augmentation_cfg.use_aug=true
  policy.use_fourier_basis=false policy.pc_encoder_sharing=<sharing> data_name=<data>
```

| released data (`--split`) | original data name | runs trained |
|---|---|---|
| `dexmulti_grasp_pull` | `2025-12-27-grasp-pull_{v1,v3}` | {xyz, xyzc} x {multi_stage, better_attention} x {shared, per_key} |
| `dexmulti_grasp_open` | `2025-12-28-grasp-open_{v1,v3}` | same 8 |
| `dexmulti_grasp_grasp` | `2025-12-26-grasp-grasp_{v1,v3}` | same 8 |
| `dexmulti_grasp_open_sample_efficiency` (`quantity_{5,10,20,50}`) | `2026-01-28-grasp-open-coconut{5,10,20,50}_v3` | xyzc x {multi_stage, better_attention} x shared |
| `dexmulti_grasp_open_sample_efficiency` (`diversity_{11obj_x4,4obj_x11,7obj_x6+2}`) | `2026-01-28-grasp-open-{11*4,4*11,7*6+2}_v3` | xyzc x {multi_stage, better_attention} x shared |

We did not record which of these variants produced the object-centric DP3 numbers in the paper; all
of them were trained with the command above.

For the three main datasets, `lerobot_to_npz.py` + `convert.py` reproduce the paper's training data
bit-exactly. For the sample-efficiency sets, the per-frame point clouds come from one canonical copy
of each recording, while the paper's runs used each set's own (stochastically re-processed) copy, so
the clouds differ slightly; robot data and actions are identical.

## Serve

```bash
python idp3/serve.py --ckpt outputs/grasp_pull_xyzc_multiStage_shared/checkpoints/latest.ckpt --port 8000
curl localhost:8000/healthz      # -> OK
python client.py --uri ws://localhost:8000 --pc-channels 4   # round-trips a dummy observation
```

The server uses the EMA weights. On connect it sends
`{n_obs_steps, n_action_steps, n_latency_steps, device}`.

## Deploy

`client.py` needs only `numpy scipy websockets msgpack-numpy`; its `make_xyzc` helper (xyzc policies) additionally needs the core package with `pip install -e ".[perception]"` (Python >= 3.10) in the robot-side environment. The robot object needs
`get_eef_pose()`, `get_hand_joints()`, `send_eef_pose(pos, rpy)` and `send_hand(q)`, which are a
subset of `dexmulti.interfaces.RobotInterface` (use pose control mode). You also pass a
callback that returns the current base-frame point clouds:

```python
import asyncio
from client import DP3Client, AsyncChunkRunner

async def main(robot, get_point_clouds):
    client = DP3Client("ws://gpu-host:8000")
    await client.connect()
    runner = AsyncChunkRunner.from_meta(client, robot, get_point_clouds, control_hz=10)
    await runner.run(rollout_len=600)
    await client.close()

# get_point_clouds() -> {"object_pc": (512, C), "drawer_pc": (512, C)}
asyncio.run(main(robot, get_point_clouds))
```

Object clouds must be 512 points per object, as in training (`farthest_point_sample` in
`dexmulti/perception/denoise.py`). For `xyzc` policies, the 4th channel is the soft
hand-object contact map, computed online exactly as in the data: `make_xyzc(object_cloud,
hand_cloud)` in `client.py`. It needs a hand surface cloud equivalent to the data's
`hand_point_cloud_base`: 512 points sampled on the LEAP hand link meshes and posed by forward
kinematics. A hand-cloud sampler is not part of this release, so if you cannot produce such a
cloud use the `xyz` variant. Poses use euler `"xyz"` in radians. The
runner sends absolute targets; clip them to your workspace and velocity limits in `send_eef_pose`.

The runner follows the original asynchronous executor. It executes the current 15-step chunk
at 10 Hz and requests the next chunk `n_latency_steps` (1) steps before the current chunk ends.
Chunks are swapped at the boundary; if the next chunk is late, the last action is held.
Observations are stacked over the last `n_obs_steps`, repeating the first frame at startup.
