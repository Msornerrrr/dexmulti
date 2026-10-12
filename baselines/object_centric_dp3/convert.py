#!/usr/bin/env python3
"""
Convert released DexMulti episodes (*.npz) into the zarr ReplayBuffer used to train
the object-centric DP3 baseline.

Reproduces the original two-step pipeline (dex_controls `converters/dp3_convert_multi.py`
-> pkl -> iDP3 fork `convert.py` -> zarr) in one step. Per episode it stores:

    object_pc          (T, 512, C)  float32   C=3 (xyz) or 4 (xyzc)
    drawer_pc          (T, 512, C)  float32   second object (drawer / container / 2nd object)
    eef_pose           (T, 6)       float32   npz `eef_pose`   (xyz + euler xyz, base frame)
    eef_pose_target    (T, 6)       float32   npz `arm_cmd`
    hand_joint         (T, 16)      float32   npz `hand_state`
    hand_joint_target  (T, 16)      float32   npz `hand_cmd`
    meta/episode_ends

`eef_rot9d` and the 28-D action [pos3, rot9 (row-major), hand16] are derived from these at
load time by the dataset (idp3/diffusion_policy_3d/dataset/custom_dataset.py).

Variants (names used by the original pipeline):
    xyz   (v1, task=object_centric_xyz):  fused_segmented_point_cloud_base_{object,drawer}
    xyzc  (v3, task=object_centric_xyzc): fused_segmented_point_cloud_base_{object,drawer}
                                          _to_hand_point_cloud_base_xyzc_contact_cloud

Episodes are read from each folder's top-level *.npz in sorted filename order (as the
original); folders are concatenated in the order given. Other npz keys are ignored.

Example:
    python convert.py --variant xyzc --out data/grasp_pull_xyzc.zarr \
        --episodes-dir <release_data>/main/grasp_pull
"""
import argparse
import pathlib
import shutil
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "idp3"))
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer  # noqa: E402

_OBJ = "fused_segmented_point_cloud_base_object"
_DRAWER = "fused_segmented_point_cloud_base_drawer"
_XYZC = "_to_hand_point_cloud_base_xyzc_contact_cloud"

# variant -> [(zarr key, npz key)]  (pc_out_names / pc_keys of the original converter)
PC_KEYS = {
    "xyz": [("object_pc", _OBJ), ("drawer_pc", _DRAWER)],
    "xyzc": [("object_pc", _OBJ + _XYZC), ("drawer_pc", _DRAWER + _XYZC)],
}
PC_KEYS["v1"], PC_KEYS["v3"] = PC_KEYS["xyz"], PC_KEYS["xyzc"]

# zarr key -> npz key
LOW_DIM_KEYS = [
    ("eef_pose", "eef_pose"),
    ("eef_pose_target", "arm_cmd"),
    ("hand_joint", "hand_state"),
    ("hand_joint_target", "hand_cmd"),
]


def list_episodes(folders, episode_list=None):
    files = []
    for folder in folders:
        folder = pathlib.Path(folder)
        if not folder.is_dir():
            raise NotADirectoryError(folder)
        found = sorted(f for f in folder.iterdir() if f.suffix == ".npz")
        if not found:
            raise RuntimeError(f"No .npz files found in {folder}")
        files.extend(found)
    if episode_list is not None:
        wanted = {pathlib.Path(x).stem for x in episode_list}
        files = [f for f in files if f.stem in wanted]
        missing = wanted - {f.stem for f in files}
        if missing:
            raise FileNotFoundError(f"episodes not found: {sorted(missing)}")
    return files


def load_episode(path, variant):
    with np.load(path, allow_pickle=False) as d:
        T = d["time"].shape[0]
        if T == 0:
            return None
        ep = {}
        for out_key, npz_key in PC_KEYS[variant]:
            if npz_key not in d.files:
                raise KeyError(f"{path.name}: missing point cloud key '{npz_key}'")
            ep[out_key] = d[npz_key].astype(np.float32)
        for out_key, npz_key in LOW_DIM_KEYS:
            ep[out_key] = d[npz_key].astype(np.float32)
    for k, v in ep.items():
        if v.shape[0] != T:
            raise ValueError(f"{path.name}: {k} has length {v.shape[0]} != T={T}")
    return ep


def convert(files, variant, out_path, overwrite=False, verbose=True):
    out_path = pathlib.Path(out_path)
    if out_path.suffix != ".zarr":
        raise ValueError(f"--out must end in .zarr, got {out_path}")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists (use --overwrite)")
    if out_path.exists() and not (out_path / ".zgroup").exists():
        raise ValueError(f"refusing to overwrite {out_path}: not a zarr store")
    rb = ReplayBuffer.create_empty_zarr()
    for i, f in enumerate(files):
        ep = load_episode(f, variant)
        if ep is None:
            print(f"[convert] skipping empty episode {f.name}")
            continue
        rb.add_episode(ep)
        if verbose:
            print(f"[convert] {i + 1}/{len(files)} {f.name} (T={len(ep['eef_pose'])})")
    if out_path.exists():  # only reached with overwrite=True
        shutil.rmtree(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rb.save_to_path(str(out_path))  # same chunking / compression as the original convert.py
    print(f"[convert] saved {rb.n_episodes} episodes / {rb.n_steps} steps -> {out_path}")
    return rb


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes-dir", nargs="+", required=True, help="folder(s) of episode .npz files")
    p.add_argument("--variant", choices=sorted(PC_KEYS), required=True)
    p.add_argument("--out", required=True, help="output .zarr directory")
    p.add_argument("--episode-list", help="optional text file, one episode name (stem or filename) per line")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    episode_list = None
    if args.episode_list:
        lines = pathlib.Path(args.episode_list).read_text().splitlines()
        episode_list = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
    files = list_episodes(args.episodes_dir, episode_list)
    convert(files, args.variant, args.out, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
