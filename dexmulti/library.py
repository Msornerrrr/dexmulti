"""Build the canonical skill library from segmented demonstrations (offline).

For every demonstration episode and each of its two skills (k=1: first object,
k=2: second object / "drawer"):

1. take the object point cloud at the skill keyframe,
2. compute centroid c and canonical yaw psi (PCA on XY),
3. store the canonical cloud  Rz(-psi) (P - c),
4. express every EE pose of the skill segment in that object frame:
   p_canon = Rz(-psi)(p - c),  R_canon = Rz(-psi) R.
   Hand joint commands are stored unchanged.

The result is saved as one ``.npz`` bundle consumed by :class:`dexmulti.retrieval.DualTaskPolicyChamfer`.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from dexmulti.config import DRAWER_PC_KEY, OBJ_PC_KEY
from dexmulti.geometry import Rz, centroid, estimate_yaw, flatten_pc, random_downsample

LIBRARY_FILENAME = "dual_task_policy_chamfer.npz"


def _pose6_to_pos_rot(pose6: np.ndarray):
    arr = np.asarray(pose6)
    pos = arr[:3].astype(np.float32)
    if arr.shape[0] >= 6:
        rot = R.from_euler("xyz", arr[3:6]).as_matrix().astype(np.float32)
    else:
        rot = np.eye(3, dtype=np.float32)
    return pos, rot


def _canonicalize_skill(pc_seq, arm_cmd, hand_cmd, start: int, end: int, pc_downsample: int):
    """Returns (pc_centered, pc_canon, yaw, pos_seq, rot_seq, hand_seq) or None."""
    if start >= len(pc_seq):
        return None

    pc = flatten_pc(pc_seq[start])
    if pc.size == 0 or pc.shape[0] < 10:
        return None

    c = centroid(pc).astype(np.float32)
    pc_centered = (pc - c).astype(np.float32)
    yaw = estimate_yaw(pc_centered)
    R_inv = Rz(-yaw)

    pc_canon = (pc_centered @ R_inv.T).astype(np.float32)
    pc_canon = random_downsample(pc_canon, pc_downsample, seed=0)

    pos_seq, rot_seq, hand_seq = [], [], []
    for t in range(start, end + 1):
        p_w, R_w = _pose6_to_pos_rot(arm_cmd[t])
        pos_seq.append((R_inv @ (p_w - c)).astype(np.float32))
        rot_seq.append((R_inv @ R_w).astype(np.float32))
        hand_seq.append(np.asarray(hand_cmd[t]) if hand_cmd is not None else None)

    if len(pos_seq) == 0:
        return None
    return pc_centered, pc_canon, yaw, pos_seq, rot_seq, hand_seq


def build_library(
    session_root: Path,
    keyframe_dir: Optional[Path] = None,
    obj_pc_key: str = OBJ_PC_KEY,
    drawer_pc_key: str = DRAWER_PC_KEY,
    pc_downsample: int = 6000,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Build the library bundle from ``<session_root>/*.npz`` and per-episode keyframes
    (keys ``task{1,2}_{start,end}``), read from ``<keyframe_dir>/<episode>.json``
    (released layout, default ``keyframes/``) or ``<keyframe_dir>/<episode>/metadata.json``
    (original recording layout, ``segmented_tasks/``)."""
    session_root = Path(session_root)
    if keyframe_dir is None:
        keyframe_dir = session_root / "keyframes"
        if not keyframe_dir.exists():
            keyframe_dir = session_root / "segmented_tasks"
    keyframe_dir = Path(keyframe_dir)

    out = {k: [] for k in (
        "obj_pc_centered", "obj_pc_canon", "obj_yaws", "obj_pos", "obj_rot", "obj_hand",
        "drawer_pc_centered", "drawer_pc_canon", "drawer_yaws", "drawer_pos", "drawer_rot", "drawer_hand",
        "episodes", "task1_starts", "task1_ends", "task2_starts", "task2_ends",
    )}

    npz_files = sorted(session_root.glob("*.npz"))
    if verbose:
        print(f"[build] {len(npz_files)} episodes in {session_root}")

    for f in npz_files:
        ep = f.stem
        meta_path = keyframe_dir / f"{ep}.json"
        if not meta_path.exists():
            meta_path = keyframe_dir / ep / "metadata.json"
        if not meta_path.exists():
            continue

        meta = json.loads(meta_path.read_text())
        ks = [meta.get(k) for k in ("task1_start", "task1_end", "task2_start", "task2_end")]
        if None in ks:
            continue
        k1s, k1e, k2s, k2e = map(int, ks)

        data = np.load(f, allow_pickle=False)
        if obj_pc_key not in data.files or drawer_pc_key not in data.files:
            continue
        arm_cmd = data["arm_cmd"] if "arm_cmd" in data.files else data["eef_pose"]
        hand_cmd = data["hand_cmd"] if "hand_cmd" in data.files else None

        s1 = _canonicalize_skill(data[obj_pc_key], arm_cmd, hand_cmd, k1s, k1e, pc_downsample)
        if s1 is None:
            continue
        s2 = _canonicalize_skill(data[drawer_pc_key], arm_cmd, hand_cmd, k2s, k2e, pc_downsample)
        if s2 is None:
            continue

        for prefix, (pc_centered, pc_canon, yaw, pos_seq, rot_seq, hand_seq) in (("obj", s1), ("drawer", s2)):
            out[f"{prefix}_pc_centered"].append(pc_centered)
            out[f"{prefix}_pc_canon"].append(pc_canon)
            out[f"{prefix}_yaws"].append(float(yaw))
            out[f"{prefix}_pos"].append(np.stack(pos_seq, axis=0))
            out[f"{prefix}_rot"].append(np.stack(rot_seq, axis=0))
            out[f"{prefix}_hand"].append(_hand_array(hand_seq))

        out["episodes"].append(ep)
        out["task1_starts"].append(k1s); out["task1_ends"].append(k1e)
        out["task2_starts"].append(k2s); out["task2_ends"].append(k2e)

    if verbose:
        print(f"[build] library size: {len(out['episodes'])} demos")

    bundle = {}
    for k, v in out.items():
        if k.endswith("_yaws"):
            bundle[k] = np.asarray(v, dtype=np.float32)
        elif k in ("episodes", "task1_starts", "task1_ends", "task2_starts", "task2_ends"):
            bundle[k] = np.asarray(v)
        else:
            bundle[k] = _object_array(v)
    return bundle


def _hand_array(hand_seq) -> np.ndarray:
    if any(h is None for h in hand_seq):
        return np.array(hand_seq, dtype=object)
    return np.stack(hand_seq).astype(np.float32)


def _object_array(items) -> np.ndarray:
    # np.array(list_of_arrays, dtype=object) would build a 2-D array when all items share
    # a shape; force a 1-D array of per-demo arrays.
    arr = np.empty(len(items), dtype=object)
    for i, x in enumerate(items):
        arr[i] = x
    return arr


LEN_SUFFIX = "__len"


def save_library(path: Path, bundle: Dict[str, np.ndarray]) -> Path:
    """Save without pickle: every per-demo key (object array of arrays) is stored as its
    concatenation along axis 0 plus ``<key>__len`` with the per-demo lengths. Returns the
    path written (``.npz`` is appended if missing)."""
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_name(path.name + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {}
    for k, v in bundle.items():
        v = np.asarray(v) if not isinstance(v, np.ndarray) else v
        if v.dtype == object and all(isinstance(x, (str, bytes)) for x in v.flat):
            flat[k] = v.astype(str)
        elif v.dtype == object:  # per-demo arrays along axis 0
            if any(x is None for x in v.flat) or any(
                    isinstance(x, np.ndarray) and x.dtype == object and any(e is None for e in x.flat) for x in v):
                raise ValueError(f"cannot save {k!r}: missing entries (e.g. episodes without hand_cmd)")
            try:
                items = [np.asarray(x, dtype=np.float32) for x in v]
            except (TypeError, ValueError) as e:
                raise ValueError(f"cannot save {k!r}: per-demo entries must be numeric arrays") from e
            flat[k] = np.concatenate(items, axis=0) if items else np.zeros((0,), np.float32)
            flat[k + LEN_SUFFIX] = np.array([len(x) for x in items], dtype=np.int64)
        else:
            flat[k] = v.astype(str) if v.dtype.kind in "OU" else v
    np.savez(path, **flat)
    return path


def load_library(path: Path, allow_pickle: bool = False) -> Dict[str, np.ndarray]:
    """Load a library saved by :func:`save_library`. Per-demo keys come back as 1-D object
    arrays of per-demo arrays.

    Libraries written by the original (pre-release) code store pickled object arrays; load
    them with ``allow_pickle=True`` only if you trust the file."""
    with np.load(path, allow_pickle=allow_pickle) as d:
        files = [k for k in d.files if k != "allow_pickle"]
        try:
            raw = {k: d[k] for k in files}
        except ValueError as e:
            raise ValueError(f"{path} is a legacy pickled library; pass allow_pickle=True "
                             "if you trust it, then re-save it with save_library()") from e
    out = {}
    for k, v in raw.items():
        if k.endswith(LEN_SUFFIX):
            continue
        if k + LEN_SUFFIX in raw:
            lens = raw[k + LEN_SUFFIX]
            out[k] = _object_array(np.split(v, np.cumsum(lens)[:-1], axis=0) if len(lens) else [])
        else:
            out[k] = v
    return out
