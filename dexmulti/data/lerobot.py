"""Read the released LeRobot datasets as DexMulti episodes (``pip install dexmulti[lerobot]``).

The demonstrations are published as LeRobot v3.0 datasets (one per task). This module reads
them with pyarrow only (no ``lerobot``/``torch``) and returns episodes in the processed format
of :mod:`dexmulti.data.episode`::

    from dexmulti.data.lerobot import LeRobotEpisodes, materialize
    ds = LeRobotEpisodes("path/to/dexmulti_grasp_pull")   # or a Hugging Face dataset id
    ep = ds.episode(0)                     # dict: time, eef_pose, ..., point clouds, skill
    kf = ds.keyframes(0)                   # task{1,2}_{start,end}
    materialize(ds, "data/grasp_pull")     # <out>/*.npz + keyframes/ + library/ for the scripts

Sample-efficiency sets: ``LeRobotEpisodes(root, split="quantity_5")`` restricts to that set and
returns the data as that set used it: its own keyframes (``skill`` is recomputed from them) and,
at the two frames ``build_library`` reads (object cloud at ``task1_start``, second-object cloud at
``task2_start``), that set's own processed clouds (``dexmulti/set_clouds/<set>.npz``), so
rebuilding the library of a set reproduces ``dexmulti/library/<set>.npz`` exactly.

Camera frames are decoded on demand from the mp4 files with PyAV
(:meth:`LeRobotEpisodes.video_frames`, ``pip install av``).
"""

import json
import re
import shutil
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from dexmulti.config import (DRAWER_CONTACT_KEY, DRAWER_PC_KEY, DRAWER_XYZC_KEY, HAND_PC_KEY, OBJ_CONTACT_KEY,
                             OBJ_PC_KEY, OBJ_XYZC_KEY)
from dexmulti.library import LIBRARY_FILENAME

FPS = 10
CAMERAS = ("cam1", "cam2")
_POSE = ["x", "y", "z", "roll", "pitch", "yaw"]
# (dexmulti key, size, element names) in the order they are concatenated.
STATE_LAYOUT = (
    ("eef_pose", 6, [f"eef_{n}" for n in _POSE]),
    ("arm_state", 7, [f"arm_joint{i}" for i in range(1, 8)]),
    ("hand_state", 16, [f"hand_joint{i:02d}" for i in range(16)]),
)
ACTION_LAYOUT = (
    ("arm_cmd", 6, [f"eef_cmd_{n}" for n in _POSE]),
    ("hand_cmd", 16, [f"hand_cmd{i:02d}" for i in range(16)]),
)
CLOUD_FEATURES = {
    "observation.pointcloud.object": OBJ_PC_KEY,
    "observation.pointcloud.drawer": DRAWER_PC_KEY,
    "observation.pointcloud.hand": HAND_PC_KEY,
}
CONTACT_FEATURES = {  # stored as (512, 1)
    "observation.contact.object": (OBJ_CONTACT_KEY, OBJ_PC_KEY, OBJ_XYZC_KEY),
    "observation.contact.drawer": (DRAWER_CONTACT_KEY, DRAWER_PC_KEY, DRAWER_XYZC_KEY),
}
SKILL_FEATURE = "observation.skill"
EXTRAS = "dexmulti"  # folder with DexMulti-specific files inside a dataset root

KF_KEYS = ("task1_start", "task1_end", "task2_start", "task2_end")


def lerobot_features(image_shape=(480, 640, 3)) -> Dict[str, Dict]:
    """The LeRobot feature schema of the released datasets (for ``LeRobotDataset.create``)."""
    v = {"dtype": "video", "shape": tuple(image_shape), "names": ["height", "width", "channels"]}
    f = {f"observation.images.{c}": dict(v) for c in CAMERAS}
    for key, layout in (("observation.state", STATE_LAYOUT), ("action", ACTION_LAYOUT)):
        names = [n for _, _, ns in layout for n in ns]
        f[key] = {"dtype": "float32", "shape": (len(names),), "names": names}
    for feat in CLOUD_FEATURES:
        f[feat] = {"dtype": "float32", "shape": (512, 3), "names": ["point", "xyz"]}
    for feat in CONTACT_FEATURES:  # (512, 1), not (512,): the HF visualizer would plot 512 series
        f[feat] = {"dtype": "float32", "shape": (512, 1), "names": ["point", "contact"]}
    f[SKILL_FEATURE] = {"dtype": "int64", "shape": (1,), "names": None}
    return f


def skill_labels(T: int, t1s: int, t1e: int, t2s: int, t2e: int) -> np.ndarray:
    """Per-frame skill id: 1 on [t1s, t1e], 2 on [t2s, t2e] (inclusive), else 0.
    Where the recorded keyframe ranges overlap, skill 1 wins; frames of [t2s, t2e] outside
    [t1s, t1e] are 2 even if t2s < t1s (a few recordings have such keyframes)."""
    s = np.zeros(T, np.int64)
    s[max(t2s, 0):max(min(t2e + 1, T), 0)] = 2
    s[max(t1s, 0):max(min(t1e + 1, T), 0)] = 1
    return s


def _column(table, name: str, shape: Tuple[int, ...]) -> np.ndarray:
    """Fixed-shape numeric column -> (rows, *shape) array (works with or without ``datasets``
    having registered its Array2D extension type)."""
    import pyarrow as pa

    col = table.column(name).combine_chunks()
    col = getattr(col, "storage", col)
    while pa.types.is_list(col.type) or pa.types.is_large_list(col.type) or pa.types.is_fixed_size_list(col.type):
        col = col.flatten()
    return col.to_numpy(zero_copy_only=False).reshape((len(table),) + tuple(shape))


def resolve_root(source: Union[str, Path], download_videos: bool = False, revision: Optional[str] = None) -> Path:
    """A local dataset folder, or a Hugging Face dataset id (``org/name``) downloaded with
    ``snapshot_download``. Anything that is not an existing folder and not shaped like
    ``org/name`` is reported as a missing local folder."""
    s = str(source)
    p = Path(s).expanduser()
    if p.exists():
        if not (p / "meta" / "info.json").exists():
            raise FileNotFoundError(f"{p} is not a LeRobot dataset (no meta/info.json)")
        return p
    if not is_hub_id(s):
        raise FileNotFoundError(f"dataset folder not found: {p}")
    from huggingface_hub import snapshot_download  # pip install dexmulti[lerobot]
    from huggingface_hub.utils import HFValidationError, RepositoryNotFoundError, RevisionNotFoundError

    patterns = ["meta/*", "data/*", f"{EXTRAS}/*", "README.md"] + (["videos/*"] if download_videos else [])
    try:
        return Path(snapshot_download(repo_id=s, repo_type="dataset", revision=revision, allow_patterns=patterns))
    except (HFValidationError, RepositoryNotFoundError, RevisionNotFoundError) as e:
        raise FileNotFoundError(f"{s!r} is neither a local dataset folder nor a Hugging Face dataset "
                                f"({type(e).__name__})") from e


def is_hub_id(source: str) -> bool:
    """True for strings shaped like a Hugging Face repo id ``org/name`` (not ``./x``, ``/x``, ``~/x``, ``a/b/c``)."""
    return re.fullmatch(r"[A-Za-z0-9][\w.-]*/[\w.-]+", source) is not None and ".." not in source


class LeRobotEpisodes:
    """Episodes of one released dataset, optionally restricted to a split (sample-efficiency set).

    With ``split``, only the episodes of that set are visible (``index``/``episode`` refuse
    others), keyframes are the set's own, ``skill`` is recomputed from them and, unless
    ``set_clouds=False``, the two library keyframe clouds come from the set's own processing
    (see the module docstring)."""

    def __init__(self, source: Union[str, Path], split: Optional[str] = None, download_videos: bool = False,
                 revision: Optional[str] = None, set_clouds: bool = True):
        import pyarrow.parquet as pq

        self.root = resolve_root(source, download_videos, revision)
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self.fps = int(self.info["fps"])
        self._check_layout()
        rows = []
        for f in sorted((self.root / "meta" / "episodes").glob("*/*.parquet")):
            cols = [c for c in pq.read_schema(f).names if not c.startswith("stats/")]
            rows += pq.read_table(f, columns=cols).to_pylist()
        self._meta = {int(r["episode_index"]): r for r in rows}
        ext = self.root / EXTRAS
        extra = json.loads((ext / "episodes.json").read_text()) if (ext / "episodes.json").exists() else []
        self._extra = {int(e["episode_index"]): e for e in extra}
        self.splits = {}
        if (ext / "splits.json").exists():
            self.splits = {k: v["episode_index"] for k, v in json.loads((ext / "splits.json").read_text()).items()}
        self.split = split
        if split is not None and split not in self.splits:
            raise KeyError(f"unknown split {split!r}; available: {sorted(self.splits) or 'none'}")
        self.indices: List[int] = sorted(self.splits[split] if split else self._meta)
        self._set_clouds = {}
        p = ext / "set_clouds" / f"{split}.npz"
        if split is not None and set_clouds and p.exists():
            with np.load(p, allow_pickle=False) as z:
                sc = {k: z[k] for k in z.files}
            self._set_clouds = {str(n): {k: v[j] for k, v in sc.items() if k != "episodes"}
                                for j, n in enumerate(sc["episodes"])}

    def _check_layout(self):
        feats = self.info["features"]
        for key, layout in (("observation.state", STATE_LAYOUT), ("action", ACTION_LAYOUT)):
            names = [n for _, _, ns in layout for n in ns]
            if list(feats[key]["shape"]) != [len(names)] or (feats[key].get("names") not in (None, names)):
                raise ValueError(f"{key} in {self.root} does not have the DexMulti layout")

    # ---- indexing -----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.indices)

    @property
    def names(self) -> List[str]:
        return [self.name(i) for i in self.indices]

    def name(self, index: int) -> str:
        e = self._extra.get(int(index))
        return e["episode"] if e else f"episode_{int(index):06d}"

    def index(self, episode: Union[int, str]) -> int:
        """``episode_index`` of an episode given by its (dataset-wide) episode_index or recording name.
        With a split, the episode must belong to it."""
        if isinstance(episode, (int, np.integer)):
            i = int(episode)
            if i not in self._meta:
                raise IndexError(f"no episode_index {episode} (dataset has {len(self._meta)} episodes)")
        else:
            i = next((j for j in self._meta if self.name(j) == episode), None)
            if i is None:
                raise KeyError(f"no episode named {episode!r}")
        if self.split is not None and i not in self.indices:
            raise KeyError(f"episode {episode!r} (episode_index {i}) is not in split {self.split!r}; its episodes "
                           f"(episode_index: name) are {', '.join(f'{j}: {self.name(j)}' for j in self.indices)}")
        return i

    def __iter__(self) -> Iterator[Tuple[str, Dict[str, np.ndarray], Optional[Dict]]]:
        """Yields (episode name, episode dict, keyframes or None) for the episodes of the split."""
        for i in self.indices:
            yield self.name(i), self.episode(i), self.keyframes(i)

    # ---- data -----------------------------------------------------------------------------
    def _data_path(self, i: int) -> Path:
        m = self._meta[i]
        return self.root / self.info["data_path"].format(chunk_index=m["data/chunk_index"],
                                                         file_index=m["data/file_index"])

    def episode(self, episode: Union[int, str]) -> Dict[str, np.ndarray]:
        """One episode in the processed DexMulti format (keys of :mod:`dexmulti.data.episode`
        plus ``skill``), all arrays in the robot base frame. With a split, ``skill`` follows the
        set's keyframes and the set's own keyframe clouds are substituted (class docstring)."""
        import pyarrow.parquet as pq

        i = self.index(episode)
        feats = self.info["features"]
        cols = ["frame_index", "timestamp", "observation.state", "action", SKILL_FEATURE,
                *CLOUD_FEATURES, *CONTACT_FEATURES]
        tb = pq.read_table(self._data_path(i), columns=cols, filters=[("episode_index", "=", i)])
        tb = tb.sort_by("frame_index")
        T = len(tb)
        if T != int(self._meta[i]["length"]):
            raise ValueError(f"episode {i}: read {T} frames, meta says {self._meta[i]['length']}")
        out = {}
        e = self._extra.get(i)
        out["time"] = (np.asarray(e["wall_time"], np.float64) if e and "wall_time" in e
                       else tb.column("timestamp").to_numpy().astype(np.float64))
        for key, layout in (("observation.state", STATE_LAYOUT), ("action", ACTION_LAYOUT)):
            arr = _column(tb, key, (sum(n for _, n, _ in layout),)).astype(np.float32)
            a = 0
            for k, n, _ in layout:
                out[k] = arr[:, a:a + n]
                a += n
        for feat, key in CLOUD_FEATURES.items():
            out[key] = _column(tb, feat, tuple(feats[feat]["shape"])).astype(np.float32)
        for feat, (key, _, _) in CONTACT_FEATURES.items():
            out[key] = _column(tb, feat, tuple(feats[feat]["shape"])).astype(np.float32).reshape(T, -1)
        sc = self._set_clouds.get(self.name(i))
        if sc is not None:  # the set's own processed clouds at its library keyframes
            for part, pc_key, c_key in (("object", OBJ_PC_KEY, OBJ_CONTACT_KEY), ("drawer", DRAWER_PC_KEY, DRAWER_CONTACT_KEY)):
                t = int(sc[f"{part}_frame"])
                out[pc_key][t], out[c_key][t] = sc[f"{part}_pc"], sc[f"{part}_contact"]
        for _, (key, pc_key, xyzc_key) in CONTACT_FEATURES.items():
            out[xyzc_key] = np.concatenate([out[pc_key], out[key][..., None]], axis=-1)
        kf = self.keyframes(i) if self.split is not None else None
        if kf is not None and all(k in kf for k in KF_KEYS):
            out["skill"] = skill_labels(T, *(int(kf[k]) for k in KF_KEYS))
        else:
            out["skill"] = _column(tb, SKILL_FEATURE, ()).astype(np.int64).reshape(T)
        return out

    def keyframes(self, episode: Union[int, str]) -> Optional[Dict]:
        """Skill keyframes (``task{1,2}_{start,end}``) as used for this split, or None."""
        name = self.name(self.index(episode))
        base = self.root / EXTRAS / "keyframes"
        p = (base / self.split / f"{name}.json") if self.split else (base / f"{name}.json")
        return json.loads(p.read_text()) if p.exists() else None

    @property
    def library_path(self) -> Optional[Path]:
        """Skill library shipped with the dataset (for the split, if any)."""
        lib = self.root / EXTRAS / "library"
        p = lib / (f"{self.split}.npz" if self.split else LIBRARY_FILENAME)
        return p if p.exists() else None

    @property
    def cameras(self) -> Optional[Dict]:
        p = self.root / EXTRAS / "cameras.json"
        return json.loads(p.read_text()) if p.exists() else None

    @property
    def tasks(self) -> List[str]:
        import pyarrow.parquet as pq

        tb = pq.read_table(self.root / "meta" / "tasks.parquet")
        col = "task" if "task" in tb.column_names else tb.schema.pandas_metadata["index_columns"][0]
        return [str(x) for x in tb.column(col).to_pylist()]

    # ---- video ------------------------------------------------------------------------------
    @property
    def video_keys(self) -> List[str]:
        return [k for k, f in self.info["features"].items() if f["dtype"] == "video"]

    def video_frames(self, episode: Union[int, str], camera: str = "cam1",
                     frames: Optional[Sequence[int]] = None, scale: float = 1.0) -> np.ndarray:
        """Decoded RGB frames (N, H, W, 3) uint8 of one camera ("cam1"/"cam2" or the full
        feature key), all frames of the episode or the given frame indices (needs ``av``)."""
        import av

        i = self.index(episode)
        key = camera if camera.startswith("observation.") else f"observation.images.{camera}"
        m = self._meta[i]
        path = self.root / self.info["video_path"].format(video_key=key, chunk_index=m[f"videos/{key}/chunk_index"],
                                                          file_index=m[f"videos/{key}/file_index"])
        if not path.exists():
            raise FileNotFoundError(f"{path} missing (download the dataset with videos)")
        t0, n = float(m[f"videos/{key}/from_timestamp"]), int(m["length"])
        want = set(range(n)) if frames is None else {int(f) for f in frames}
        if want and (min(want) < 0 or max(want) >= n):
            raise IndexError(f"frame index out of range [0, {n})")
        last = max(want) if want else -1
        got = {}
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            start = t0 + (min(want) if want else 0) / self.fps
            c.seek(int(max(start - 1e-3, 0) / s.time_base), stream=s, backward=True, any_frame=False)
            for fr in c.decode(s):
                k = int(round((float(fr.pts * s.time_base) - t0) * self.fps))
                if k > last:
                    break
                if k in want:
                    if scale != 1.0:
                        fr = fr.reformat(width=int(fr.width * scale) // 2 * 2, height=int(fr.height * scale) // 2 * 2)
                    got[k] = fr.to_ndarray(format="rgb24")
        missing = sorted(want - set(got))
        if missing:
            raise ValueError(f"{path}: could not decode frames {missing[:5]}... of episode {i}")
        order = sorted(want) if frames is None else [int(f) for f in frames]
        return np.stack([got[k] for k in order]) if order else np.zeros((0, 0, 0, 3), np.uint8)


def materialize(dataset: Union[LeRobotEpisodes, str, Path], out_dir: Union[str, Path],
                split: Optional[str] = None, overwrite: bool = False) -> Path:
    """Write a dataset (or one split) in the npz session layout used by the scripts:
    ``<out>/<episode>.npz``, ``<out>/keyframes/<episode>.json`` and
    ``<out>/library/dual_task_policy_chamfer.npz`` (the shipped library of that split).
    With ``overwrite=True`` the episodes, keyframes and library of an earlier run are removed first.
    ``split`` applies when ``dataset`` is a path / hub id; for a ``LeRobotEpisodes`` it must be None or
    equal to that object's split (else ValueError)."""
    if isinstance(dataset, LeRobotEpisodes):
        if split is not None and split != dataset.split:
            raise ValueError(f"split={split!r} differs from the split of the given dataset ({dataset.split!r}); "
                             f"pass LeRobotEpisodes(root, split={split!r}) or the dataset path instead")
        ds = dataset
    else:
        ds = LeRobotEpisodes(dataset, split=split)
    out = Path(out_dir)
    old = [*out.glob("*.npz"), *out.glob("keyframes/*.json"), out / "library" / LIBRARY_FILENAME]
    old = [p for p in old if p.is_file()]
    if old and not overwrite:
        raise FileExistsError(f"{out} already has episodes / keyframes / a library (pass overwrite=True)")
    for p in old:  # never mix episodes of an earlier run with this one
        p.unlink()
    (out / "keyframes").mkdir(parents=True, exist_ok=True)
    for name, ep, kf in ds:
        np.savez(out / f"{name}.npz", **ep)
        if kf is not None:
            (out / "keyframes" / f"{name}.json").write_text(json.dumps(kf, indent=1))
    if ds.library_path is not None:
        (out / "library").mkdir(exist_ok=True)
        shutil.copyfile(ds.library_path, out / "library" / LIBRARY_FILENAME)
    return out
