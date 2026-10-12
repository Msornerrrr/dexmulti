"""LeRobot dataset loader: a tiny dataset in the released on-disk layout (written with pyarrow
only), read back as DexMulti episodes, materialized and turned into a library. If ``lerobot``
and ``av`` are installed, the same check runs on a dataset written by LeRobot itself."""

import json

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

from dexmulti.config import (DRAWER_CONTACT_KEY, DRAWER_PC_KEY, DRAWER_XYZC_KEY, HAND_PC_KEY,  # noqa: E402
                             OBJ_CONTACT_KEY, OBJ_PC_KEY, OBJ_XYZC_KEY)
from dexmulti.data.lerobot import (ACTION_LAYOUT, CLOUD_FEATURES, CONTACT_FEATURES, SKILL_FEATURE,  # noqa: E402
                                   STATE_LAYOUT, LeRobotEpisodes, is_hub_id, lerobot_features, materialize,
                                   skill_labels)
from dexmulti.geometry import Rz  # noqa: E402
from dexmulti.library import LIBRARY_FILENAME, build_library, load_library, save_library  # noqa: E402

KF = {"task1_start": 5, "task1_end": 14, "task2_start": 18, "task2_end": 29}


def make_episode(rng, yaw, T=30):
    box = lambda size: (rng.uniform(-0.5, 0.5, (512, 3)) * np.asarray(size)).astype(np.float32)  # noqa: E731
    obj = box((0.12, 0.04, 0.05)) @ Rz(yaw).T + [0.4, -0.1, 0.05]
    dr = box((0.25, 0.15, 0.1)) + [0.65, 0.1, 0.1]
    arm = np.zeros((T, 6), np.float32)
    arm[:, :3] = np.linspace([0.3, 0.0, 0.3], [0.5, 0.1, 0.2], T)
    arm[:, 3], arm[:, 5] = np.pi, np.linspace(-0.3, 0.3, T)
    ep = {"time": 1.7e9 + np.arange(T) * 0.1001, "eef_pose": arm + 0.001, "arm_cmd": arm,
          "arm_state": rng.normal(size=(T, 7)).astype(np.float32), "hand_state": rng.normal(size=(T, 16)).astype(np.float32),
          "hand_cmd": np.tile(np.linspace(0, 1, T)[:, None], (1, 16)).astype(np.float32),
          OBJ_PC_KEY: np.repeat(obj[None], T, 0).astype(np.float32), DRAWER_PC_KEY: np.repeat(dr[None], T, 0).astype(np.float32),
          HAND_PC_KEY: rng.normal(size=(T, 512, 3)).astype(np.float32),
          OBJ_CONTACT_KEY: rng.uniform(size=(T, 512)).astype(np.float32),
          DRAWER_CONTACT_KEY: rng.uniform(size=(T, 512)).astype(np.float32)}
    return ep


def _fixed(arr, inner):
    """numpy (N, *inner) float32 -> nested fixed-size-list arrow array (as LeRobot/datasets write)."""
    flat = pa.array(np.asarray(arr, np.float32).reshape(-1))
    for n in reversed(inner):
        flat = pa.FixedSizeListArray.from_arrays(flat, n)
    return flat


def write_tiny_dataset(root, episodes, splits=None):
    """Minimal LeRobot v3.0 layout (no videos): what LeRobotEpisodes reads."""
    feats = {k: v for k, v in lerobot_features().items() if v["dtype"] != "video"}
    info = {"codebase_version": "v3.0", "fps": 10, "features": {k: {**v, "shape": list(v["shape"])} for k, v in feats.items()},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "dexmulti" / "keyframes").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))
    cols, metas, extra, start = {}, [], [], 0
    for i, (name, ep) in enumerate(episodes.items()):
        T = len(ep["time"])
        c = {"episode_index": pa.array(np.full(T, i, np.int64)), "frame_index": pa.array(np.arange(T)),
             "timestamp": pa.array(np.arange(T, dtype=np.float32) / 10), "index": pa.array(start + np.arange(T)),
             "observation.state": _fixed(np.concatenate([ep[k] for k, _, _ in STATE_LAYOUT], 1), (29,)),
             "action": _fixed(np.concatenate([ep[k] for k, _, _ in ACTION_LAYOUT], 1), (22,)),
             SKILL_FEATURE: pa.array(skill_labels(T, *KF.values()))}
        c.update({f: _fixed(ep[k], (512, 3)) for f, k in CLOUD_FEATURES.items()})
        c.update({f: _fixed(ep[k], (512, 1)) for f, (k, _, _) in CONTACT_FEATURES.items()})
        for k, v in c.items():
            cols.setdefault(k, []).append(v)
        metas.append({"episode_index": i, "length": T, "data/chunk_index": 0, "data/file_index": 0,
                      "dataset_from_index": start, "dataset_to_index": start + T})
        extra.append({"episode_index": i, "episode": name, "length": T, "wall_time": ep["time"].tolist()})
        (root / "dexmulti" / "keyframes" / f"{name}.json").write_text(json.dumps(KF))
        start += T
    pq.write_table(pa.table({k: pa.concat_arrays(v) for k, v in cols.items()}), root / "data/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist(metas), root / "meta/episodes/chunk-000/file-000.parquet")
    (root / "dexmulti" / "episodes.json").write_text(json.dumps(extra))
    if splits:
        (root / "dexmulti" / "splits.json").write_text(json.dumps(
            {s: {"episode_index": v} for s, v in splits.items()}))
        for s, idx in splits.items():
            (root / "dexmulti" / "keyframes" / s).mkdir()
            for i in idx:
                (root / "dexmulti" / "keyframes" / s / f"{list(episodes)[i]}.json").write_text(
                    json.dumps({**KF, "task2_end": 25}))


def check_roundtrip(ds, episodes, tmp_path):
    assert ds.names == list(episodes)
    for name, ep in episodes.items():
        got = ds.episode(name)
        for k, v in ep.items():
            np.testing.assert_array_equal(got[k], v, err_msg=k)
        np.testing.assert_array_equal(got[DRAWER_XYZC_KEY][..., 3], ep[DRAWER_CONTACT_KEY])
        np.testing.assert_array_equal(got["skill"], skill_labels(len(ep["time"]), *KF.values()))
        assert ds.keyframes(name) == KF
    # materialized npz layout -> same library as from the original episodes
    orig = tmp_path / "orig"
    (orig / "keyframes").mkdir(parents=True)
    for name, ep in episodes.items():
        np.savez(orig / f"{name}.npz", **ep)
        (orig / "keyframes" / f"{name}.json").write_text(json.dumps(KF))
    out = materialize(ds, tmp_path / "npz")
    a, b = build_library(orig, verbose=False), build_library(out, verbose=False)
    assert list(a["episodes"]) == list(episodes)
    for k in a:
        for x, y in zip(np.atleast_1d(a[k]), np.atleast_1d(b[k])):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=k)


def test_skill_labels_overlap():
    np.testing.assert_array_equal(skill_labels(8, 1, 2, 4, 6), [0, 1, 1, 0, 2, 2, 2, 0])
    # recorded skill 2 starts inside skill 1: skill 1 wins where they overlap
    np.testing.assert_array_equal(skill_labels(8, 1, 4, 3, 6), [0, 1, 1, 1, 1, 2, 2, 0])
    # recorded skill 2 starts before skill 1: frames of skill 2 outside skill 1 stay 2
    np.testing.assert_array_equal(skill_labels(8, 2, 4, 1, 6), [0, 2, 1, 1, 1, 2, 2, 0])
    assert skill_labels(5, 1, 2, 3, 99).tolist() == [0, 1, 1, 2, 2]


def test_tiny_dataset_roundtrip(tmp_path, rng):
    episodes = {f"session_{i}": make_episode(rng, yaw) for i, yaw in enumerate([0.2, -0.7, 1.1])}
    root = tmp_path / "ds"
    write_tiny_dataset(root, episodes, splits={"quantity_2": [0, 2]})
    save_library(root / "dexmulti" / "library" / "quantity_2.npz", {"episodes": np.array(["session_0", "session_2"])})
    check_roundtrip(LeRobotEpisodes(root), episodes, tmp_path)

    # split view: the set's keyframes drive `skill`; the set's own keyframe clouds are substituted
    kf_set = {**KF, "task2_end": 25}
    other = {"object": make_episode(rng, 2.5)[OBJ_PC_KEY][KF["task1_start"]], "contact": rng.uniform(size=512).astype(np.float32)}
    (root / "dexmulti" / "set_clouds").mkdir()
    np.savez(root / "dexmulti" / "set_clouds" / "quantity_2.npz", episodes=np.array(["session_2"]),
             object_frame=np.array([KF["task1_start"]]), object_pc=other["object"][None], object_contact=other["contact"][None],
             drawer_frame=np.array([KF["task2_start"]]), drawer_pc=episodes["session_2"][DRAWER_PC_KEY][KF["task2_start"]][None],
             drawer_contact=episodes["session_2"][DRAWER_CONTACT_KEY][KF["task2_start"]][None])
    sub = LeRobotEpisodes(root, split="quantity_2")
    assert sub.names == ["session_0", "session_2"] and sub.keyframes(2) == kf_set
    e2 = sub.episode("session_2")
    np.testing.assert_array_equal(e2["skill"], skill_labels(30, *kf_set.values()))
    np.testing.assert_array_equal(e2[OBJ_PC_KEY][KF["task1_start"]], other["object"])
    np.testing.assert_array_equal(e2[OBJ_XYZC_KEY][KF["task1_start"]], np.c_[other["object"], other["contact"]])
    np.testing.assert_array_equal(np.delete(e2[OBJ_PC_KEY], KF["task1_start"], 0),
                                  np.delete(episodes["session_2"][OBJ_PC_KEY], KF["task1_start"], 0))
    raw = LeRobotEpisodes(root, split="quantity_2", set_clouds=False).episode(2)
    np.testing.assert_array_equal(raw[OBJ_PC_KEY], episodes["session_2"][OBJ_PC_KEY])
    for bad in (1, "session_1"):  # not in the split
        with pytest.raises(KeyError, match="not in split"):
            sub.index(bad)

    # rebuilding from the split equals building from the set's own data
    ref = tmp_path / "q2_ref"
    (ref / "keyframes").mkdir(parents=True)
    for name in sub.names:
        ep = dict(episodes[name])
        if name == "session_2":
            ep[OBJ_PC_KEY] = ep[OBJ_PC_KEY].copy()
            ep[OBJ_PC_KEY][KF["task1_start"]] = other["object"]
        np.savez(ref / f"{name}.npz", **ep)
        (ref / "keyframes" / f"{name}.json").write_text(json.dumps(kf_set))
    out = materialize(sub, tmp_path / "q2")
    assert sorted(p.stem for p in out.glob("*.npz")) == ["session_0", "session_2"]
    assert list(load_library(out / "library" / LIBRARY_FILENAME)["episodes"]) == ["session_0", "session_2"]
    a, b = build_library(ref, verbose=False), build_library(out, verbose=False)
    for k in a:
        for x, y in zip(np.atleast_1d(a[k]), np.atleast_1d(b[k])):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=k)
    with pytest.raises(KeyError):
        LeRobotEpisodes(root, split="nope")


def test_materialize_overwrite_removes_stale(tmp_path, rng):
    episodes = {f"session_{i}": make_episode(rng, 0.1 * i, T=20) for i in range(3)}
    root = tmp_path / "ds"
    write_tiny_dataset(root, episodes, splits={"one": [1]})
    out = materialize(LeRobotEpisodes(root), tmp_path / "npz")
    (out / "library").mkdir()
    (out / "library" / LIBRARY_FILENAME).write_bytes(b"old")
    with pytest.raises(FileExistsError):
        materialize(LeRobotEpisodes(root, split="one"), out)
    materialize(LeRobotEpisodes(root, split="one"), out, overwrite=True)  # this split has no library
    assert [p.stem for p in out.glob("*.npz")] == ["session_1"]
    assert [p.stem for p in (out / "keyframes").glob("*.json")] == ["session_1"]
    assert not (out / "library" / LIBRARY_FILENAME).exists()



def test_materialize_rejects_mismatched_split(tmp_path, rng):
    episodes = {f"session_{i}": make_episode(rng, 0.1 * i, T=20) for i in range(3)}
    root = tmp_path / "ds"
    write_tiny_dataset(root, episodes, splits={"one": [1], "two": [0, 2]})
    with pytest.raises(ValueError, match="split"):
        materialize(LeRobotEpisodes(root), tmp_path / "a", split="one")
    with pytest.raises(ValueError, match="split"):
        materialize(LeRobotEpisodes(root, split="one"), tmp_path / "a", split="two")
    out = materialize(LeRobotEpisodes(root, split="two"), tmp_path / "b", split="two")  # same split is fine
    assert sorted(p.stem for p in out.glob("*.npz")) == ["session_0", "session_2"]
    out = materialize(root, tmp_path / "c", split="one")  # path + split
    assert [p.stem for p in out.glob("*.npz")] == ["session_1"]

def test_missing_local_path_is_not_a_hub_id(tmp_path):
    for src in (tmp_path / "nope", "data/lerobot/dexmulti_grasp_pul", "./dexmulti_grasp_pul", "~/x/y"):
        with pytest.raises(FileNotFoundError, match="not found"):
            LeRobotEpisodes(src)
    assert is_hub_id("some-org/dexmulti_grasp_pull") and not is_hub_id("../a/b")


def test_written_by_lerobot(tmp_path, rng):
    """The real writer (LeRobot v3.0) -> loader, incl. video decoding."""
    pytest.importorskip("av")
    lr = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    from lerobot.configs.video import RGBEncoderConfig

    episodes = {f"session_{i}": make_episode(rng, yaw, T=12) for i, yaw in enumerate([0.3, -0.4])}
    root = tmp_path / "lr"
    ds = lr.LeRobotDataset.create(repo_id="test/tiny", fps=10, features=lerobot_features((48, 64, 3)), root=root,
                                  use_videos=True, rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=10, g=10))
    images = {}
    for name, ep in episodes.items():
        T = len(ep["time"])
        images[name] = (np.arange(T)[:, None, None, None] * 20 + np.zeros((T, 48, 64, 3))).astype(np.uint8)
        state = np.concatenate([ep[k] for k, _, _ in STATE_LAYOUT], 1)
        action = np.concatenate([ep[k] for k, _, _ in ACTION_LAYOUT], 1)
        skill = skill_labels(T, 1, 4, 6, 10)
        for t in range(T):
            f = {"observation.images.cam1": images[name][t], "observation.images.cam2": images[name][t],
                 "observation.state": state[t], "action": action[t], SKILL_FEATURE: skill[t:t + 1], "task": "test"}
            f.update({k: ep[v][t] for k, v in CLOUD_FEATURES.items()})
            f.update({k: ep[v][t][:, None] for k, (v, _, _) in CONTACT_FEATURES.items()})
            ds.add_frame(f)
        ds.save_episode()
    ds.finalize()
    (root / "dexmulti").mkdir()
    (root / "dexmulti" / "episodes.json").write_text(json.dumps(
        [{"episode_index": i, "episode": n, "wall_time": e["time"].tolist()} for i, (n, e) in enumerate(episodes.items())]))

    got = LeRobotEpisodes(root)
    assert got.names == list(episodes) and got.tasks == ["test"]
    for name, ep in episodes.items():
        e = got.episode(name)
        for k, v in ep.items():
            np.testing.assert_array_equal(e[k], v, err_msg=k)
        frames = got.video_frames(name, "cam2")
        assert frames.shape == images[name].shape
        assert np.abs(frames[:, 8:-8, 8:-8].astype(int) - images[name][:, 8:-8, 8:-8]).max() <= 6
        np.testing.assert_array_equal((got.video_frames(name, "cam1", frames=[7, 2]).mean((1, 2, 3)) / 20).round(), [7, 2])
