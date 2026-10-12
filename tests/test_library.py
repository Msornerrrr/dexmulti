import json

import numpy as np

from dexmulti.config import DRAWER_PC_KEY, OBJ_PC_KEY
from dexmulti.geometry import Rz, canon_action_to_world
from dexmulti.library import build_library, load_library, save_library
from dexmulti.retrieval import DualTaskPolicyChamfer


def box(rng, size, n=512):
    return (rng.uniform(-0.5, 0.5, (n, 3)) * np.asarray(size)).astype(np.float32)


def write_episode(root, name, rng, obj_pose, dr_pose, T=30):
    obj = box(rng, (0.12, 0.04, 0.05)) @ Rz(obj_pose[2]).T + [obj_pose[0], obj_pose[1], 0.05]
    dr = box(rng, (0.25, 0.15, 0.1)) @ Rz(dr_pose[2]).T + [dr_pose[0], dr_pose[1], 0.1]
    arm = np.zeros((T, 6), np.float32)
    arm[:, :3] = np.linspace([0.3, 0.0, 0.3], [0.5, 0.1, 0.2], T)
    arm[:, 3] = np.pi
    arm[:, 5] = np.linspace(-0.3, 0.3, T)
    hand = np.tile(np.linspace(0, 1, T)[:, None], (1, 16)).astype(np.float32)
    np.savez(root / f"{name}.npz", time=np.arange(T) * 0.1, arm_cmd=arm, eef_pose=arm, hand_cmd=hand,
             **{OBJ_PC_KEY: np.repeat(obj[None], T, 0), DRAWER_PC_KEY: np.repeat(dr[None], T, 0)})
    (root / "keyframes").mkdir(exist_ok=True)
    (root / "keyframes" / f"{name}.json").write_text(json.dumps(
        {"task1_start": 5, "task1_end": 14, "task2_start": 18, "task2_end": T - 1}))
    return obj, arm, hand


def test_canonical_actions_map_back_to_demo(tmp_path, rng):
    obj, arm, hand = write_episode(tmp_path, "ep0", rng, (0.4, -0.1, 0.7), (0.65, 0.1, -0.2))
    lib = build_library(tmp_path, verbose=False)
    assert list(lib["episodes"]) == ["ep0"]
    c = obj.mean(0)
    yaw = float(lib["obj_yaws"][0])
    for k, t in enumerate(range(5, 15)):
        pos, rpy = canon_action_to_world(c, yaw, lib["obj_pos"][0][k], lib["obj_rot"][0][k])
        np.testing.assert_allclose(pos, arm[t, :3], atol=1e-5)
        np.testing.assert_allclose(np.cos(rpy - arm[t, 3:]), 1.0, atol=1e-5)
        np.testing.assert_allclose(lib["obj_hand"][0][k], hand[t])


def test_save_load_and_self_retrieval(tmp_path, rng):
    for i, yaw in enumerate([0.0, 0.8, -1.2]):
        write_episode(tmp_path, f"ep{i}", rng, (0.35, 0.05 * i, yaw), (0.65, 0.0, 0.1 * i))
    path = tmp_path / "lib.npz"
    save_library(path, build_library(tmp_path, verbose=False))
    policy = DualTaskPolicyChamfer(load_library(path))
    assert len(policy) == 3
    d = np.load(tmp_path / "ep1.npz")
    _, _, _, idx, _ = policy.retrieve_object(d[OBJ_PC_KEY][0], d[DRAWER_PC_KEY][0], 6000,
                                             obj_yaw_observable=False, drawer_yaw_observable=False)
    assert idx == 1


def test_library_format_edge_cases(tmp_path):
    import pytest

    # zero demos round-trip
    empty = build_library(tmp_path, verbose=False)
    loaded = load_library(save_library(tmp_path / "empty.npz", empty))
    assert len(loaded["episodes"]) == 0 and len(loaded["obj_pos"]) == 0
    # string object arrays are stored as plain strings
    names = np.array(["a", "b"], dtype=object)
    lib = {"episodes": names, "obj_pos": np.array([np.zeros((2, 3)), np.ones((3, 3))], dtype=object)}
    out = load_library(save_library(tmp_path / "s", lib))  # .npz suffix appended
    assert list(out["episodes"]) == ["a", "b"] and [len(x) for x in out["obj_pos"]] == [2, 3]
    # missing hand commands are rejected instead of being saved as NaN
    bad = {"obj_hand": np.array([np.array([None, None], dtype=object)], dtype=object)}
    with pytest.raises(ValueError):
        save_library(tmp_path / "bad.npz", bad)
