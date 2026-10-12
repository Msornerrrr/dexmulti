import numpy as np

from dexmulti.config import DRAWER_CONTACT_KEY, DRAWER_PC_KEY, HAND_PC_KEY, OBJ_CONTACT_KEY, OBJ_PC_KEY
from dexmulti.data.keyframes import contact_persistence, longest_true_run, segment_episode


def test_persistence_bridges_short_gaps_only():
    x = np.array([1, 1, 0, 0, 1, 1] + [0] * 30, bool)
    a = contact_persistence(x, K_occl=3)
    assert a[:6].all() and not a[-1]
    assert longest_true_run(np.array([0, 1, 1, 0, 1, 1, 1, 0], bool)) == (4, 6)


def test_segment_synthetic_episode(rng):
    T, N = 80, 200
    obj = rng.uniform(-0.03, 0.03, (T, N, 3)).astype(np.float32) + [0.4, 0.0, 0.05]
    obj[45:, :, 2] += 0.1  # lifted after frame 45
    dr = rng.uniform(-0.05, 0.05, (T, N, 3)).astype(np.float32) + [0.65, 0.0, 0.1]
    hand = np.repeat(np.array([[0.4, 0.0, 0.5]], np.float32), N, 0)[None].repeat(T, 0)
    hand[30:50] = obj[30:50, :, :]  # touching the object
    hand[60:] = dr[60:]  # touching the second object
    c_obj = np.zeros((T, N), np.float32); c_obj[30:50] = 1.0
    c_dr = np.zeros((T, N), np.float32); c_dr[60:] = 1.0
    res = segment_episode({"time": np.arange(T) * 0.1, OBJ_PC_KEY: obj, DRAWER_PC_KEY: dr, HAND_PC_KEY: hand,
                           OBJ_CONTACT_KEY: c_obj, DRAWER_CONTACT_KEY: c_dr})
    assert res["task1_start"] == 30 and 45 <= res["task1_end"] <= 50
    assert res["task2_start"] == 60 and res["task2_end"] == T - 1
