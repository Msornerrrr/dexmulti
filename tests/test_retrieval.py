import numpy as np

from dexmulti.geometry import Rz
from dexmulti.retrieval import DualTaskPolicyChamfer


def test_yaw_search_recovers_rotation(grasp_pull_library, rng):
    policy = DualTaskPolicyChamfer(grasp_pull_library)
    i = 7
    canon_obj = np.asarray(grasp_pull_library["obj_pc_canon"][i], dtype=np.float32)
    canon_dr = np.asarray(grasp_pull_library["drawer_pc_canon"][i], dtype=np.float32)
    theta = policy.thetas[5]
    # A scene whose object is the demo's canonical cloud rotated by -theta (+ offset).
    q_obj = canon_obj @ Rz(-theta).T + [0.4, 0.1, 0.05]
    q_dr = canon_dr + [0.65, 0.0, 0.1]
    _, _, _, idx, score = policy.retrieve_object(q_obj, q_dr, 6000, False, False)
    assert idx == i and score < 1e-6
    assert np.isclose(policy.obj_theta_star, theta)


def test_skill2_uses_same_demo(grasp_pull_library):
    policy = DualTaskPolicyChamfer(grasp_pull_library)
    q_obj = np.asarray(grasp_pull_library["obj_pc_canon"][3], dtype=np.float32)
    q_dr = np.asarray(grasp_pull_library["drawer_pc_canon"][3], dtype=np.float32)
    policy.retrieve_object(q_obj, q_dr, 6000, False, True)
    pos, _, _ = policy.retrieve_drawer(q_dr, 6000, yaw_observable=True)
    assert policy.demo_idx == 3 and pos is grasp_pull_library["drawer_pos"][3]
    assert policy.drawer_theta_star is None
