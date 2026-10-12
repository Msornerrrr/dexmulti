"""Outputs must match the original implementation used in the paper experiments.

Golden values were produced by running the original code (tag ``iros2026-paper``) on
recorded data; see ``docs/reproducing.md``.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from conftest import FIXTURES
from dexmulti.execution import clip_and_clamp
from dexmulti.geometry import canon_action_to_world, resolve_yaw_sign_by_world_action
from dexmulti.perception.state import ObjectStateEstimator
from dexmulti.retrieval import DualTaskPolicyChamfer


def _theta(v):
    return None if np.isnan(v) else float(v)


def test_retrieval_matches_original(grasp_pull_library):
    g = np.load(FIXTURES / "grasp_pull_retrieval_golden.npz")
    policy = DualTaskPolicyChamfer(grasp_pull_library)
    for i in range(len(g["idx"])):
        _, _, _, idx, score = policy.retrieve_object(
            g["obj"][i], g["dr"][i], 6000, bool(g["obj_obs"][i]), bool(g["dr_obs"][i]), yaw_task=float(g["yaw_task"][i])
        )
        assert idx == g["idx"][i]
        assert abs(score - g["score"][i]) < 1e-9
        assert policy.obj_theta_star == _theta(g["theta_obj"][i])
        policy.retrieve_drawer(g["dr"][i], 6000, bool(g["dr_obs"][i]))
        assert policy.drawer_theta_star == _theta(g["theta_dr"][i])


def test_tracker_matches_original():
    g = np.load(FIXTURES / "tracker_golden.npz")
    est = ObjectStateEstimator()
    for t, cloud in enumerate(g["clouds"]):
        st = est.update(cloud)
        if not g["valid"][t]:
            continue
        np.testing.assert_allclose(st.center, g["center"][t], atol=1e-6)
        assert abs(st.yaw - g["yaw"][t]) < 1e-5
        assert st.yaw_observable == g["obs"][t]
        np.testing.assert_allclose(st.sphere_centers, g["sc"][t], atol=1e-6)
        np.testing.assert_allclose(st.sphere_radii, g["sr"][t], atol=1e-6)


def test_geometry_and_arm_limits_match_original():
    g = np.load(FIXTURES / "geometry_golden.npz")
    for i in range(len(g["yaw"])):
        pw, ew = canon_action_to_world(g["c"][i].astype(np.float32), float(g["yaw"][i]),
                                       g["p"][i].astype(np.float32), g["R"][i].astype(np.float32))
        np.testing.assert_allclose(pw, g["pw"][i], atol=1e-6)
        # compare rotations, not Euler angles (ill-conditioned near gimbal lock)
        np.testing.assert_allclose(R.from_euler("xyz", ew).as_matrix(), R.from_euler("xyz", g["ew"][i]).as_matrix(), atol=1e-6)
        ys = resolve_yaw_sign_by_world_action(float(g["yaw"][i]), g["R"][i].astype(np.float32))
        assert abs(ys - g["ys"][i]) < 1e-9
        cp, cr = clip_and_clamp(g["tgt_p"][i], g["tgt_r"][i], g["cur"][i].astype(np.float32))
        np.testing.assert_allclose(cp, g["cp"][i], atol=1e-6)
        np.testing.assert_allclose(R.from_euler("xyz", cr).as_matrix(), R.from_euler("xyz", g["cr"][i]).as_matrix(), atol=1e-6)
