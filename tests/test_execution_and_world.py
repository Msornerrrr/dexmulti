import numpy as np
from scipy.spatial.transform import Rotation as R

from dexmulti.alignment.targets import alignment_hand_target, alignment_yaw
from dexmulti.alignment.world import build_world
from dexmulti.config import ArmLimits
from dexmulti.execution import bridge_waypoint, clip_and_clamp, replay_waypoints


def test_clamp_limits_step():
    cur = np.array([0.4, 0.0, 0.3, np.pi, 0.0, 0.0], np.float32)
    pos, rpy = clip_and_clamp([0.6, 0.2, 0.3], [np.pi, 0.0, 1.0], cur)
    lim = ArmLimits()
    assert np.linalg.norm(pos - cur[:3]) <= lim.max_step_pos_m + 1e-9
    dR = R.from_euler("xyz", cur[3:]).inv() * R.from_euler("xyz", rpy)
    assert np.degrees(np.linalg.norm(dR.as_rotvec())) <= lim.max_step_rot_deg + 1e-6


def test_clip_to_workspace():
    cur = np.array([0.69, 0.0, 0.16, np.pi, 0.0, 0.0], np.float32)
    pos, _ = clip_and_clamp([0.9, 0.0, 0.0], cur[3:], cur)
    assert pos[0] <= 0.7 + 1e-9 and pos[2] >= 0.15 - 1e-9


def test_bridge_endpoints():
    Ra, Rb = R.from_euler("xyz", [np.pi, 0, 0]), R.from_euler("xyz", [np.pi, 0, 0.5])
    p0, r0 = bridge_waypoint(0.0, [0, 0, 0], Ra, [1, 1, 1], Rb)
    p1, r1 = bridge_waypoint(1.0, [0, 0, 0], Ra, [1, 1, 1], Rb)
    np.testing.assert_allclose(p0, 0) and np.testing.assert_allclose(p1, 1)
    assert np.allclose((R.from_euler("xyz", r1).inv() * Rb).magnitude(), 0, atol=1e-6)


def test_replay_identity_frame():
    pos = np.array([[0.1, 0.0, 0.2]], np.float32)
    rot = np.eye(3, dtype=np.float32)[None]
    (p, rpy, h), = list(replay_waypoints(np.zeros(3, np.float32), 0.0, pos, rot, [None]))
    np.testing.assert_allclose(p, pos[0]) and np.testing.assert_allclose(rpy, 0, atol=1e-7)
    assert h is None


def test_alignment_rules():
    rot = R.from_euler("xyz", [np.pi, 0, 0]).as_matrix().astype(np.float32)
    assert np.isclose(alignment_yaw(0.3, None, rot), 0.3)  # tracked yaw
    assert np.isclose(alignment_yaw(0.3, 0.5, rot), -0.5)  # retrieval yaw -theta*
    open_q, demo_q = np.zeros(16), np.ones(16)
    assert alignment_hand_target(1.0, 10.0, True, demo_q, open_q).sum() == 0
    assert alignment_hand_target(8.0, 10.0, True, demo_q, open_q).sum() == 16
    assert alignment_hand_target(1.0, 10.0, False, demo_q, open_q).sum() == 16


def test_world_keeps_primitive_count():
    c, r = np.random.rand(6, 3), np.full(6, 0.02)
    w_in = build_world(c, r, c, r, exclude_object=False)
    w_out = build_world(c, r, c, r, exclude_object=True)
    assert len(w_in) == len(w_out) == 2 + 12
