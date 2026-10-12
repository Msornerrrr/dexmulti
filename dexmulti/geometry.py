"""Point-cloud and SE(2)-about-z helpers shared by retrieval, alignment and execution.

Poses are ``[x, y, z, roll, pitch, yaw]`` in the robot base frame (meters, radians,
scipy Euler ``"xyz"``), matching the xArm driver convention used during data collection.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R


def flatten_pc(pc) -> np.ndarray:
    pc = np.asarray(pc)
    if pc.ndim > 2:
        pc = pc.reshape(-1, pc.shape[-1])
    return pc[:, :3]


def centroid(pc: np.ndarray) -> np.ndarray:
    return np.mean(pc, axis=0)


def Rz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def random_downsample(P: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    if max_points <= 0 or P.shape[0] <= max_points:
        return P.astype(np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.choice(P.shape[0], size=max_points, replace=False)
    return P[idx].astype(np.float32)


def estimate_yaw(pc_centered: np.ndarray) -> float:
    """Canonical yaw: direction of the first principal axis of the XY projection."""
    if pc_centered.shape[0] < 10:
        return 0.0

    Pxy = pc_centered[:, :2]
    _, S, Vt = np.linalg.svd(Pxy, full_matrices=False)

    if S[1] < 1e-6:
        return 0.0

    v = Vt[0]
    return float(np.arctan2(v[1], v[0]))


def chamfer_distance(P: np.ndarray, Q: np.ndarray) -> float:
    """Symmetric Chamfer distance (mean nearest-neighbor distance, both directions)."""
    tree_Q = cKDTree(Q)
    tree_P = cKDTree(P)
    d1 = tree_Q.query(P, k=1)[0].mean()
    d2 = tree_P.query(Q, k=1)[0].mean()
    return float(d1 + d2)


def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def pose6_to_xyz_yaw(pose6: np.ndarray):
    """Split a ``[x, y, z, roll, pitch, yaw]`` object pose into (centroid, yaw)."""
    est = np.asarray(pose6, dtype=np.float32)
    if est.shape[0] < 6:
        raise ValueError(f"Expected 6D pose, got shape {est.shape}")
    return est[:3], float(est[5])


def canon_action_to_world(c_est: np.ndarray, yaw_est: float,
                          pos_canon: np.ndarray, rot_canon: np.ndarray):
    """Map a canonical (object-frame) EE pose to the robot base frame.

    p_world = c + Rz(yaw) p_canon,  R_world = Rz(yaw) R_canon
    """
    R_est = Rz(yaw_est)
    pos_world = c_est + (R_est @ pos_canon.astype(np.float32))
    R_world = R_est @ rot_canon.astype(np.float32)
    euler_world = R.from_matrix(R_world).as_euler("xyz").astype(np.float32)
    euler_world[2] = wrap_to_pi(euler_world[2])
    return pos_world.astype(np.float32), euler_world


def resolve_yaw_sign_by_world_action(
    yaw_raw: float,
    rot_canon: np.ndarray,
    max_abs_world_yaw: float = np.deg2rad(120.0),
) -> float:
    """Resolve the PCA yaw sign ambiguity (yaw vs. yaw + pi).

    Picks the hypothesis whose resulting *world-frame commanded EE yaw* is closest to 0
    (hand facing forward), rejecting hypotheses beyond ``max_abs_world_yaw``.
    """
    candidates = [yaw_raw, yaw_raw + np.pi]

    best = None
    best_score = float("inf")

    for yaw_obj in candidates:
        yaw_obj = wrap_to_pi(yaw_obj)
        R_world = Rz(yaw_obj) @ rot_canon
        yaw_world = wrap_to_pi(R.from_matrix(R_world).as_euler("xyz")[2])
        abs_yaw = abs(yaw_world)

        if abs_yaw > max_abs_world_yaw:
            continue

        if abs_yaw < best_score:
            best_score = abs_yaw
            best = yaw_obj

    # Fallback: both rejected, take the one with smaller |world yaw|. Note this returns
    # the *unwrapped* candidate, as in the original implementation.
    if best is None:
        best = min(
            candidates,
            key=lambda y: abs(wrap_to_pi(R.from_matrix(Rz(y) @ rot_canon).as_euler("xyz")[2])),
        )

    return best
