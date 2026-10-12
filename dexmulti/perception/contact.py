"""Soft hand-object contact map (CordViP-style), used for keyframe detection and as the
4th channel of the object-centric DP3 baseline input.

For each object point v with normal n, the distance to the hand point cloud is
weighted by how well the direction to the hand aligns with the normal:

    D(v) = min_h exp(gamma * (1 - |<(h - v)/||h - v||, n>|)) * ||h - v||
    c(v) = 1 - 2 * (sigmoid(theta * D) - 0.5)        in [0, 1]
"""

import numpy as np


def estimate_normals(points: np.ndarray, k: int = 30) -> np.ndarray:
    import open3d as o3d  # optional dependency: pip install dexmulti[perception]

    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    return np.asarray(cloud.normals, dtype=np.float32)


def aligned_distance(v_o: np.ndarray, n_o: np.ndarray, hand_pts: np.ndarray, gamma: float = 1.0) -> float:
    directions = hand_pts - v_o
    d = np.linalg.norm(directions, axis=1)
    dirs_norm = directions / (d[:, None] + 1e-8)
    alignment = np.abs(np.dot(dirs_norm, n_o))
    return float(np.min(np.exp(gamma * (1 - alignment)) * d))


def contact_value(D: float, theta: float = 10.0) -> float:
    sig = 1.0 / (1.0 + np.exp(-D * theta))
    return 1 - 2 * (sig - 0.5)


def contact_map(obj_pts: np.ndarray, hand_pts: np.ndarray, gamma: float = 1.0, theta: float = 10.0) -> np.ndarray:
    """(M,3) object points, (N,3) hand points -> (M,) contact scores in [0, 1]."""
    obj_pts = obj_pts[np.isfinite(obj_pts).all(axis=1)]
    hand_pts = hand_pts[np.isfinite(hand_pts).all(axis=1)]
    normals = estimate_normals(obj_pts)
    scores = np.zeros(obj_pts.shape[0], dtype=np.float32)
    for i in range(obj_pts.shape[0]):
        scores[i] = contact_value(aligned_distance(obj_pts[i], normals[i], hand_pts, gamma), theta)
    return np.clip(scores, 0.0, 1.0)


def contact_map_sequence(obj_seq: np.ndarray, hand_seq: np.ndarray):
    """(T,M,3), (T,N,3) -> contact map (T,M) and xyzc cloud (T,M,4), zero-padded."""
    T = obj_seq.shape[0]
    maps, xyzc = [], []
    for t in range(T):
        obj = obj_seq[t][..., :3]
        obj = obj[np.isfinite(obj).all(axis=1)]
        c = contact_map(obj, hand_seq[t][..., :3])
        maps.append(c)
        xyzc.append(np.concatenate([obj, c[:, None]], axis=1))
    M = max(m.shape[0] for m in maps)
    cmap = np.zeros((T, M), dtype=np.float32)
    cloud = np.zeros((T, M, 4), dtype=np.float32)
    for t in range(T):
        cmap[t, : maps[t].shape[0]] = maps[t]
        cloud[t, : xyzc[t].shape[0]] = xyzc[t]
    return cmap, cloud
