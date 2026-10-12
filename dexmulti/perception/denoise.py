"""Segmented RGB-D -> object point cloud (CPU implementation).

Per camera:  masked depth back-projection -> base frame -> workspace crop
             -> random downsample -> statistical outlier removal -> radius outlier removal
Fused:       concatenate cameras -> DBSCAN (drop noise + small clusters) -> FPS to 512 points

The experiments ran the same chain on GPU (cuML + PyTorch3D) inside ROS nodes; this
module reproduces the logic with numpy / scipy / scikit-learn so it can be used offline
or plugged into any perception stack. Random downsampling makes outputs stochastic.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class CleanConfig:
    rand_points: int = 8192
    # Radius outlier removal (as run: approximate, mean distance to the k nearest
    # neighbours must be below ``radius``)
    radius: float = 0.02
    min_neighbors: int = 20
    # Statistical outlier removal
    k: int = 20
    std_ratio: float = 1.0
    # DBSCAN
    eps: float = 0.015
    min_samples: int = 10
    min_cluster_size: int = 20
    num_points: int = 4096
    crop: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {"x": (0.2, 1.0), "y": (-0.4, 0.4), "z": (0.0, 0.7)}
    )


# Settings used for the segmented object clouds in the paper.
SEGMENTED = CleanConfig(
    rand_points=6000, radius=0.012, min_neighbors=40, k=16, std_ratio=0.3,
    eps=0.008, min_samples=30, min_cluster_size=60, num_points=512,
)


def backproject(depth: np.ndarray, mask: Optional[np.ndarray], intrinsics: Sequence[float],
                depth_scale: float = 0.001) -> np.ndarray:
    """depth (H,W) + optional mask -> (N,3) points in the camera frame."""
    fx, fy, cx, cy = intrinsics
    if depth.ndim == 3:
        depth = depth[..., 0]
    if mask is None:
        mask = depth > 0
    elif mask.ndim == 3:
        mask = mask.any(axis=-1)
    ys, xs = np.nonzero(mask)
    z = depth[ys, xs].astype(np.float32) * depth_scale
    return np.stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z], axis=1).astype(np.float32)


def transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    return (points @ T[:3, :3].T + T[:3, 3]).astype(np.float32)


def crop(points: np.ndarray, bounds) -> np.ndarray:
    m = np.ones(len(points), dtype=bool)
    for i, ax in enumerate("xyz"):
        lo, hi = bounds[ax]
        m &= (points[:, i] >= lo) & (points[:, i] <= hi)
    return points[m]


def random_downsample(points: np.ndarray, n: int, rng=None) -> np.ndarray:
    if len(points) <= n:
        return points
    rng = np.random.default_rng() if rng is None else rng
    return points[rng.choice(len(points), size=n, replace=False)]


def _mean_knn_dist(points: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(points))
    d, _ = cKDTree(points).query(points, k=k)
    d = np.atleast_2d(d)
    return d[:, 1:].mean(axis=1)  # first neighbour is the point itself


def statistical_outlier_removal(points: np.ndarray, k: int, std_ratio: float) -> np.ndarray:
    md = _mean_knn_dist(points, k)
    return points[md < md.mean() + std_ratio * md.std()]


def radius_outlier_removal(points: np.ndarray, radius: float, min_neighbors: int) -> np.ndarray:
    return points[_mean_knn_dist(points, min_neighbors + 1) < radius]


def dbscan_denoise(points: np.ndarray, eps: float, min_samples: int, min_cluster_size: int) -> np.ndarray:
    from sklearn.cluster import DBSCAN  # optional dependency

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    keep = labels != -1
    if not keep.any():
        return points[:0]
    vals, counts = np.unique(labels[keep], return_counts=True)
    big = vals[counts >= max(min_cluster_size, 1)]
    return points[keep & np.isin(labels, big)]


def farthest_point_sample(points: np.ndarray, n: int) -> np.ndarray:
    """FPS starting from index 0 (same as PyTorch3D's default); pads by repetition
    when fewer than ``n`` points are available."""
    N = len(points)
    if N == 0:
        return points
    k = min(n, N)
    idx = np.zeros(k, dtype=np.int64)
    dist = np.full(N, np.inf)
    for i in range(1, k):
        dist = np.minimum(dist, np.sum((points - points[idx[i - 1]]) ** 2, axis=1))
        idx[i] = int(np.argmax(dist))
    out = points[idx]
    if k < n:
        out = np.tile(out, ((n + k - 1) // k, 1))[:n]
    return out


def clean_single_view(depth, mask, intrinsics, cam_to_base: np.ndarray, cfg: CleanConfig = SEGMENTED,
                      depth_scale: float = 0.001, rng=None) -> Optional[np.ndarray]:
    pts = backproject(depth, mask, intrinsics, depth_scale)
    if len(pts) == 0:
        return None
    pts = crop(transform(pts, cam_to_base), cfg.crop)
    if len(pts) == 0:
        return None
    pts = random_downsample(pts, cfg.rand_points, rng)
    pts = statistical_outlier_removal(pts, cfg.k, cfg.std_ratio)
    if len(pts) == 0:
        return None
    pts = radius_outlier_removal(pts, cfg.radius, cfg.min_neighbors)
    return pts if len(pts) else None


def fuse_views(views: List[Optional[np.ndarray]], cfg: CleanConfig = SEGMENTED) -> Optional[np.ndarray]:
    views = [v for v in views if v is not None and len(v)]
    if not views:
        return None
    pts = dbscan_denoise(np.concatenate(views, axis=0), cfg.eps, cfg.min_samples, cfg.min_cluster_size)
    if len(pts) == 0:
        return None
    return farthest_point_sample(pts, cfg.num_points)
