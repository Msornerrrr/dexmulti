"""Planner collision world: static planes + per-object collision spheres.

Primitives are plain dicts so they can be logged / sent over a wire:
  {"type": "plane", "point": [x, y, z], "normal": [nx, ny, nz]}
  {"type": "sphere", "center": [x, y, z], "radius": r}
"""

from typing import Dict, List, Optional

import numpy as np

from dexmulti.config import WorldConfig


def static_world(cfg: WorldConfig = WorldConfig()) -> List[Dict]:
    return [
        {"type": "plane", "point": [0.0, 0.0, cfg.ground_z], "normal": [0.0, 0.0, 1.0]},
        {"type": "plane", "point": [cfg.wall_x, 0.0, 0.0], "normal": [-1.0, 0.0, 0.0]},
    ]


def spheres(centers: Optional[np.ndarray], radii: Optional[np.ndarray]) -> List[Dict]:
    if centers is None or radii is None:
        return []
    centers = np.asarray(centers)
    radii = np.asarray(radii).reshape(-1)
    if centers.ndim != 2 or centers.shape[1] < 3 or radii.shape[0] != centers.shape[0]:
        return []
    out = []
    for c, r in zip(centers[:, :3], radii):
        if not np.isfinite(c).all() or not np.isfinite(r) or r <= 0.0:
            continue
        out.append({"type": "sphere", "center": [float(c[0]), float(c[1]), float(c[2])], "radius": float(r)})
    return out


def placeholder_spheres(centers: Optional[np.ndarray]) -> List[Dict]:
    """Same number of spheres as ``centers`` but tiny and far away. Used to drop an
    object from the collision world while keeping the number of primitives (and hence
    the JIT-compiled planner problem) unchanged."""
    if centers is None:
        return []
    centers = np.asarray(centers)
    if centers.ndim != 2 or centers.shape[1] < 3:
        return []
    return [
        {"type": "sphere", "center": [1.0, 1.0, 1.0], "radius": 0.01}
        for c in centers[:, :3] if np.isfinite(c).all()
    ]


def build_world(obj_centers, obj_radii, drawer_centers, drawer_radii,
                exclude_object: bool, cfg: WorldConfig = WorldConfig()) -> List[Dict]:
    """World used during alignment. As run in the experiments, the first object is an
    obstacle while approaching it to grasp (skill 1) and is replaced by placeholders
    once it is held in the hand (skill 2)."""
    world = static_world(cfg)
    world += placeholder_spheres(obj_centers) if exclude_object else spheres(obj_centers, obj_radii)
    world += spheres(drawer_centers, drawer_radii)
    return world
