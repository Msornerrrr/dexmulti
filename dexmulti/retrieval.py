"""Chamfer retrieval in canonical object space (paper Eq. 2 / Alg. 1).

A single demonstration index i* is retrieved jointly over the first object and the
second interaction target ("drawer" in code: drawer / container / second object):

    i* = argmin_i  d_obj(i) + lambda_drawer * d_drawer(i)

When an object's yaw is observable (see ``perception.tracker``), its query cloud is
rotated by the tracked task yaw and compared once. Otherwise the Chamfer distance is
minimized over a discrete set of yaw hypotheses and the best angle theta* is stored for
alignment.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from dexmulti.geometry import Rz, centroid, chamfer_distance, estimate_yaw, random_downsample
from dexmulti.library import load_library


def canonicalize_query_pc(pc_world: np.ndarray, max_points: int) -> np.ndarray:
    """Centroid-subtract, rotate by -PCA yaw, downsample."""
    pc = pc_world.astype(np.float32)
    c = centroid(pc).astype(np.float32)
    pc_centered = pc - c
    yaw_obj = estimate_yaw(pc_centered)
    pc_canon = pc_centered @ Rz(-yaw_obj).T
    return random_downsample(pc_canon, max_points, seed=0).astype(np.float32)


def chamfer_min_over_yaw(pc_centered: np.ndarray, demo_pc_centered: np.ndarray, thetas: np.ndarray):
    """min_theta Chamfer(Rz(theta) P, Q); returns (distance, theta*)."""
    best_d = float("inf")
    best_theta = 0.0
    for th in thetas:
        d = chamfer_distance(pc_centered @ Rz(th).T, demo_pc_centered)
        if d < best_d:
            best_d = d
            best_theta = th
    return best_d, best_theta


class DualTaskPolicyChamfer:
    """Retrieve-once policy over a library built by :func:`dexmulti.library.build_library`."""

    def __init__(self, library: Union[str, Path, Dict[str, np.ndarray]], num_yaw_bins: int = 24):
        d = load_library(library) if isinstance(library, (str, Path)) else library

        self.obj_pc_canon = d["obj_pc_canon"]
        self.obj_pos = d["obj_pos"]
        self.obj_rot = d["obj_rot"]
        self.obj_hand = d["obj_hand"]

        self.drawer_pc_centered = d["drawer_pc_centered"]
        self.drawer_pc_canon = d["drawer_pc_canon"]
        self.drawer_pos = d["drawer_pos"]
        self.drawer_rot = d["drawer_rot"]
        self.drawer_hand = d["drawer_hand"]

        self.episodes = d["episodes"]
        self.thetas = np.linspace(-np.pi, np.pi, num_yaw_bins, endpoint=False)

        self.demo_idx: Optional[int] = None
        self.obj_theta_star: Optional[float] = None
        self.drawer_theta_star: Optional[float] = None

    def __len__(self) -> int:
        return len(self.episodes)

    def retrieve_object(
        self,
        query_pc_world: np.ndarray,
        drawer_pc_world: np.ndarray,
        max_points: int,
        obj_yaw_observable: bool,
        drawer_yaw_observable: bool,
        yaw_task: Optional[float] = None,
        lambda_drawer: float = 0.3,
        exclude: Optional[int] = None,
    ):
        """Joint retrieval over both objects. Returns the canonical skill-1 segment
        ``(pos_seq, rot_seq, hand_seq, demo_idx, score)``.

        ``exclude`` skips one library index (used for leave-one-out evaluation)."""
        # A fresh retrieval starts a new episode.
        self.demo_idx = self.obj_theta_star = self.drawer_theta_star = None

        obj_pc = query_pc_world.astype(np.float32)
        obj_pc_centered = random_downsample(obj_pc - centroid(obj_pc), max_points, seed=0)

        dr_pc = drawer_pc_world.astype(np.float32)
        dr_pc_centered = random_downsample(dr_pc - centroid(dr_pc), max_points, seed=0)

        best_i = None
        best_score = float("inf")
        best_obj_theta = None

        for i, obj_demo_pc in enumerate(self.obj_pc_canon):
            if i == exclude:
                continue
            if obj_demo_pc is None or len(obj_demo_pc) < 10:
                continue

            if obj_yaw_observable:
                assert yaw_task is not None
                d_obj = chamfer_distance(obj_pc_centered @ Rz(-yaw_task).T, obj_demo_pc)
                obj_theta = None
            else:
                d_obj, obj_theta = chamfer_min_over_yaw(obj_pc_centered, obj_demo_pc, self.thetas)

            dr_demo_pc = self.drawer_pc_canon[i]
            if dr_demo_pc is None or len(dr_demo_pc) < 10:
                continue

            if drawer_yaw_observable:
                # As run in the paper experiments: when the second object's yaw is
                # observable its *centered* (not yaw-canonicalized) query cloud is compared
                # with the canonical demo cloud. Unlike the object branch, no rotation is
                # applied here.
                d_dr = chamfer_distance(dr_pc_centered, dr_demo_pc)
            else:
                d_dr, _ = chamfer_min_over_yaw(dr_pc_centered, dr_demo_pc, self.thetas)

            score = d_obj + lambda_drawer * d_dr
            if score < best_score:
                best_score = score
                best_i = i
                best_obj_theta = obj_theta

        if best_i is None:
            raise RuntimeError("No valid demo found.")

        self.demo_idx = best_i
        self.obj_theta_star = best_obj_theta
        # drawer_theta_star is (re)computed in retrieve_drawer() right before skill 2,
        # when the second object is observed again.

        return (
            self.obj_pos[best_i],
            self.obj_rot[best_i],
            self.obj_hand[best_i],
            int(best_i),
            float(best_score),
        )

    def retrieve_drawer(self, drawer_pc_world: np.ndarray, max_points: int, yaw_observable: bool):
        """Skill 2 uses the *same* demo index. If the second object's yaw is not
        observable, re-estimate theta* against that demo's canonical cloud."""
        if self.demo_idx is None:
            raise RuntimeError("retrieve_object() must be called first.")

        if not yaw_observable:
            pc = drawer_pc_world.astype(np.float32)
            pc_centered = random_downsample(pc - centroid(pc), max_points, seed=0)
            _, theta_star = chamfer_min_over_yaw(
                pc_centered, self.drawer_pc_canon[self.demo_idx], self.thetas
            )
            self.drawer_theta_star = theta_star

        return (
            self.drawer_pos[self.demo_idx],
            self.drawer_rot[self.demo_idx],
            self.drawer_hand[self.demo_idx],
        )
