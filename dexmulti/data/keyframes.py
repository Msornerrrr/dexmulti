"""Automatic skill segmentation of a two-skill demonstration (offline).

Skill k starts at the first frame of the longest run of *sustained* hand-object
contact. Contact evidence at frame t requires both
  * a non-empty soft contact map (any point with contact > ``contact_thr``), and
  * the object centroid being within ``dist_thr`` of the hand point cloud, where
    ``dist_thr`` scales with the object's size (robust to segmentation leaking onto
    the hand).
Short gaps (occlusions) are bridged with a leaky budget. Skill 1 ends when the
object is lifted by an adaptive height; skill 2 runs to the end of the episode.

Output keys: ``task1_start``, ``task1_end``, ``task2_start``, ``task2_end`` (+ diagnostics).
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from dexmulti.config import DRAWER_CONTACT_KEY, DRAWER_PC_KEY, HAND_PC_KEY, OBJ_CONTACT_KEY, OBJ_PC_KEY


@dataclass
class KeyframeConfig:
    contact_thr: float = 0.8
    occl_frames: int = 10
    lift_height: Optional[float] = None  # fixed lift dz override
    lift_frac: float = 0.30
    lift_dz_min: float = 0.02
    lift_dz_max: float = 0.035
    height_q_low: float = 0.05
    height_q_high: float = 0.95
    lift_frames: int = 5
    dist_scale: float = 1.3
    radius_quantile: float = 0.90
    dist_thr_min: float = 0.03
    dist_thr_max: float = 0.15
    min_pc_points_for_radius: int = 50


def compute_contact_density(contact_map: np.ndarray, thr: float) -> np.ndarray:
    return np.sum(contact_map > thr, axis=1)


def contact_persistence(active_like: np.ndarray, K_occl: int = 10) -> np.ndarray:
    """Persistent contact with a leaky occlusion budget: a True frame refills the budget
    (+1, capped at K_occl), a False frame inside a run spends it (-1); the run ends when
    the budget is exhausted."""
    x = active_like > 0
    T = len(x)
    active = np.zeros(T, dtype=np.bool_)
    budget = 0
    for t in range(T):
        if x[t]:
            if t == 0 or not active[t - 1]:
                active[t] = True
                budget = K_occl
            else:
                active[t] = True
                budget = min(K_occl, budget + 1)
        else:
            if t > 0 and active[t - 1] and budget > 0:
                active[t] = True
                budget -= 1
            else:
                active[t] = False
                budget = 0
    return active


def longest_true_run(x: np.ndarray):
    """(start, end) inclusive of the longest contiguous True run, or (None, None)."""
    best_start, best_len = None, 0
    cur_start, cur_len = None, 0
    for t, v in enumerate(x):
        if v:
            if cur_start is None:
                cur_start, cur_len = t, 1
            else:
                cur_len += 1
        else:
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
            cur_start, cur_len = None, 0
    if cur_len > best_len:
        best_len, best_start = cur_len, cur_start
    if best_start is None:
        return None, None
    return best_start, best_start + best_len - 1


def compute_centroid_z(pc_seq) -> np.ndarray:
    z = np.zeros(len(pc_seq), dtype=np.float32)
    for t, pc in enumerate(pc_seq):
        if isinstance(pc, np.ndarray) and pc.ndim == 2 and pc.shape[0] > 0:
            z[t] = float(np.mean(pc[:, 2]))
        else:
            z[t] = z[t - 1] if t > 0 else 0.0
    return z


def find_lift_termination(z: np.ndarray, start_t: int, dz: float = 0.02, K: int = 5) -> int:
    """First frame at which z has stayed above z[start_t] + dz for K consecutive frames."""
    z0 = float(z[start_t])
    count = 0
    for t in range(start_t + 1, len(z)):
        if float(z[t]) > z0 + dz:
            count += 1
            if count >= K:
                return t
        else:
            count = 0
    return len(z) - 1


def first_valid_pc(pc_seq, min_points: int = 30):
    for t, pc in enumerate(pc_seq):
        if isinstance(pc, np.ndarray) and pc.ndim == 2 and pc.shape[0] >= min_points:
            return t, pc
    return None, None


def estimate_radius_from_pc(pc: np.ndarray, robust_q: float = 0.90) -> float:
    if pc is None or pc.shape[0] < 5:
        return 0.0
    c = np.mean(pc[:, :3], axis=0)
    return float(np.quantile(np.linalg.norm(pc[:, :3] - c[None, :], axis=1), robust_q))


def estimate_height_from_pc(pc: np.ndarray, low_q: float = 0.05, high_q: float = 0.95) -> float:
    if pc is None or pc.shape[0] < 5:
        return 0.0
    z = pc[:, 2].astype(np.float32)
    return float(max(0.0, float(np.quantile(z, high_q)) - float(np.quantile(z, low_q))))


def compute_lift_dz(pc0_obj: np.ndarray, cfg: KeyframeConfig) -> float:
    if cfg.lift_height is not None:
        return float(np.clip(cfg.lift_height, cfg.lift_dz_min, cfg.lift_dz_max))
    h_obj = estimate_height_from_pc(pc0_obj, cfg.height_q_low, cfg.height_q_high)
    return float(np.clip(cfg.lift_frac * h_obj, cfg.lift_dz_min, cfg.lift_dz_max))


def min_dist_centroid_to_hand(pc_obj: np.ndarray, pc_hand: np.ndarray) -> float:
    if (
        pc_obj is None or pc_hand is None
        or pc_obj.ndim != 2 or pc_hand.ndim != 2
        or pc_obj.shape[0] == 0 or pc_hand.shape[0] == 0
    ):
        return float("inf")
    c_obj = np.mean(pc_obj[:, :3], axis=0).astype(np.float32)
    diff = pc_hand[:, :3].astype(np.float32) - c_obj[None, :]
    return float(np.sqrt(np.min(np.einsum("ij,ij->i", diff, diff))))


def compute_min_dist_series(obj_pc_seq, hand_pc_seq) -> np.ndarray:
    T = min(len(obj_pc_seq), len(hand_pc_seq))
    d = np.full(T, np.inf, dtype=np.float32)
    last = np.inf
    for t in range(T):
        dt = min_dist_centroid_to_hand(obj_pc_seq[t], hand_pc_seq[t])
        if np.isfinite(dt):
            last = dt
        d[t] = last
    return d


def _contact_start(pc_seq, hand_seq, contact, cfg: KeyframeConfig):
    density = compute_contact_density(contact, cfg.contact_thr)
    _, pc0 = first_valid_pc(pc_seq, min_points=cfg.min_pc_points_for_radius)
    r = estimate_radius_from_pc(pc0, robust_q=cfg.radius_quantile)
    dist_thr = float(np.clip(cfg.dist_scale * r, cfg.dist_thr_min, cfg.dist_thr_max))
    dist = compute_min_dist_series(pc_seq, hand_seq)
    valid = (density > 0) & (dist < dist_thr)
    start, _ = longest_true_run(contact_persistence(valid, K_occl=cfg.occl_frames))
    return start, dist_thr, r, pc0


def segment_episode(data: Dict[str, np.ndarray], cfg: KeyframeConfig = KeyframeConfig()) -> Optional[Dict]:
    """Segment one episode (dict of arrays, see ``dexmulti.data.episode``)."""
    T = len(data["time"]) if "time" in data else len(data[OBJ_PC_KEY])
    hand_seq = list(data[HAND_PC_KEY])

    if OBJ_CONTACT_KEY not in data:
        return None
    obj_seq = list(data[OBJ_PC_KEY])
    t1_start, dist_thr_obj, r_obj, pc0_obj = _contact_start(obj_seq, hand_seq, np.asarray(data[OBJ_CONTACT_KEY]), cfg)
    if t1_start is None:
        return None
    lift_dz = compute_lift_dz(pc0_obj, cfg)
    t1_end = find_lift_termination(compute_centroid_z(obj_seq), t1_start, dz=lift_dz, K=cfg.lift_frames)

    result = {
        "task1_start": int(t1_start),
        "task1_end": int(t1_end),
        "task1_dist_thr": dist_thr_obj,
        "task1_radius_est": r_obj,
        "task1_lift_dz": float(lift_dz),
        "task1_contact_key": OBJ_CONTACT_KEY,
    }

    if DRAWER_CONTACT_KEY not in data:
        return result
    dr_seq = list(data[DRAWER_PC_KEY])
    t2_start, dist_thr_dr, r_dr, _ = _contact_start(dr_seq, hand_seq, np.asarray(data[DRAWER_CONTACT_KEY]), cfg)
    if t2_start is None:  # no contact detected: start right after skill 1
        t2_start = int(min(t1_end + 1, T - 1))

    result.update({
        "task2_start": int(t2_start),
        "task2_end": int(T - 1),
        "task2_dist_thr": dist_thr_dr,
        "task2_radius_est": r_dr,
        "task2_contact_key": DRAWER_CONTACT_KEY,
    })
    return result
