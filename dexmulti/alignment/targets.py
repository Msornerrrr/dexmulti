"""Alignment target computation (closed loop, re-evaluated every control step)."""

from typing import Optional

import numpy as np

from dexmulti.geometry import canon_action_to_world, resolve_yaw_sign_by_world_action


def alignment_yaw(yaw_tracked: float, theta_star: Optional[float], rot_canon: np.ndarray,
                  max_abs_world_yaw: float = np.deg2rad(120.0)) -> float:
    """Object yaw used to map the canonical demo into the world.

    * yaw observable at retrieval (theta* is None): the tracker's task yaw, which the
      tracker itself holds constant while yaw becomes temporarily unobservable;
    * otherwise: the retrieval yaw, -theta* (the query was rotated by theta* to match
      the canonical demo).
    The PCA sign ambiguity is then resolved against the commanded EE yaw.
    """
    yaw_raw = -theta_star if theta_star is not None else yaw_tracked
    return resolve_yaw_sign_by_world_action(yaw_raw, rot_canon, max_abs_world_yaw)


def alignment_target(center: np.ndarray, yaw_tracked: float, theta_star: Optional[float],
                     pos_canon: np.ndarray, rot_canon: np.ndarray,
                     max_abs_world_yaw: float = np.deg2rad(120.0)):
    """Returns (yaw used, target position (3,), target rpy (3,)) for the first
    canonical action of the retrieved skill."""
    yaw = alignment_yaw(yaw_tracked, theta_star, rot_canon, max_abs_world_yaw)
    pos, rpy = canon_action_to_world(np.asarray(center, dtype=np.float32), yaw, pos_canon, rot_canon)
    return yaw, pos, rpy


def alignment_hand_target(elapsed_s: float, duration_s: float, grasp_task: bool,
                          demo_hand_cmd, open_hand: np.ndarray, open_fraction: float = 0.75) -> np.ndarray:
    """Grasp skills approach with an open hand and close to the demo's first hand
    command for the last part of the alignment window; other skills (the hand is
    already holding the first object) track the demo's first hand command throughout."""
    if grasp_task and elapsed_s < duration_s * open_fraction:
        return np.asarray(open_hand, dtype=np.float32).copy()
    return np.asarray(demo_hand_cmd, dtype=np.float32)
