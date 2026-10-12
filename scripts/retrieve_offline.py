#!/usr/bin/env python3
"""Leave-one-out retrieval on recorded demonstrations (no robot needed).

For each held-out episode, the object states are estimated by running the tracker on
the episode's first frames (before the hand reaches the object), approximating how it ran
on the robot: at 30 Hz on the 10 Hz data (each frame fed 3x), after a few seconds on the
static scene (the start-pose move). One demo is retrieved
from the *other* demos, and its first grasp action is mapped into the held-out scene.
The error to the grasp pose the human demonstrator actually used is reported.

  python scripts/retrieve_offline.py --session data/npz/grasp_pull
  python scripts/retrieve_offline.py --session data/npz/grasp_pull --viz   # needs dexmulti[viz]
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from dexmulti.alignment.targets import alignment_target
from dexmulti.config import DRAWER_PC_KEY, OBJ_PC_KEY, DexMultiConfig
from dexmulti.data.episode import load_episode
from dexmulti.geometry import flatten_pc
from dexmulti.library import LIBRARY_FILENAME, build_library, load_library
from dexmulti.perception.state import ObjectStateEstimator
from dexmulti.retrieval import DualTaskPolicyChamfer


def estimate(clouds, n, cfg, data_hz=10.0, settle_s=3.0):
    est = ObjectStateEstimator(cfg.tracker)
    per_frame = max(1, int(round(cfg.tracker.rate_hz / data_hz)))
    for _ in range(int(settle_s * cfg.tracker.rate_hz)):
        est.update(flatten_pc(clouds[0]))
    for c in clouds[:n]:
        for _ in range(per_frame):
            est.update(flatten_pc(c))
    return est.state


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--library", type=Path, default=None)
    p.add_argument("--viz", action="store_true", help="show each retrieval in viser")
    a = p.parse_args()
    cfg = DexMultiConfig()

    if a.library is not None and not a.library.exists():
        p.error(f"--library {a.library} does not exist")
    lib_path = a.library or a.session / "library" / LIBRARY_FILENAME
    lib = load_library(lib_path) if lib_path.exists() else build_library(a.session, verbose=False)
    if len(lib["episodes"]) < 2:
        p.error(f"need at least 2 demos for leave-one-out, found {len(lib['episodes'])} in {a.session}")
    policy = DualTaskPolicyChamfer(lib, num_yaw_bins=cfg.retrieval.num_yaw_bins)

    viz = None
    if a.viz:
        from dexmulti.viz import RetrievalViewer
        viz = RetrievalViewer()

    pos_err, rot_err = [], []
    for i, ep in enumerate(lib["episodes"]):
        d = load_episode(a.session / f"{ep}.npz")
        k1 = int(lib["task1_starts"][i])
        n = max(2, k1 // 2)  # frames well before contact
        obj, dr = estimate(d[OBJ_PC_KEY], n, cfg), estimate(d[DRAWER_PC_KEY], n, cfg)

        pos_seq, rot_seq, _, j, score = policy.retrieve_object(
            flatten_pc(d[OBJ_PC_KEY][n - 1]), flatten_pc(d[DRAWER_PC_KEY][n - 1]), cfg.retrieval.max_points,
            obj.yaw_observable, dr.yaw_observable, yaw_task=obj.yaw,
            lambda_drawer=cfg.retrieval.lambda_drawer, exclude=i,
        )
        yaw_used, pos, rpy = alignment_target(obj.center, obj.yaw, policy.obj_theta_star, pos_seq[0], rot_seq[0],
                                       cfg.deploy.max_abs_world_yaw_rad)
        gt = d["arm_cmd"][k1]
        pos_err.append(np.linalg.norm(pos - gt[:3]))
        rot_err.append(np.degrees((R.from_euler("xyz", rpy).inv() * R.from_euler("xyz", gt[3:6])).magnitude()))
        print(f"{ep}: retrieved {lib['episodes'][j]} (score {score:.4f}, yaw observable {obj.yaw_observable}) "
              f"grasp pose error {100 * pos_err[-1]:.1f} cm / {rot_err[-1]:.1f} deg")
        if viz is not None:
            viz.show(d, n - 1, pos_seq, rot_seq, obj.center, yaw_used, gt_pose=gt,
                     title=f"{ep} -> {lib['episodes'][j]}")
            input("enter for next ...")

    print(f"\n{len(pos_err)} episodes: median grasp pose error {100 * np.median(pos_err):.1f} cm, "
          f"{np.median(rot_err):.1f} deg")


if __name__ == "__main__":
    main()
