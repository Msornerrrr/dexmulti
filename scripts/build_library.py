#!/usr/bin/env python3
"""Build the canonical skill library of one session.

  python scripts/build_library.py --session data/npz/grasp_pull
writes data/npz/grasp_pull/library/dual_task_policy_chamfer.npz
"""

import argparse
from pathlib import Path

from dexmulti.library import LIBRARY_FILENAME, build_library, save_library


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", type=Path, required=True, help="folder with episode .npz files")
    p.add_argument("--keyframes", type=Path, default=None, help="default: <session>/keyframes")
    p.add_argument("--out", type=Path, default=None, help=f"default: <session>/library/{LIBRARY_FILENAME}")
    p.add_argument("--pc-downsample", type=int, default=6000)
    a = p.parse_args()

    bundle = build_library(a.session, a.keyframes, pc_downsample=a.pc_downsample)
    if len(bundle["episodes"]) == 0:
        raise SystemExit(f"no demos with keyframes found in {a.session}")
    out = save_library(a.out or a.session / "library" / LIBRARY_FILENAME, bundle)
    print(f"saved {len(bundle['episodes'])} demos to {out}")


if __name__ == "__main__":
    main()
