#!/usr/bin/env python3
"""Convert a released LeRobot dataset (or one sample-efficiency set) to the npz session layout
used by build_library.py / retrieve_offline.py (``pip install dexmulti[lerobot]``).

  python scripts/lerobot_to_npz.py --dataset data/lerobot/dexmulti_grasp_pull --out data/npz/grasp_pull
  python scripts/lerobot_to_npz.py --dataset data/lerobot/dexmulti_grasp_open_sample_efficiency \\
      --split quantity_5 --out data/npz/quantity_5
  python scripts/build_library.py --session data/npz/grasp_pull --out /tmp/rebuilt.npz
"""

import argparse
from pathlib import Path

from dexmulti.data.lerobot import LeRobotEpisodes, materialize


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="local LeRobot dataset folder or Hugging Face dataset id")
    p.add_argument("--split", default=None, help="sample-efficiency set, e.g. quantity_5 or diversity_4obj_x11")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true", help="replace the episodes / keyframes / library of an earlier run")
    a = p.parse_args()
    try:
        ds = LeRobotEpisodes(a.dataset, split=a.split)
    except (FileNotFoundError, KeyError) as e:
        raise SystemExit(f"error: {e.args[0] if e.args else e}")
    try:
        out = materialize(ds, a.out, overwrite=a.overwrite)
    except FileExistsError:
        raise SystemExit(f"error: {a.out} already contains a converted dataset; pass --overwrite to replace it")
    print(f"wrote {len(ds)} episodes to {out}" + (f" (library: {ds.library_path.name})" if ds.library_path else ""))


if __name__ == "__main__":
    main()
