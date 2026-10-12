#!/usr/bin/env python3
"""Download the released demonstrations (LeRobot datasets) from Hugging Face.

  python scripts/download_data.py --dataset grasp_pull            # one task
  python scripts/download_data.py --dataset all --no-videos       # everything, without camera videos

Each dataset lands in <out>/dexmulti_<name>/. Convert one to the npz layout used by the other
scripts with scripts/lerobot_to_npz.py, or view an episode with scripts/visualize_demo.py.
"""

import argparse
from pathlib import Path

HF_ORG = "HJ666"
DATASETS = ["grasp_pull", "grasp_open", "grasp_grasp", "grasp_open_sample_efficiency"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=[*DATASETS, "all"], default="grasp_pull")
    p.add_argument("--out", type=Path, default=Path("data/lerobot"))
    p.add_argument("--no-videos", action="store_true", help="skip the camera videos (point clouds / robot data only)")
    a = p.parse_args()

    from huggingface_hub import snapshot_download  # pip install dexmulti[lerobot]

    for name in DATASETS if a.dataset == "all" else [a.dataset]:
        repo = f"{HF_ORG}/dexmulti_{name}"
        path = snapshot_download(repo_id=repo, repo_type="dataset", local_dir=a.out / f"dexmulti_{name}",
                                 ignore_patterns=["videos/*"] if a.no_videos else None)
        print(f"{repo} -> {path}")


if __name__ == "__main__":
    main()
