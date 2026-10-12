#!/usr/bin/env python3
"""Detect skill keyframes (contact-based) for every episode of a session.

  python scripts/segment_keyframes.py --session data/my_task
writes data/my_task/keyframes/<episode>.json. Existing files are kept unless --overwrite.
Requires contact maps in the episodes (see dexmulti.perception.contact).
"""

import argparse
import json
from pathlib import Path

from dexmulti.data.episode import iter_session, keyframe_path, load_episode
from dexmulti.data.keyframes import KeyframeConfig, segment_episode


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--contact-thr", type=float, default=KeyframeConfig.contact_thr)
    a = p.parse_args()

    cfg = KeyframeConfig(contact_thr=a.contact_thr)
    n_ok = n_skip = 0
    for ep, path, kf in iter_session(a.session):
        if kf is not None and not a.overwrite:
            n_skip += 1
            continue
        res = segment_episode(load_episode(path), cfg)
        if res is None or "task2_start" not in res:
            print(f"[warn] {ep}: no keyframes found")
            continue
        out = keyframe_path(a.session, ep)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"episode": ep, **res}, indent=2))
        n_ok += 1
    print(f"wrote {n_ok} keyframe files, kept {n_skip} existing")


if __name__ == "__main__":
    main()
