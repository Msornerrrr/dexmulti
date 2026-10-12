"""Released episode format (one ``.npz`` per demonstration, 10 Hz, robot base frame).

==========================================================  ============  ==========================
key                                                         shape         meaning
==========================================================  ============  ==========================
time                                                        (T,)          seconds (wall clock)
eef_pose                                                    (T, 6)        measured EE pose [xyz, rpy]
arm_cmd                                                     (T, 6)        commanded EE pose [xyz, rpy]
arm_state                                                   (T, 7)        xArm7 joint positions
hand_state                                                  (T, 16)       LEAP joint positions
hand_cmd                                                    (T, 16)       LEAP joint commands
fused_segmented_point_cloud_base_object                     (T, 512, 3)   first object
fused_segmented_point_cloud_base_drawer                     (T, 512, 3)   second object / target
hand_point_cloud_base                                       (T, 512, 3)   hand surface points (FK)
<obj>_to_hand_point_cloud_base_contact_map                  (T, 512)      contact score per point
<obj>_to_hand_point_cloud_base_xyzc_contact_cloud           (T, 512, 4)   xyz + contact
==========================================================  ============  ==========================

Poses use scipy Euler ``"xyz"`` angles. ``<obj>`` is either point-cloud key above.
Keyframes live next to the episodes in ``keyframes/<episode>.json``.
"""

import json
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

LOW_DIM_KEYS = ("time", "eef_pose", "arm_cmd", "arm_state", "hand_state", "hand_cmd")


def load_episode(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def load_keyframes(path: Path) -> Dict:
    return json.loads(Path(path).read_text())


def keyframe_path(session_root: Path, episode: str) -> Path:
    """Keyframes of a released session: ``<session>/keyframes/<episode>.json``."""
    return Path(session_root) / "keyframes" / f"{episode}.json"


def iter_session(session_root: Path) -> Iterator[Tuple[str, Path, Optional[Path]]]:
    """Yields (episode name, episode path, keyframe path or None)."""
    session_root = Path(session_root)
    for f in sorted(session_root.glob("*.npz")):
        kf = keyframe_path(session_root, f.stem)
        yield f.stem, f, kf if kf.exists() else None
