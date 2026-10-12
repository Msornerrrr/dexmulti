"""All constants used by the DexMulti experiments, in one place.

Defaults reproduce the configuration used for the IROS 2026 paper
(xArm7 + LEAP Hand, two RealSense cameras, robot base frame = xArm ``link_base``).
Values that look arbitrary are kept exactly as run; see comments.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

Bounds = Dict[str, Tuple[float, float]]


@dataclass
class ArmLimits:
    """Safety limits applied to every Cartesian command (see ``execution.clip_and_clamp``)."""

    # Per-step relative clamp, applied at CTRL_HZ: 0.1 m/s and 25 deg/s.
    max_step_pos_m: float = 0.1 / 10
    max_step_rot_deg: float = 25.0 / 10
    workspace: Bounds = field(
        default_factory=lambda: {"x": (0.1, 0.7), "y": (-0.5, 0.5), "z": (0.15, 0.5)}
    )


@dataclass
class RetrievalConfig:
    num_yaw_bins: int = 24  # discrete yaw hypotheses Theta when yaw is unobservable
    lambda_drawer: float = 0.3  # weight of the second object in the joint score
    max_points: int = 6000  # random downsample of query / library clouds
    min_points: int = 10


@dataclass
class TrackerConfig:
    """``perception.tracker.ObjectFrameTracker`` + ``MultiSphereEnvelope`` settings."""

    rate_hz: float = 30.0
    # The ROS node called ``tracker.update(cloud)`` with the default dt=0.01 even though
    # it ran at 30 Hz. Kept as run; the EKF noise parameters were tuned with this value.
    ekf_dt: float = 0.01
    num_spheres: int = 6
    sphere_ema_alpha: float = 0.1


@dataclass
class WorldConfig:
    """Static collision geometry added to the planner world (robot base frame)."""

    ground_z: float = 0.03
    wall_x: float = 0.68  # plane with normal -x behind the workspace


@dataclass
class DeployConfig:
    ctrl_hz: float = 10.0
    align_seconds: float = 10.0
    bridge_seconds: float = 3.0
    # Grasp alignment keeps the hand open for this fraction of the alignment window,
    # then switches to the first hand command of the retrieved demo.
    grasp_open_fraction: float = 0.75
    # PCA yaw sign is resolved so that the commanded world-frame EE yaw is closest to 0;
    # hypotheses beyond this are rejected.
    max_abs_world_yaw_rad: float = float(np.deg2rad(120.0))
    start_pos: Tuple[float, float, float] = (0.35, 0.0, 0.45)
    start_rpy: Tuple[float, float, float] = (3.14, 0.0, 0.0)
    # LEAP Hand joint order as used by the LEAP ROS node (4 per finger, thumb last).
    start_hand: Tuple[float, ...] = (
        0.2, 0.0, 0.2, 0.5,
        0.2, 0.0, 0.2, 0.5,
        0.2, 0.0, 0.2, 0.5,
        1.3, 0.0, -0.1, 0.2,
    )


@dataclass
class DexMultiConfig:
    arm: ArmLimits = field(default_factory=ArmLimits)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)


# Dataset keys (processed episodes). Kept identical to the original recordings so that
# released files and the paper's internal data share one format.
OBJ_PC_KEY = "fused_segmented_point_cloud_base_object"
DRAWER_PC_KEY = "fused_segmented_point_cloud_base_drawer"
HAND_PC_KEY = "hand_point_cloud_base"
OBJ_CONTACT_KEY = f"{OBJ_PC_KEY}_to_{HAND_PC_KEY}_contact_map"
DRAWER_CONTACT_KEY = f"{DRAWER_PC_KEY}_to_{HAND_PC_KEY}_contact_map"
OBJ_XYZC_KEY = f"{OBJ_PC_KEY}_to_{HAND_PC_KEY}_xyzc_contact_cloud"
DRAWER_XYZC_KEY = f"{DRAWER_PC_KEY}_to_{HAND_PC_KEY}_xyzc_contact_cloud"
