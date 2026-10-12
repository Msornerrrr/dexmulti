"""What DexMulti needs from a robot and a perception stack.

Implement these two protocols for your setup (see ``robots/xarm_leap_ros1`` for the
adapter used in the paper). All quantities are in the robot base frame; poses are
``[x, y, z, roll, pitch, yaw]`` (meters, radians, scipy Euler ``"xyz"``).

Commands are issued at ``DeployConfig.ctrl_hz`` (10 Hz). Upsampling to the hand's PD
rate (LEAP: 40 Hz) and to the arm controller is the adapter's job.
"""

from typing import Optional, Protocol

import numpy as np

from dexmulti.perception.state import ObjectState


class RobotInterface(Protocol):
    def get_eef_pose(self) -> np.ndarray:
        """(6,) current end-effector (xArm link7) pose."""

    def get_arm_joints(self) -> np.ndarray:
        """(7,) arm joint positions."""

    def get_hand_joints(self) -> np.ndarray:
        """(16,) hand joint positions."""

    def set_control_mode(self, mode: str) -> None:
        """'joint' (alignment: streamed joint targets) or 'pose' (bridge/replay: streamed
        Cartesian targets, non-blocking)."""

    def send_arm_joints(self, q: np.ndarray) -> None:
        """(7,) joint target, only used in 'joint' mode."""

    def send_eef_pose(self, pos: np.ndarray, rpy: np.ndarray) -> None:
        """Cartesian target, only used in 'pose' mode. The caller has already applied
        ``execution.clip_and_clamp``."""

    def send_hand(self, q: np.ndarray) -> None:
        """(16,) hand joint target."""


class Perception(Protocol):
    def get_cloud(self, name: str) -> np.ndarray:
        """(N, 3) latest segmented, fused cloud of object ``name`` ('object' or 'drawer')."""

    def get_state(self, name: str) -> Optional[ObjectState]:
        """Latest tracked state of object ``name`` (see ``perception.ObjectStateEstimator``)."""
