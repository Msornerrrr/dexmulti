"""Per-object state estimation loop: ObjectFrameTracker + MultiSphereEnvelope.

Pure-Python equivalent of the ROS node that ran during the experiments
(30 Hz, one instance per segmented object cloud). Its outputs are what the
deployment loop consumes: centroid + task yaw, the yaw-observability flag, and
collision spheres for the planner.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from dexmulti.config import TrackerConfig
from dexmulti.perception.envelope import MultiSphereEnvelope
from dexmulti.perception.tracker import ObjectFrameTracker, ObjectFrameTrackerBaseline


@dataclass
class ObjectState:
    center: np.ndarray            # (3,) base frame
    yaw: float                    # task yaw (held when not observable)
    yaw_observable: bool
    obs_score: float
    sphere_centers: np.ndarray    # (K, 3) base frame
    sphere_radii: np.ndarray      # (K,)

    @property
    def pose6(self) -> np.ndarray:
        """[x, y, z, 0, 0, yaw]: the object pose message published during experiments."""
        return np.array([*self.center, 0.0, 0.0, self.yaw], dtype=np.float32)


class ObjectStateEstimator:
    def __init__(self, cfg: TrackerConfig = TrackerConfig(), baseline: bool = False):
        self.cfg = cfg
        self.tracker = ObjectFrameTrackerBaseline() if baseline else ObjectFrameTracker()
        self.envelope = MultiSphereEnvelope(k=cfg.num_spheres)
        self._prev_centers = None
        self._prev_radii = None
        self.state: Optional[ObjectState] = None

    def update(self, cloud_world: np.ndarray) -> Optional[ObjectState]:
        """Feed one segmented object cloud (N, >=3) in the base frame."""
        if cloud_world is None or len(cloud_world) == 0:
            return self.state
        cloud = np.asarray(cloud_world, dtype=np.float32)[:, :3]

        self.tracker.update(cloud, dt=self.cfg.ekf_dt)
        center, yaw = self.tracker.get_pose()
        if center is None:
            return self.state
        yaw_obs, score = self.tracker.get_yaw_observability()

        # Envelope is fit in the object-centered frame, then EMA-smoothed.
        self.envelope.update(cloud - center, dt=1.0 / self.cfg.rate_hz)
        centers_obj, radii = self.envelope.get_spheres()
        if self._prev_centers is not None:
            a = self.cfg.sphere_ema_alpha
            centers_obj = a * centers_obj + (1.0 - a) * self._prev_centers
            radii = a * radii + (1.0 - a) * self._prev_radii
        self._prev_centers = centers_obj.copy()
        self._prev_radii = radii.copy()

        if centers_obj.shape[0] == 0:
            return self.state

        self.state = ObjectState(
            center=np.asarray(center, dtype=np.float32),
            yaw=float(yaw),
            yaw_observable=bool(yaw_obs),
            obs_score=float(score),
            sphere_centers=(centers_obj + center).astype(np.float32),
            sphere_radii=np.asarray(radii, dtype=np.float32),
        )
        return self.state
