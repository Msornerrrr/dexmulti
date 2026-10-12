"""Uncertainty-aware object frame tracker (the state estimator described in the paper).

One tracker per object. A constant-velocity EKF over (x, y, z, yaw) with PCA yaw
measurements, a geometry/uncertainty based yaw-observability belief with hysteresis,
and a gated "task yaw" that is held when yaw is not observable.

``ObjectFrameTrackerBaseline`` is the raw-PCA-yaw ablation (paper Table II).
"""

import numpy as np


# ============================================================
# Math utilities
# ============================================================

def wrap_to_pi(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def angle_diff(a: float, b: float) -> float:
    return wrap_to_pi(a - b)


def centroid(pc: np.ndarray) -> np.ndarray:
    return np.mean(pc, axis=0)


# ============================================================
# Geometry measurement
# ============================================================

def pca_yaw_xy(pts_world: np.ndarray, center_xyz: np.ndarray):
    """
    PCA yaw on XY plane.

    Returns:
      yaw_meas (float)
      anisotropy (float) = sigma1 / sigma2
      xy_extent (float)  = sigma1 / sqrt(N)

    Returns (None, None, None) if degenerate.
    """
    pts_obj = pts_world - center_xyz[None, :]
    Pxy = pts_obj[:, :2].astype(np.float32)

    if Pxy.shape[0] < 2:
        return None, None, None

    try:
        _, S, Vt = np.linalg.svd(Pxy, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, None, None

    if S.shape[0] < 2 or S[1] < 1e-8:
        return None, None, None

    anisotropy = float(S[0] / S[1])
    v = Vt[0].astype(np.float32)
    yaw_meas = float(np.arctan2(v[1], v[0]))
    
    xy_extent = S[0] / np.sqrt(Pxy.shape[0])
    return yaw_meas, anisotropy, xy_extent


# ============================================================
# EKF
# ============================================================

class _EKFSE2Z:
    """
    EKF-lite:
      x = [x, y, z, yaw, vx, vy, vz, wyaw]
    """

    def __init__(
        self,
        q_pos=2e-4,
        q_yaw=2e-4,
        q_vel=2e-3,
        q_w=2e-3,
        r_pos=2e-2,
        r_yaw_base=3e-2,
        nis_gate_pos=25.0,
        nis_gate_yaw=16.0,
        vel_damping=0.85,
        yaw_rate_damping=0.85,
    ):
        self.q_pos = q_pos
        self.q_yaw = q_yaw
        self.q_vel = q_vel
        self.q_w = q_w

        self.r_pos = r_pos
        self.r_yaw_base = r_yaw_base

        self.nis_gate_pos = nis_gate_pos
        self.nis_gate_yaw = nis_gate_yaw

        self.vel_damping = vel_damping
        self.yaw_rate_damping = yaw_rate_damping

        self.x = np.zeros(8, dtype=np.float32)
        self.P = np.eye(8, dtype=np.float32) * 1e-2
        self.initialized = False

    def initialize(self, center_xyz, yaw0=0.0):
        self.x[:] = 0.0
        self.x[0:3] = center_xyz
        self.x[3] = yaw0
        self.P[:] = np.eye(8, dtype=np.float32) * 1e-2
        self.initialized = True

    def predict(self, dt):
        dt = max(dt, 1e-4)

        F = np.eye(8, dtype=np.float32)
        F[0, 4] = dt
        F[1, 5] = dt
        F[2, 6] = dt
        F[3, 7] = dt

        self.x = F @ self.x
        self.x[3] = wrap_to_pi(self.x[3])

        Q = np.zeros((8, 8), dtype=np.float32)
        Q[0:3, 0:3] = np.eye(3) * self.q_pos
        Q[3, 3] = self.q_yaw
        Q[4:7, 4:7] = np.eye(3) * self.q_vel
        Q[7, 7] = self.q_w

        self.P = F @ self.P @ F.T + Q

        self.x[4:7] *= self.vel_damping
        self.x[7] *= self.yaw_rate_damping

    def update_pos(self, pos):
        H = np.zeros((3, 8), dtype=np.float32)
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0

        y = pos - self.x[0:3]
        S = H @ self.P @ H.T + np.eye(3) * self.r_pos

        if y.T @ np.linalg.inv(S) @ y > self.nis_gate_pos:
            return False

        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(8) - K @ H) @ self.P
        self.x[3] = wrap_to_pi(self.x[3])
        return True

    def update_yaw(self, yaw_meas, r_yaw):
        dy = angle_diff(yaw_meas, self.x[3])
        S = self.P[3, 3] + r_yaw

        if (dy * dy) / S > self.nis_gate_yaw:
            return False

        K = self.P[:, 3] / S
        self.x += K * dy
        self.P -= np.outer(K, self.P[3])
        self.x[3] = wrap_to_pi(self.x[3])
        return True


# ============================================================
# ObjectFrameTracker
# ============================================================

class ObjectFrameTracker:
    """
    Stable SE(2)+Z object tracker with:

      - EKF estimation
      - geometry-aware yaw observability
      - temporal observability belief
      - task-level yaw gating

    Output yaw is a *control yaw*, not raw estimate.
    """

    def __init__(self, min_points=20):
        self.min_points = min_points

        # --- EKF ---
        self.ekf = _EKFSE2Z()

        # --- Observability belief ---
        self.obs_score = 0.5
        self.yaw_observable = False

        self.obs_alpha = 0.03
        self.obs_on = 0.7
        self.obs_off = 0.3
        self.yaw_aniso_ref = 2.0
        self.xy_extent_ref = 0.025

        # --- Task yaw gating ---
        self.yaw_std_ref = np.deg2rad(5.0)
        self.yaw_gate_smooth = 0.2

        self.beta = 0.0
        self.yaw_task = 0.0
        self.yaw_task_initialized = False

    # ------------------------------------------------------------
    # Observability logic
    # ------------------------------------------------------------

    def _update_yaw_observability(self, anisotropy, yaw_std, xy_extent):
        geom_signal = np.clip(
            (anisotropy - 1.0) / (self.yaw_aniso_ref - 1.0), 0.0, 1.0
        )
        size_signal = np.clip(
            xy_extent / self.xy_extent_ref,
            0.0,
            1.0
        )
        stat_signal = np.clip(
            self.yaw_std_ref / max(yaw_std, 1e-6), 0.0, 1.0
        )

        instant = geom_signal * size_signal *  stat_signal
        self.obs_score = (1 - self.obs_alpha) * self.obs_score + self.obs_alpha * instant

        if self.yaw_observable:
            if self.obs_score < self.obs_off:
                self.yaw_observable = False
        else:
            if self.obs_score > self.obs_on:
                self.yaw_observable = True

    def _compute_beta(self, yaw_std):
        if not self.yaw_observable:
            return 0.0
        return np.exp(-(yaw_std / self.yaw_std_ref) ** 2)

    # ------------------------------------------------------------
    # Task yaw update
    # ------------------------------------------------------------

    def _update_yaw_task(self, yaw_est, beta_target):
        def blend(prev, target, beta):
            return wrap_to_pi(prev + beta * angle_diff(target, prev))

        if not self.yaw_task_initialized:
            self.yaw_task = yaw_est
            self.beta = beta_target
            self.yaw_task_initialized = True
            return

        self.beta = (1 - self.yaw_gate_smooth) * self.beta + \
                    self.yaw_gate_smooth * beta_target

        self.yaw_task = blend(self.yaw_task, yaw_est, self.beta)

    # ------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------

    def update(self, pts_world: np.ndarray, dt: float = 0.01):
        if pts_world is None or len(pts_world) < self.min_points:
            return

        pts_world = np.asarray(pts_world, dtype=np.float32)

        if not self.ekf.initialized:
            self.ekf.initialize(centroid(pts_world), yaw0=0.0)
            return

        self.ekf.predict(dt)
        self.ekf.update_pos(centroid(pts_world))

        center, yaw_est = self.ekf.x[0:3], self.ekf.x[3]
        yaw_meas, anisotropy, xy_extent  = pca_yaw_xy(pts_world, center)
        if yaw_meas is None:
            return

        # sign disambiguation
        if np.cos(yaw_meas - yaw_est) < 0:
            yaw_meas = wrap_to_pi(yaw_meas + np.pi)

        r_mult = max(1.0, self.yaw_aniso_ref / max(anisotropy, 1e-6))
        self.ekf.update_yaw(yaw_meas, self.ekf.r_yaw_base * r_mult)

        yaw_std = np.sqrt(max(self.ekf.P[3, 3], 1e-9))

        self._update_yaw_observability(anisotropy, yaw_std, xy_extent)
        beta_target = self._compute_beta(yaw_std)
        self._update_yaw_task(self.ekf.x[3], beta_target)

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------

    def get_pose(self):
        if not self.ekf.initialized:
            return None, None
        return self.ekf.x[0:3].copy(), float(self.yaw_task)

    def get_yaw_observability(self):
        return self.yaw_observable, float(self.obs_score)

class ObjectFrameTrackerBaseline:
    """
    Baseline object tracker:
      - Position = centroid of current point cloud
      - Yaw = PCA yaw on XY plane at current timestep
      - No filtering, no temporal state
      - Yaw is always considered observable
    """

    def __init__(self, min_points=20):
        self.min_points = min_points

        self._center = None
        self._yaw = None

        # For API compatibility
        self.yaw_observable = True
        self.obs_score = 1.0

    # ------------------------------------------------------------
    # Main update (raw observation only)
    # ------------------------------------------------------------

    def update(self, pts_world: np.ndarray, dt: float = 0.01):
        if pts_world is None or len(pts_world) < self.min_points:
            return

        pts_world = np.asarray(pts_world, dtype=np.float32)

        # --- Position ---
        center = centroid(pts_world)

        # --- Yaw ---
        yaw_meas, anisotropy, xy_extent = pca_yaw_xy(pts_world, center)
        if yaw_meas is None:
            return

        self._center = center
        self._yaw = yaw_meas

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------

    def get_pose(self):
        if self._center is None or self._yaw is None:
            return None, None
        return self._center.copy(), float(self._yaw)

    def get_yaw_observability(self):
        return True, 1.0
