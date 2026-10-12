"""Multi-sphere collision envelope of an object point cloud (K spheres along the
principal axis, critically damped smoothing). Used as planner collision geometry."""

import numpy as np


def safe_norm(x, axis=-1, eps=1e-9):
    return np.sqrt(np.sum(x * x, axis=axis) + eps)


class CriticallyDampedFilter:
    """
    Numerically stable critically damped second-order filter.
    Unconditionally stable for any dt > 0.
    """

    def __init__(self, x0, wn: float):
        self.x = np.array(x0, dtype=np.float32)
        self.v = np.zeros_like(self.x)
        self.wn = float(wn)

    def update(self, x_target, dt: float):
        wn = self.wn
        dt = max(dt, 1e-4)

        k1 = 2.0 / (dt * wn)
        k2 = 1.0 / (dt * dt * wn * wn)

        denom = 1.0 + k1 + k2
        x_new = (self.x + dt * self.v + k2 * x_target) / denom
        v_new = (x_new - self.x) / dt

        self.x = x_new
        self.v = v_new
        return self.x


class MultiSphereEnvelope:
    """
    Medial-axis multi-sphere envelope for SMALL K (4–16).

    Final version:
      - Medial centers
      - Thickness + axial extent radius
      - Critically damped temporal smoothing
      - Conservative but not under-sized
    """

    def __init__(
        self,
        k: int,
        radius_percentile: float = 30.0,
        axial_mix: float = 0.8,          # weight on orthogonal thickness
        inward_scale: float = 0.95,
        min_points_per_bin: int = 40,
        wn_center: float = 8.0,
        wn_radius: float = 8.0,
        max_radius_growth: float = 1.25, # per-update clamp
    ):
        self.k = int(k)
        self.radius_percentile = float(radius_percentile)
        self.axial_mix = float(axial_mix)
        self.inward_scale = float(inward_scale)
        self.min_points_per_bin = int(min_points_per_bin)
        self.max_radius_growth = float(max_radius_growth)

        self.wn_center = float(wn_center)
        self.wn_radius = float(wn_radius)

        self._initialized = False
        self.axis = None
        self.bin_edges = None

        self.centers = None
        self.radii = None
        self._center_filters = []
        self._radius_filters = []

    # ------------------------------------------------------------
    # PCA helpers
    # ------------------------------------------------------------
    @staticmethod
    def principal_axis(pts):
        X = pts - pts.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        return Vt

    # ------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------
    def initialize(self, pts: np.ndarray):
        pts = np.asarray(pts, dtype=np.float32)

        Vt = self.principal_axis(pts)
        self.axis = Vt[0]
        self.axis /= safe_norm(self.axis)

        proj = pts @ self.axis
        self.bin_edges = np.percentile(
            proj, np.linspace(0.0, 100.0, self.k + 1)
        ).astype(np.float32)

        self.centers = np.zeros((self.k, 3), dtype=np.float32)
        self.radii = np.zeros((self.k,), dtype=np.float32)

        self._center_filters = []
        self._radius_filters = []

        for _ in range(self.k):
            self._center_filters.append(
                CriticallyDampedFilter(np.zeros(3), self.wn_center)
            )
            self._radius_filters.append(
                CriticallyDampedFilter(np.zeros(1), self.wn_radius)
            )

        self._initialized = True
        self.update(pts, dt=0.01)

    # ------------------------------------------------------------
    # Update
    # ------------------------------------------------------------
    def update(self, pts: np.ndarray, dt: float):
        if pts.shape[0] < self.min_points_per_bin:
            return

        if not self._initialized:
            self.initialize(pts)
            return

        pts = np.asarray(pts, dtype=np.float32)
        proj = pts @ self.axis

        for i in range(self.k):
            mask = (proj >= self.bin_edges[i]) & (proj <= self.bin_edges[i + 1])
            bin_pts = pts[mask]

            if bin_pts.shape[0] < self.min_points_per_bin:
                continue

            # ---- local PCA ----
            mu = bin_pts.mean(axis=0, keepdims=True)
            X = bin_pts - mu
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            local_axis = Vt[0]
            local_axis /= safe_norm(local_axis)

            # ---- medial center ----
            t = mu[0] @ local_axis
            c_est = t * local_axis

            # ---- orthogonal thickness ----
            v = bin_pts - c_est
            ortho = v - (v @ local_axis)[:, None] * local_axis[None, :]
            d_perp = safe_norm(ortho, axis=1)
            r_perp = np.percentile(d_perp, self.radius_percentile)

            # ---- axial extent ----
            axial_proj = bin_pts @ local_axis
            r_axial = 0.5 * (axial_proj.max() - axial_proj.min())

            # ---- combined radius ----
            r_est = (
                self.axial_mix * r_perp
                + (1.0 - self.axial_mix) * r_axial
            )
            r_est *= self.inward_scale

            # ---- smooth + clamp ----
            c_s = self._center_filters[i].update(c_est, dt)
            r_s = self._radius_filters[i].update(
                np.array([r_est], dtype=np.float32), dt
            )[0]

            if self.radii[i] > 0:
                r_s = min(r_s, self.max_radius_growth * self.radii[i])

            self.centers[i] = c_s
            self.radii[i] = r_s

    # ------------------------------------------------------------
    # API
    # ------------------------------------------------------------
    def get_spheres(self):
        if not self._initialized:
            return np.zeros((0, 3)), np.zeros((0,))
        return self.centers.copy(), self.radii.copy()
