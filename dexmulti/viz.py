"""Minimal viser viewer for retrieval results (``pip install dexmulti[viz]``)."""

import numpy as np
from scipy.spatial.transform import Rotation as R

from dexmulti.config import DRAWER_PC_KEY, HAND_PC_KEY, OBJ_PC_KEY
from dexmulti.execution import replay_waypoints


class RetrievalViewer:
    def __init__(self, port: int = 8080):
        import viser

        self.server = viser.ViserServer(port=port)
        self.server.scene.add_grid("/ground", width=2.0, height=2.0, cell_size=0.1)
        print(f"[viz] open http://localhost:{port}")

    def show(self, episode, t, pos_seq, rot_seq, center, yaw, gt_pose=None, title=""):
        """Scene clouds at frame t + the retrieved skill mapped with (center, yaw)."""
        sc = self.server.scene
        colors = {OBJ_PC_KEY: (230, 120, 40), DRAWER_PC_KEY: (60, 130, 220), HAND_PC_KEY: (150, 150, 150)}
        for key, color in colors.items():
            if key in episode:
                pts = np.asarray(episode[key][t])[:, :3]
                sc.add_point_cloud(f"/scene/{key}", points=pts, colors=np.tile(color, (len(pts), 1)), point_size=0.004)
        waypoints = list(replay_waypoints(np.asarray(center, np.float32), yaw, pos_seq, rot_seq, [None] * len(pos_seq)))
        line = np.stack([w[0] for w in waypoints])
        sc.add_spline_catmull_rom("/retrieved/path", points=line, color=(20, 180, 90), line_width=3.0)
        for k in (0, len(waypoints) - 1):
            pos, rpy, _ = waypoints[k]
            wxyz = np.roll(R.from_euler("xyz", rpy).as_quat(), 1)
            sc.add_frame(f"/retrieved/ee_{k}", wxyz=wxyz, position=pos, axes_length=0.05, axes_radius=0.003)
        if gt_pose is not None:
            wxyz = np.roll(R.from_euler("xyz", gt_pose[3:6]).as_quat(), 1)
            sc.add_frame("/demo_grasp", wxyz=wxyz, position=gt_pose[:3], axes_length=0.07, axes_radius=0.002)
        print(f"[viz] {title}")
