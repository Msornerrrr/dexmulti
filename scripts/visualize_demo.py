#!/usr/bin/env python3
"""Interactive viewer for one released demonstration (``pip install dexmulti[viz,lerobot]``).

Shows the two camera videos, the object / second-object / hand point clouds (optionally
coloured by hand contact), the end-effector trajectory coloured by skill phase, the current
measured and commanded end-effector frames and the two cameras, with a timeline slider and
playback.

  python scripts/visualize_demo.py --dataset data/lerobot/dexmulti_grasp_pull --episode 0
  python scripts/visualize_demo.py --dataset HJ666/dexmulti_grasp_pull --episode session_20251227_221756
  python scripts/visualize_demo.py --dataset .../dexmulti_grasp_open_sample_efficiency --split quantity_5
  python scripts/visualize_demo.py --dataset ... --check 20     # load + decode 20 frames, no server

--episode is the dataset-wide episode_index (as in LeRobot) or the recording name; with --split it
must be an episode of that set (the error message lists them).
"""

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from dexmulti.config import DRAWER_CONTACT_KEY, DRAWER_PC_KEY, HAND_PC_KEY, OBJ_CONTACT_KEY, OBJ_PC_KEY
from dexmulti.data.lerobot import CAMERAS, LeRobotEpisodes

SKILL_NAMES = {0: "approach / transition", 1: "skill 1 (first object)", 2: "skill 2 (second object)"}
SKILL_COLORS = {0: (160, 160, 160), 1: (240, 130, 30), 2: (40, 120, 230)}
CLOUDS = {  # cloud key: (label, base colour, contact key)
    OBJ_PC_KEY: ("object", (240, 130, 30), OBJ_CONTACT_KEY),
    DRAWER_PC_KEY: ("drawer / second object", (40, 120, 230), DRAWER_CONTACT_KEY),
    HAND_PC_KEY: ("hand", (150, 150, 150), None),
}


def wxyz_from_rpy(rpy) -> np.ndarray:
    return np.roll(R.from_euler("xyz", rpy).as_quat(), 1)


def contact_colors(c: np.ndarray) -> np.ndarray:
    """Light grey (no contact) to red (contact = 1)."""
    c = np.clip(np.asarray(c, np.float32), 0.0, 1.0)[:, None]
    lo, hi = np.array([215, 215, 215], np.float32), np.array([220, 20, 20], np.float32)
    return ((1 - c) * lo + c * hi).astype(np.uint8)


class Demo:
    """One episode with decoded camera frames; :meth:`frame` returns what is drawn at step t."""

    def __init__(self, ds: LeRobotEpisodes, episode, n_frames=None, image_scale=0.5):
        self.ds, self.index = ds, ds.index(episode)
        self.name = ds.name(self.index)
        self.ep = ds.episode(self.index)
        self.kf = ds.keyframes(self.index)
        self.T = len(self.ep["time"])
        frames = None if n_frames is None else list(range(min(n_frames, self.T)))
        self.images = {c: ds.video_frames(self.index, c, frames=frames, scale=image_scale) for c in CAMERAS}
        self.n = min(len(v) for v in self.images.values())

    def frame(self, t: int, contact: bool):
        ep = self.ep
        clouds = {}
        for key, (_, base, ckey) in CLOUDS.items():
            pts = ep[key][t]
            col = contact_colors(ep[ckey][t]) if (contact and ckey) else np.tile(np.uint8(base), (len(pts), 1))
            clouds[key] = (pts, col)
        return {
            "clouds": clouds,
            "eef": (ep["eef_pose"][t, :3], wxyz_from_rpy(ep["eef_pose"][t, 3:6])),
            "cmd": (ep["arm_cmd"][t, :3], wxyz_from_rpy(ep["arm_cmd"][t, 3:6])),
            "skill": int(ep["skill"][t]),
            "images": {c: self.images[c][min(t, self.n - 1)] for c in CAMERAS},
        }

    def trajectory_segments(self):
        p = self.ep["eef_pose"][:, :3]
        seg = np.stack([p[:-1], p[1:]], axis=1)
        col = np.array([SKILL_COLORS[int(s)] for s in self.ep["skill"][1:]], np.uint8)
        return seg, np.repeat(col[:, None], 2, axis=1)

    def info_md(self, t: int) -> str:
        s = int(self.ep["skill"][t])
        kf = ""
        if self.kf:
            src = f" of set {self.ds.split}" if self.ds.split else ""
            kf = (f"\n\nkeyframes{src}: skill 1 [{self.kf['task1_start']}, {self.kf['task1_end']}], "
                  f"skill 2 [{self.kf['task2_start']}, {self.kf['task2_end']}]")
        return (f"**{self.name}** (episode {self.index})\n\nframe {t} / {self.T - 1} · "
                f"{t / self.ds.fps:.1f} s\n\n**{SKILL_NAMES[s]}**{kf}")


def run_viewer(ds: LeRobotEpisodes, episode, port: int, image_scale: float):
    import viser

    server = viser.ViserServer(port=port)
    sc = server.scene
    sc.add_grid("/ground", width=1.5, height=1.5, cell_size=0.1, position=(0.4, 0.0, 0.0))
    sc.add_frame("/base", axes_length=0.1, axes_radius=0.004)
    cams = ds.cameras or {}
    state = {"demo": None}

    gui = server.gui
    names = ds.names
    pick = gui.add_dropdown("episode", names, initial_value=ds.name(ds.index(episode)))
    slider = gui.add_slider("frame", min=0, max=1, step=1, initial_value=0)
    playing = gui.add_checkbox("play", False)
    speed = gui.add_number("playback x", 1.0, min=0.1, max=10.0, step=0.1)
    contact = gui.add_checkbox("colour by contact", True)
    show_frustum_img = gui.add_checkbox("images in 3D", False)
    info = gui.add_markdown("")
    img_handles = {c: gui.add_image(np.zeros((8, 8, 3), np.uint8), label=c, format="jpeg") for c in CAMERAS}

    def draw_cameras(images=None):
        for c in CAMERAS:
            p = cams.get(c)
            if p is None:
                continue
            T = np.asarray(p["T_base_cam"])  # OpenCV optical frame -> base, as viser's frustum
            sc.add_camera_frustum(f"/cameras/{c}", fov=2 * np.arctan2(240.0, p["fx_fy_cx_cy"][1]), aspect=640 / 480,
                                  scale=0.08, wxyz=np.roll(R.from_matrix(T[:3, :3]).as_quat(), 1), position=T[:3, 3],
                                  image=None if images is None else images[c], format="jpeg")

    def draw(t: int):
        d = state["demo"]
        if d is None:
            return
        f = d.frame(t, contact.value)
        for key, (pts, col) in f["clouds"].items():
            sc.add_point_cloud(f"/clouds/{CLOUDS[key][0]}", points=pts, colors=col, point_size=0.004)
        sc.add_frame("/ee/measured", wxyz=f["eef"][1], position=f["eef"][0], axes_length=0.06, axes_radius=0.004)
        sc.add_frame("/ee/commanded", wxyz=f["cmd"][1], position=f["cmd"][0], axes_length=0.035, axes_radius=0.002)
        for c, img in f["images"].items():
            img_handles[c].image = img
        if show_frustum_img.value:
            draw_cameras(f["images"])
        info.content = d.info_md(t)

    def load(name):
        print(f"[viz] loading {name} ...")
        d = Demo(ds, name, image_scale=image_scale)
        state["demo"] = d
        seg, col = d.trajectory_segments()
        sc.add_line_segments("/ee/trajectory", points=seg, colors=col, thickness=3.0, thickness_units="screen")
        slider.max = d.T - 1
        slider.value = 0
        draw(0)

    pick.on_update(lambda _: load(pick.value))
    slider.on_update(lambda _: draw(int(slider.value)))
    contact.on_update(lambda _: draw(int(slider.value)))
    show_frustum_img.on_update(lambda _: draw_cameras() if not show_frustum_img.value else draw(int(slider.value)))
    load(pick.value)
    draw_cameras()
    print(f"[viz] open http://localhost:{port}")
    while True:
        d = state["demo"]
        if playing.value and d is not None:
            slider.value = (int(slider.value) + 1) % d.T
        time.sleep(1.0 / (ds.fps * max(speed.value, 0.1)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="local LeRobot dataset folder or Hugging Face dataset id")
    p.add_argument("--episode", default=None,
                   help="dataset-wide episode_index or recording name (default: first episode of the dataset / split)")
    p.add_argument("--split", default=None, help="sample-efficiency set (episode list + keyframes of that set)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--image-scale", type=float, default=0.5, help="resize factor of the camera images in the GUI")
    p.add_argument("--check", type=int, default=None, metavar="N",
                   help="load the episode, decode N frames per camera, build the scene data and exit")
    a = p.parse_args()

    try:
        ds = LeRobotEpisodes(a.dataset, split=a.split, download_videos=not Path(a.dataset).expanduser().exists())
    except (FileNotFoundError, KeyError) as e:
        raise SystemExit(f"error: {e.args[0] if e.args else e}")
    episode = ds.indices[0] if a.episode is None else int(a.episode) if a.episode.isdigit() else a.episode
    try:
        episode = ds.index(episode)
    except (KeyError, IndexError) as e:
        raise SystemExit(f"error: {e.args[0]}")
    if a.check is not None:
        d = Demo(ds, episode, n_frames=a.check, image_scale=a.image_scale)
        for t in range(d.n):
            f = d.frame(t, contact=True)
            assert all(np.isfinite(pts).all() and len(pts) == len(col) for pts, col in f["clouds"].values())
        seg, col = d.trajectory_segments()
        print(f"[check] {d.name}: T={d.T}, decoded {d.n} frames/camera "
              f"{ {c: v.shape[1:] for c, v in d.images.items()} }, trajectory {len(seg)} segments, "
              f"skills {np.bincount(d.ep['skill'], minlength=3).tolist()}, cameras {sorted(ds.cameras or {})}")
        return
    run_viewer(ds, episode, a.port, a.image_scale)


if __name__ == "__main__":
    main()
