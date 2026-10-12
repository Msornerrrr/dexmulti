"""
ROS-free deploy client for the object-centric DP3 policy server (idp3/serve.py).

Protocol (msgpack_numpy over websockets, unchanged from the original system):
  * on connect the server sends metadata {n_obs_steps, n_action_steps, n_latency_steps, device}
  * request : {"eef_pos": (1,To,3), "eef_rot9d": (1,To,9), "hand_joint": (1,To,16),
               "object_pc": (1,To,N,C), "drawer_pc": (1,To,N,C)}   float32
  * reply   : {"action": (1, n_action_steps, 28), "server_timing": {...}}
    action = [pos(3), rot9 (row-major 3x3), hand(16)], absolute, robot base frame.

`AsyncChunkRunner` re-implements the original async action-chunk executor: it executes
the current chunk at `control_hz` and requests the next chunk `n_latency_steps` before the
current one runs out, swapping chunks at the horizon boundary (holding the last action if
the next chunk is late). Plug in any robot with

    get_eef_pose() -> (6,) [x, y, z, roll, pitch, yaw]  (euler "xyz", rad, base frame)
    get_hand_joints() -> (16,)
    send_eef_pose(pos (3,), rpy (3,))
    send_hand(q (16,))

and an observation callback returning the segmented point clouds, e.g.
`lambda: {"object_pc": (512, C), "drawer_pc": (512, C)}` (C=3 xyz or 4 xyzc, base frame).

Quick check against a running server:  python client.py --uri ws://localhost:8000 --pc-channels 4
"""
import asyncio
import time
from collections import deque
from typing import Callable, Dict, Optional, Protocol

import msgpack_numpy as m
import numpy as np
import websockets
from scipy.spatial.transform import Rotation as R

PC_KEYS = ("object_pc", "drawer_pc")
ACTION_DIM = 28


class Robot(Protocol):
    def get_eef_pose(self) -> np.ndarray: ...
    def get_hand_joints(self) -> np.ndarray: ...
    def send_eef_pose(self, pos: np.ndarray, rpy: np.ndarray) -> None: ...
    def send_hand(self, q: np.ndarray) -> None: ...


# --------------------------------------------------------------------------- client
class DP3Client:
    def __init__(self, uri: str = "ws://localhost:8000"):
        self.uri = uri
        self.ws = None
        self.meta: Optional[dict] = None
        self._packer = m.Packer()

    async def connect(self) -> dict:
        self.ws = await websockets.connect(self.uri, max_size=None, compression=None)
        self.meta = m.unpackb(await self.ws.recv())
        return self.meta

    def mark_stale(self) -> None:
        """Drop the connection immediately (no close handshake, so it cannot block). A reply
        still queued for an abandoned request goes with it; the next infer() reconnects."""
        if self.ws is not None:
            try:
                self.ws.transport.abort()
            except Exception:
                pass
        self.ws = None

    async def infer(self, obs: Dict[str, np.ndarray]) -> dict:
        if self.ws is None:  # lazily reconnect after mark_stale()
            await self.connect()
        await self.ws.send(self._packer.pack(obs))
        resp = await self.ws.recv()
        try:
            out = m.unpackb(resp)
        except Exception:
            out = None
        if not isinstance(out, dict):  # the server sends a plain traceback on error
            text = resp.decode(errors="replace") if isinstance(resp, bytes) else str(resp)
            raise RuntimeError(f"policy server error:\n{text}")
        return out

    async def close(self):
        if self.ws is not None:
            await self.ws.close()
            self.ws = None

    async def reconnect(self) -> dict:
        self.mark_stale()
        return await self.connect()


# --------------------------------------------------------------- obs / action utils
def eef_pose_to_pos_rot9d(eef_pose):
    eef_pose = np.asarray(eef_pose, dtype=np.float64)
    return eef_pose[:3], R.from_euler("xyz", eef_pose[3:6]).as_matrix().reshape(9)


def unpack_action(action):
    """(28,) [pos3, rot9, hand16] -> pos (3,), rpy euler xyz (3,), hand (16,)."""
    a = np.asarray(action, dtype=np.float64)
    return a[:3], R.from_matrix(a[3:12].reshape(3, 3)).as_euler("xyz"), a[12:]


class ObsHistory:
    """Keeps the last n_obs_steps observations; the first one is repeated to fill the window."""

    def __init__(self, n_obs_steps: int, pc_keys=PC_KEYS):
        self.n = n_obs_steps
        self.pc_keys = tuple(pc_keys)
        self.reset()

    def reset(self):
        self.pos, self.rot, self.hand = (deque(maxlen=self.n) for _ in range(3))
        self.pcs = {k: deque(maxlen=self.n) for k in self.pc_keys}

    def push(self, eef_pose, hand_joint, pcs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        pos, rot9 = eef_pose_to_pos_rot9d(eef_pose)
        hand = np.asarray(hand_joint)
        while True:
            self.pos.append(pos)
            self.rot.append(rot9)
            self.hand.append(hand)
            for k in self.pc_keys:
                self.pcs[k].append(np.asarray(pcs[k]))
            if len(self.pos) == self.n:
                break
        return self.obs()

    def obs(self) -> Dict[str, np.ndarray]:
        def stack(d):
            return np.stack(d)[None].astype(np.float32)
        out = {"eef_pos": stack(self.pos), "eef_rot9d": stack(self.rot), "hand_joint": stack(self.hand)}
        out.update({k: stack(self.pcs[k]) for k in self.pc_keys})
        return out


# ---------------------------------------------------------------------------- runner
class AsyncChunkRunner:
    def __init__(self, client: DP3Client, robot: Robot,
                 get_point_clouds: Callable[[], Dict[str, np.ndarray]],
                 n_obs_steps: int, n_action_steps: int, n_latency_steps: int,
                 control_hz: float = 10.0, pc_keys=PC_KEYS,
                 is_shutdown: Callable[[], bool] = lambda: False,
                 infer_timeout: float = 3.0):
        assert n_action_steps > n_latency_steps, "n_action_steps must be > n_latency_steps"
        self.client, self.robot, self.get_point_clouds = client, robot, get_point_clouds
        self.n_action_steps, self.n_latency_steps = n_action_steps, n_latency_steps
        self.dt = 1.0 / control_hz
        self.history = ObsHistory(n_obs_steps, pc_keys)
        self.is_shutdown = is_shutdown
        self.infer_timeout = infer_timeout

    @classmethod
    def from_meta(cls, client: DP3Client, robot: Robot, get_point_clouds, **kw):
        meta = client.meta
        return cls(client, robot, get_point_clouds, int(meta["n_obs_steps"]),
                   int(meta["n_action_steps"]), int(meta["n_latency_steps"]), **kw)

    # -- robot I/O
    def observe(self):
        self.latest_obs = self.history.push(self.robot.get_eef_pose(), self.robot.get_hand_joints(),
                                            self.get_point_clouds())

    def execute(self, action):
        pos, rpy, hand = unpack_action(action)
        self.robot.send_eef_pose(pos, rpy)
        self.robot.send_hand(hand)

    # -- inference worker
    async def _worker(self):
        while not self._stopping:
            await self._infer_event.wait()
            self._infer_event.clear()
            if self._stopping:
                break
            if self.latest_obs is None:
                continue
            self._inflight = True
            obs = {k: v.copy() for k, v in self.latest_obs.items()}
            try:
                t0 = time.monotonic()
                result = await asyncio.wait_for(self.client.infer(obs), timeout=self.infer_timeout)
                self.last_infer_s = time.monotonic() - t0
                self._next = list(np.asarray(result["action"])[0, :self.n_action_steps])
                self._next_ready = True
                self._first_ready.set()
            except asyncio.TimeoutError:
                # The late reply would otherwise be read by the *next* request and every
                # following chunk would lag one observation behind: drop the connection;
                # the next request reconnects.
                print("[AsyncChunkRunner] inference timeout, dropping connection")
                self.client.mark_stale()
            finally:
                self._inflight = False

    def _swap(self):
        self._current, self._next, self._next_ready = self._next, [], False
        self._idx = 0
        self._triggered = False

    async def run(self, rollout_len: int = 500, early_stop_fn: Callable[[], bool] = lambda: False):
        self.history.reset()
        self.latest_obs = None
        self._infer_event, self._first_ready = asyncio.Event(), asyncio.Event()
        self._current, self._next = [], []
        self._next_ready = self._inflight = self._triggered = self._stopping = False
        self._idx = 0
        step = 0
        worker = asyncio.create_task(self._worker())
        try:
            # warm-up: keep observing until the first chunk arrives
            self.observe()
            self._infer_event.set()
            while not self._first_ready.is_set() and not self.is_shutdown():
                await asyncio.sleep(self.dt)
                if worker.done():
                    worker.result()  # re-raise server / connection errors
                self.observe()
                if not self._inflight and not self._infer_event.is_set() and not self._first_ready.is_set():
                    self._infer_event.set()  # first request was dropped (e.g. timeout): ask again
            if not self._first_ready.is_set():
                return step
            self._swap()

            while step < rollout_len and not self.is_shutdown() and not early_stop_fn():
                t_start = time.monotonic()
                if worker.done():
                    worker.result()
                if self._idx >= self.n_action_steps:  # horizon boundary
                    if self._next_ready:
                        self._swap()
                    elif not self._inflight and not self._infer_event.is_set():
                        self._infer_event.set()  # fallback request
                self.observe()
                if (self._idx >= self.n_action_steps - self.n_latency_steps and not self._triggered
                        and not self._inflight and not self._next_ready):
                    self._infer_event.set()
                    self._triggered = True
                action = self._current[min(self._idx, len(self._current) - 1)]
                self.execute(action)
                self._idx += 1
                step += 1
                await asyncio.sleep(max(0.0, self.dt - (time.monotonic() - t_start)))
        finally:
            inflight = self._inflight
            self._stopping = True
            self._infer_event.set()
            worker.cancel()
            await asyncio.wait({worker}, timeout=1.0)
            if inflight:  # its reply is still queued; never let it start the next rollout
                self.client.mark_stale()
        return step


def make_xyzc(object_cloud: np.ndarray, hand_cloud: np.ndarray, n_points: int = 512) -> np.ndarray:
    """(N,3) object cloud + (M,3) hand surface cloud -> (n_points, 4) xyz + contact, as in the
    training data: finite points, farthest-point sampling to n_points (repeat-padded), then the
    soft contact map. The hand cloud must match the data's ``hand_point_cloud_base`` (512 points
    sampled on the hand link surfaces via forward kinematics). Requires ``dexmulti[perception]``."""
    from dexmulti.perception.contact import contact_map
    from dexmulti.perception.denoise import farthest_point_sample

    obj = np.asarray(object_cloud, dtype=np.float32)[:, :3]
    obj = farthest_point_sample(obj[np.isfinite(obj).all(axis=1)], n_points)
    return np.concatenate([obj, contact_map(obj, np.asarray(hand_cloud)[:, :3])[:, None]], axis=1).astype(np.float32)


# ----------------------------------------------------------------------- self-check
async def _check(uri: str, pc_channels: int, n_points: int):
    client = DP3Client(uri)
    meta = await client.connect()
    print("server metadata:", meta)
    hist = ObsHistory(int(meta["n_obs_steps"]))
    rng = np.random.default_rng(0)
    pcs = {k: rng.uniform(-0.3, 0.3, (n_points, pc_channels)) for k in PC_KEYS}
    obs = hist.push(np.array([0.35, 0.0, 0.4, np.pi, 0.0, 0.0]), np.zeros(16), pcs)
    result = await client.infer(obs)
    action = np.asarray(result["action"])
    await client.close()
    assert action.shape == (1, int(meta["n_action_steps"]), ACTION_DIM), action.shape
    pos, rpy, hand = unpack_action(action[0, 0])
    print(f"action {action.shape} first step: pos={pos.round(3)} rpy={rpy.round(3)} hand[:4]={hand[:4].round(3)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Round-trip a dummy observation through the policy server.")
    ap.add_argument("--uri", default="ws://localhost:8000")
    ap.add_argument("--pc-channels", type=int, default=4, help="3 for xyz, 4 for xyzc policies")
    ap.add_argument("--n-points", type=int, default=512)
    args = ap.parse_args()
    asyncio.run(_check(args.uri, args.pc_channels, args.n_points))
