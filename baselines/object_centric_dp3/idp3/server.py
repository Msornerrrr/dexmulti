"""Websocket inference server (from the iDP3 fork; protocol unchanged).

DexMulti release: metadata "device" reports the device the policy actually runs on
(the fork reported cfg.training.device); unused import removed.
"""
import asyncio
import logging
import time
import traceback
import torch
import websockets.asyncio.server as ws_server
import websockets.frames
import msgpack_numpy as m
from termcolor import cprint

class WebsocketIDP3Server:
    """Async WebSocket server for IDP3 policy inference."""
    def __init__(self, policy, cfg, host="0.0.0.0", port=8000):
        self.policy = policy
        self.cfg = cfg
        self.host = host
        self.port = port

    async def run(self):
        async with ws_server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            process_request=self._health_check,
        ) as server:
            cprint(f"[IDP3Server] Listening on ws://{self.host}:{self.port}", "cyan")
            await server.serve_forever()

    async def _handler(self, websocket: ws_server.ServerConnection):
        logger = logging.getLogger("idp3.server")
        logger.info(f"Connection from {websocket.remote_address} opened")

        packer = m.Packer()
        metadata = {
            "n_obs_steps": self.cfg.policy.n_obs_steps,
            "n_action_steps": self.cfg.policy.n_action_steps,
            "n_latency_steps": self.cfg.n_latency_steps,
            "device": str(self.policy.device),
        }
        await websocket.send(packer.pack(metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = m.unpackb(await websocket.recv())
                return_attn = obs.pop("_debug_return_attn", False)

                # Convert to tensors
                obs_t = {
                    k: torch.from_numpy(v).float().to(self.policy.device)
                    for k, v in obs.items()
                }

                infer_t = time.monotonic()
                with torch.no_grad():
                    result = self.policy.predict_action(obs_t, return_attn=return_attn)
                infer_time = time.monotonic() - infer_t
                cprint(f"[Server] Pure inference took {infer_time*1000:.2f} ms", "cyan")

                action = result["action"].cpu().numpy()

                payload = {
                    "action": action,
                    "server_timing": {
                        "infer_ms": infer_time * 1000,
                        "prev_total_ms": (prev_total_time * 1000) if prev_total_time else None,
                    },
                }
                if return_attn and "attention" in result:
                    payload["attention"] = {}

                    attn_dict = result["attention"]
                    for key, val in attn_dict.items():
                        # val: torch.Tensor
                        payload["attention"][key] = val.detach().cpu().numpy()

                await websocket.send(packer.pack(payload))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection {websocket.remote_address} closed")
                break
            except Exception:
                tb = traceback.format_exc()
                await websocket.send(tb.encode())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal error; traceback sent."
                )
                raise

    def _health_check(self, connection, request):
        if request.path == "/healthz":
            return connection.respond(200, "OK\n")
        return None
