"""
Serve a trained object-centric DP3 checkpoint over websocket (msgpack_numpy).

Usage (from baselines/object_centric_dp3/):
    python idp3/serve.py --ckpt outputs/<run>/checkpoints/latest.ckpt --port 8000

Replaces the fork's hydra-based eval.py: the policy config is read from the
checkpoint itself, so no overrides need to be repeated. Protocol (unchanged, see
server.py): on connect the server sends {n_obs_steps, n_action_steps,
n_latency_steps, device}; each request is an obs dict of float arrays shaped
(1, n_obs_steps, ...) and the reply is {"action": (1, n_action_steps, 28), ...}.
"""
import argparse
import asyncio
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import dill
import hydra
import torch
from omegaconf import OmegaConf
from termcolor import cprint

from server import WebsocketIDP3Server

OmegaConf.register_new_resolver("eval", eval, replace=True)


def load_policy(ckpt_path: str, device: str):
    ckpt_path = pathlib.Path(ckpt_path).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    payload = torch.load(ckpt_path.open('rb'), pickle_module=dill,
                         map_location='cpu', weights_only=False)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=str(ckpt_path.parent.parent))
    workspace.load_payload(payload, exclude_keys=('optimizer',))
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.eval()
    policy.to(device)
    policy.reset()
    cprint(f"[serve] loaded {ckpt_path} (epoch={workspace.epoch}, "
           f"pc_encoder={cfg.policy.pointnet_type}, "
           f"sharing={cfg.policy.pc_encoder_sharing})", "green")
    return policy, cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="path to checkpoints/latest.ckpt")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    policy, cfg = load_policy(args.ckpt, args.device)
    server = WebsocketIDP3Server(policy=policy, cfg=cfg, host=args.host, port=args.port)
    cprint("[serve] Starting async WebSocket server...", "cyan")
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
