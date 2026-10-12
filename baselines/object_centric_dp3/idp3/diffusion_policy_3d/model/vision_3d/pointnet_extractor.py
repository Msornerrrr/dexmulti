import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
import diffusion_policy_3d.model.vision_3d.point_process as point_process


# ============================================================
# MLP Utility
# ============================================================

def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), i.e., a stack of Linear + activation.
    """
    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


# ============================================================
# Fourier Positional Encoding for 3D Points
# ============================================================

def fourier_encode_3d(points: torch.Tensor, L: int = 10, include_input: bool = False):
    """
    Apply Fourier positional encoding to 3D coordinates.
    Args:
        points: (..., 3) xyz coordinates
        L: number of frequency bands
        include_input: whether to concatenate original xyz
    Returns:
        (..., 3 * 2 * L [+3])
    """
    freq = (2.0 ** torch.arange(L, device=points.device, dtype=points.dtype))
    angles = 2 * math.pi * points[..., None] * freq  # (..., 3, L)
    sin, cos = torch.sin(angles), torch.cos(angles)
    encoded = torch.cat([sin, cos], dim=-1)  # (..., 3, 2L)
    encoded = encoded.reshape(*points.shape[:-1], 3 * 2 * L)
    if include_input:
        encoded = torch.cat([points, encoded], dim=-1)
    return encoded


# ============================================================
# Meta Helpers (robust to dict / OmegaConf / simple objects)
# ============================================================

def _get_meta_type(meta) -> Optional[str]:
    # Supports both attribute access and dict-style configs
    t = getattr(meta, "type", None)
    if t is not None:
        return t
    if isinstance(meta, dict):
        return meta.get("type", None)
    return None


def _get_meta_shape(meta):
    s = getattr(meta, "shape", None)
    if s is not None:
        return s
    if isinstance(meta, dict) and "shape" in meta:
        return meta["shape"]
    raise ValueError(f"Cannot retrieve 'shape' from meta: {meta}")


# ============================================================
# iDP3 Encoder (multi-pointcloud + low-dim)
# ============================================================

class iDP3Encoder(nn.Module):
    """
    iDP3Encoder

    - Handles one or multiple point clouds, indicated in obs_shape_meta by type="point_cloud".
    - Encodes low-dimensional state features indicated by type="low_dim".
    - Combines point cloud features and low-dim features into a single embedding.

    Design:
      * Shared PointNet backbone for all point clouds.
      * Pluggable fusion for multi-pointcloud:
          - "mean"   : average view features
          - "sum"    : sum view features
          - "concat" : concatenate view features along channel dim
      * Low-dim encoding modes:
          - "concat_encode" : concat all low-dim inputs, single MLP
          - "encode_concat" : per-key MLP, then concat + fusion MLP
    """

    def __init__(self,
                 obs_shape_meta: Dict,
                 state_mlp_size=(64, 64),
                 state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_fourier_basis: bool = False,
                 pointnet_type: str = 'dp3_encoder',  # kept for compatibility (currently unused)
                 point_downsample: bool = False,
                 low_dim_mode: str = "concat_encode",   # or "encode_concat"
                 per_key_dim: int = 32,
                 pc_fusion_mode: str = "concat",          # "mean", "sum", "concat"
                 pc_encoder_sharing="shared",         # "shared", "per_key", "per_group"
                 pc_groups=None
                 ):
        super().__init__()
        self.low_dim_mode = low_dim_mode
        self.use_fourier_basis = use_fourier_basis
        self.pc_fusion_mode = pc_fusion_mode

        # ------------------- Parse obs_shape_meta -------------------
        # Pointcloud keys (can be 1 or many)
        self.pc_keys = [
            k for k, v in obs_shape_meta.items()
            if _get_meta_type(v) == "point_cloud"
        ]
        if len(self.pc_keys) == 0:
            # Backwards-compat fallback
            if "point_cloud" in obs_shape_meta:
                self.pc_keys = ["point_cloud"]
            else:
                raise ValueError(
                    "iDP3Encoder: No point_cloud entries found in obs_shape_meta "
                    "and 'point_cloud' not present as a key."
                )

        # Low-dim keys
        self.low_dim_keys = [
            k for k, v in obs_shape_meta.items()
            if _get_meta_type(v) == "low_dim"
        ]

        cprint(f"[iDP3Encoder] pointcloud keys: {self.pc_keys}", "yellow")
        cprint(f"[iDP3Encoder] low-dim keys  : {self.low_dim_keys}", "yellow")

        # Store shapes
        # Assume all pointclouds share the same feature dimension
        first_pc_shape = _get_meta_shape(obs_shape_meta[self.pc_keys[0]])
        self.point_cloud_shape = first_pc_shape
        self.low_dim_shapes = {k: _get_meta_shape(obs_shape_meta[k]) for k in self.low_dim_keys}

        # ------------------- Point Cloud Encoder Backbone -------------------
        self.L_fourier = getattr(pointcloud_encoder_cfg, "fourier_L", 10)

        input_channels = self.point_cloud_shape[-1]
        if self.use_fourier_basis:
            num_coord = 3
            num_extra = input_channels - num_coord
            input_channels = (3 * 2 * self.L_fourier) + num_extra

        self.num_points = getattr(pointcloud_encoder_cfg, "num_points", 4096)
        out_channels = getattr(pointcloud_encoder_cfg, "out_channels", 128)
        self.pc_out_channels = out_channels  # per-view feature dim before fusion

        # ===========================
        # Build PointNet encoders
        # ===========================
        self.pc_encoder_sharing = pc_encoder_sharing
        if pc_encoder_sharing == "shared":
            # One shared encoder
            self.shared_pc_encoder = self._build_pointnet_backbone(
                pointnet_type=pointnet_type,
                input_channels=input_channels,
                out_channels=out_channels,
                pointcloud_encoder_cfg=pointcloud_encoder_cfg
            )

        elif pc_encoder_sharing == "per_key":
            # One encoder per pointcloud key
            self.pc_encoders = nn.ModuleDict()
            for key in self.pc_keys:
                self.pc_encoders[key] = self._build_pointnet_backbone(
                pointnet_type=pointnet_type,
                input_channels=input_channels,
                out_channels=out_channels,
                pointcloud_encoder_cfg=pointcloud_encoder_cfg
            )

        elif pc_encoder_sharing == "per_group":
            # e.g., {"hand": ["hand_points"], "object": ["obj_pc", "obj_pc2"]}
            self.pc_groups = pc_groups
            self.pc_encoders = nn.ModuleDict()
            unique_groups = set(pc_groups.values())
            for grp in unique_groups:
                self.pc_encoders[grp] = self._build_pointnet_backbone(
                pointnet_type=pointnet_type,
                input_channels=input_channels,
                out_channels=out_channels,
                pointcloud_encoder_cfg=pointcloud_encoder_cfg
            )
        else:
            raise ValueError(f"Invalid pc_encoder_sharing={pc_encoder_sharing}")

        self.downsample = point_downsample
        if self.downsample:
            self.point_preprocess = point_process.uniform_sampling_torch
        else:
            self.point_preprocess = nn.Identity()

        # ------------------- Low-Dim Encoder -------------------
        self._build_low_dim_encoder(
            state_mlp_size=state_mlp_size,
            state_mlp_activation_fn=state_mlp_activation_fn,
            per_key_dim=per_key_dim
        )

        # ------------------- Combined Feature Dim -------------------
        # Pointcloud feature dim after fusion
        if self.pc_fusion_mode == "concat" and len(self.pc_keys) > 1:
            pc_feat_dim = self.pc_out_channels * len(self.pc_keys)
        else:
            pc_feat_dim = self.pc_out_channels

        self.n_output_channels = pc_feat_dim + self.state_output_dim
        cprint(f"[iDP3Encoder] output dim: {self.n_output_channels}", "red")

    # ============================================================
    # Backbone builder
    # ============================================================

    def _build_pointnet_backbone(self,
                                 pointnet_type: str,
                                 input_channels: int,
                                 out_channels: int,
                                 pointcloud_encoder_cfg):
        """
        Factory for the point cloud backbone. Shared across PCs.
        DexMulti release: only multi_stage_pointnet and better_attention_pointnet
        are vendored (attention/seg PointNet++ variants removed).
        """
        if pointnet_type == "multi_stage_pointnet":
            from .multi_stage_pointnet import MultiStagePointNetEncoder
            return MultiStagePointNetEncoder(
                in_channels=input_channels, out_channels=out_channels
            )
        elif pointnet_type == "better_attention_pointnet":
            from .better_attention_pointnet import BetterAttentionPointNetEncoder
            return BetterAttentionPointNetEncoder(
                in_channels=input_channels,
                out_channels=out_channels,
            )
        else:
            raise NotImplementedError(f"Unsupported pointnet_type: {pointnet_type}")

    # ============================================================
    # Low-Dim Encoder builder
    # ============================================================

    def _build_low_dim_encoder(self,
                               state_mlp_size,
                               state_mlp_activation_fn,
                               per_key_dim: int):
        """
        Build low-dimensional state encoder, with two modes:
          - concat_encode
          - encode_concat
        """
        if len(self.low_dim_keys) == 0:
            # No low-dim state – degenerate case.
            self.state_output_dim = 0
            self.state_mlp = None
            self.state_mlps = nn.ModuleDict()
            self.fusion_mlp = None
            return

        if self.low_dim_mode == "concat_encode":
            total_input_dim = sum(self.low_dim_shapes[k][0] for k in self.low_dim_keys)
            net_arch = state_mlp_size[:-1]
            output_dim = state_mlp_size[-1]
            self.state_mlp = nn.Sequential(
                *create_mlp(total_input_dim, output_dim, net_arch, state_mlp_activation_fn)
            )
            self.state_output_dim = output_dim
            self.state_mlps = nn.ModuleDict()
            self.fusion_mlp = None

        elif self.low_dim_mode == "encode_concat":
            # Per-key encoders
            self.state_mlps = nn.ModuleDict()
            for key in self.low_dim_keys:
                shape = self.low_dim_shapes[key]
                self.state_mlps[key] = nn.Sequential(
                    *create_mlp(shape[0], per_key_dim, state_mlp_size[:-1], state_mlp_activation_fn)
                )
            total_concat_dim = per_key_dim * len(self.low_dim_keys)
            self.fusion_mlp = nn.Sequential(
                *create_mlp(total_concat_dim, state_mlp_size[-1], [], state_mlp_activation_fn)
            )
            self.state_output_dim = state_mlp_size[-1]
            self.state_mlp = None

        else:
            raise ValueError(f"Invalid low_dim_mode: {self.low_dim_mode}")

    # ============================================================
    # Forward: Point Cloud encoding + Fusion + Low-Dim encoding
    # ============================================================

    def _encode_single_pointcloud(self, points: torch.Tensor, pc_key, return_attn: bool = False):
        """
        Apply preprocessing, optional Fourier encoding, and backbone extractor
        to a single pointcloud tensor of shape [B, N, C].
        """
        if self.downsample:
            points = self.point_preprocess(points, self.num_points)

        if self.use_fourier_basis:
            xyz = points[..., :3]
            extra = points[..., 3:] if points.shape[-1] > 3 else None
            xyz_encoded = fourier_encode_3d(xyz, L=self.L_fourier, include_input=False)
            points = torch.cat([xyz_encoded, extra], dim=-1) if extra is not None else xyz_encoded

        if self.pc_encoder_sharing == "shared":
            enc = self.shared_pc_encoder

        elif self.pc_encoder_sharing == "per_key":
            enc = self.pc_encoders[pc_key]

        elif self.pc_encoder_sharing == "per_group":
            grp = self.pc_groups[pc_key]
            enc = self.pc_encoders[grp]

        if return_attn:
            return enc(points, return_attn=return_attn)

        return enc(points)

    def _fuse_pointcloud_features(self, pc_feats: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse a list of per-pointcloud features [B, C] into a single [B, C'].
        Supported modes: mean, sum, concat.
        """
        if len(pc_feats) == 1:
            return pc_feats[0]

        if self.pc_fusion_mode == "mean":
            return torch.stack(pc_feats, dim=0).mean(dim=0)
        elif self.pc_fusion_mode == "sum":
            return torch.stack(pc_feats, dim=0).sum(dim=0)
        elif self.pc_fusion_mode == "concat":
            return torch.cat(pc_feats, dim=-1)
        else:
            raise ValueError(f"Unknown pc_fusion_mode: {self.pc_fusion_mode}")

    def _encode_low_dim(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode low-dimensional state based on chosen low_dim_mode.
        """
        if self.state_output_dim == 0:
            # No low-dim state
            return observations[self.low_dim_keys[0]].new_zeros(
                (*observations[self.low_dim_keys[0]].shape[:-1], 0)
            )

        if self.low_dim_mode == "concat_encode":
            state_inputs = [observations[k] for k in self.low_dim_keys]
            state = torch.cat(state_inputs, dim=-1)
            state_feat = self.state_mlp(state)
        else:
            feats = [self.state_mlps[k](observations[k]) for k in self.low_dim_keys]
            fused = torch.cat(feats, dim=-1)
            state_feat = self.fusion_mlp(fused)
        return state_feat

    def forward(self, observations: Dict, return_attn: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        # --------- Pointcloud features for each key ---------
        pc_feats = []
        attn_dict = {} if return_attn else None

        for pc_key in self.pc_keys:
            points = observations[pc_key]
            if return_attn:
                feat, attn = self._encode_single_pointcloud(points, pc_key, return_attn=return_attn)
                attn_dict[pc_key] = attn
            else:
                feat = self._encode_single_pointcloud(points, pc_key)

            pc_feats.append(feat)

        pn_feat = self._fuse_pointcloud_features(pc_feats)

        # --------- Low-dim features ---------
        state_feat = self._encode_low_dim(observations)

        # --------- Combine ---------
        final_feat = torch.cat([pn_feat, state_feat], dim=-1)

        if return_attn:
            return final_feat, attn_dict
        return final_feat

    def output_shape(self):
        return self.n_output_channels
