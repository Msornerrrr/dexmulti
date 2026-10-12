from typing import Dict
import torch
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.common.normalize_util import (
    array_to_stats,
    get_range_normalizer_from_stat,
    get_identity_normalizer_from_stat,
    get_pointcloud_normalizer_from_pc,
    get_action_normalizer_from_stat,
)
from diffusion_policy_3d.dataset.base_dataset import BaseDataset


def _resolve_path(path):
    import os
    if os.path.isabs(path):
        return path
    # hydra changes CWD to the run dir; to_absolute_path resolves against the
    # original launch CWD (or the current CWD when hydra is not initialised).
    from hydra.utils import to_absolute_path
    return to_absolute_path(path)


class DexCustomDataset(BaseDataset):
    """
    Dataset for diffusion policy with *world-frame* point cloud and pose.
    Keys expected in replay buffer:
    - point_cloud          : (T, N, 3)
    - eef_pose             : (T, 6)  [x, y, z, r, p, y]
    - eef_pose_target      : (T, 6)
    - hand_joint           : (T, D_hand)
    - hand_joint_target    : (T, D_hand)
    """
    def __init__(self,
                zarr_path,
                shape_meta,
                horizon=1,
                pad_before=0,
                pad_after=0,
                n_obs_steps=0,
                n_action_steps=0,
                n_latency_steps=0,
                seed=42,
                val_ratio=0.0,
                max_train_episodes=None,
                task_name=None):
        super().__init__()
        self.task_name = task_name
        # extract pointcloud observation names
        self.pc_keys = [
            k for k, v in shape_meta["obs"].items()
            if v.get("type", None) == "point_cloud"
        ]

        # Load replay buffer.
        # DexMulti release: the fork used `"../../../" + zarr_path` (relative to the
        # hydra run dir). Here zarr_path is absolute or relative to the launch CWD.
        zarr_path = _resolve_path(zarr_path)
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=[
                *self.pc_keys,
                "eef_pose", "eef_pose_target",
                "hand_joint", "hand_joint_target"
            ]
        )

        # Train/Val split
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed
        )

        # Sequence sampler setup
        self.sequence_length = (n_obs_steps - 1) + n_latency_steps + horizon
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.sequence_length,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask
        )

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.train_mask = train_mask

        self.curr_idx = n_obs_steps - 1
        self.obs_start_idx = 0
        self.obs_end_idx = n_obs_steps
        self.act_start_idx = self.curr_idx + n_latency_steps
        self.act_end_idx = self.act_start_idx + n_action_steps
        self.act_horizon_end_idx = self.act_start_idx + horizon

    # ---------------------- dataset split ----------------------
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.sequence_length,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    # ---------------------- utilities ----------------------
    @staticmethod
    def eef_pose_to_pos_rotmat9(eef_pose: np.ndarray):
        """Convert eef_pose (x, y, z, r, p, y) to pos + flattened 9D rotation matrix."""
        pos = eef_pose[..., :3]
        rotmat = R.from_euler("xyz", eef_pose[..., 3:]).as_matrix().reshape(-1, 9)
        return pos.astype(np.float32), rotmat.astype(np.float32)

    def _sample_to_data(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        pc_map = {
            pc_key: sample[pc_key].astype(np.float32)
            for pc_key in self.pc_keys
        }
        eef_pose = sample["eef_pose"].astype(np.float32)
        eef_target = sample["eef_pose_target"].astype(np.float32)
        hand_joint = sample["hand_joint"].astype(np.float32)
        hand_target = sample["hand_joint_target"].astype(np.float32)

        # --- Convert EEF poses to pos + 9D rotation matrix ---
        eef_pos, eef_rotmat9 = self.eef_pose_to_pos_rotmat9(eef_pose)
        eef_target_pos, eef_target_rotmat9 = self.eef_pose_to_pos_rotmat9(eef_target)

        # --- Build action ---
        action = np.concatenate([eef_target_pos, eef_target_rotmat9, hand_target], axis=-1)
        action = action[self.act_start_idx:self.act_horizon_end_idx]

        data = {
            "obs": {
                "eef_pos": eef_pos,
                "eef_rot9d": eef_rotmat9,
                "hand_joint": hand_joint,
            },
            "action": action,
        }

        # Attach ALL pointclouds
        for k, pc in pc_map.items():
            data["obs"][k] = pc

        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def __len__(self) -> int:
        return len(self.sampler)

    # ---------------------- normalization ----------------------
    def get_normalizer(self):
        """
        Compute field-wise normalizers directly from data (world-frame).
        Rotation is represented as a flattened 9D rotation matrix.
        """
        normalizer = LinearNormalizer()

        # --- EEF pos ---
        stat = array_to_stats(self.replay_buffer["eef_pose"][..., :3].astype(np.float32))
        normalizer["eef_pos"] = get_range_normalizer_from_stat(stat)

        # --- EEF rotmat9 ---
        # Important: be careful! 
        # scipy.spatial.transform.Rotation returns float64 arrays, even with float32 inputs
        eef_rotmat9_all = R.from_euler("xyz", self.replay_buffer["eef_pose"][..., 3:]).as_matrix()
        eef_rotmat9_all = eef_rotmat9_all.reshape(-1, 9).astype(np.float32)
        stat = array_to_stats(eef_rotmat9_all)
        normalizer["eef_rot9d"] = get_identity_normalizer_from_stat(stat)

        # --- Hand joints ---
        stat = array_to_stats(self.replay_buffer["hand_joint"].astype(np.float32))
        normalizer["hand_joint"] = get_range_normalizer_from_stat(stat)

        # --- Point cloud ---
        for pc_key in self.pc_keys:
            normalizer[pc_key] = get_pointcloud_normalizer_from_pc(
                self.replay_buffer[pc_key].astype(np.float32)
            )

        # --- Actions (EEF target + hand target) ---
        eef_target = self.replay_buffer["eef_pose_target"].astype(np.float32)
        eef_target_pos, eef_target_rotmat9 = self.eef_pose_to_pos_rotmat9(eef_target)
        hand_target = self.replay_buffer["hand_joint_target"].astype(np.float32)
        action_all = np.concatenate([eef_target_pos, eef_target_rotmat9, hand_target], axis=-1)
        stat = array_to_stats(action_all)
        normalizer["action"] = get_action_normalizer_from_stat(stat)

        return normalizer
