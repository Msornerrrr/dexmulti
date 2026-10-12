from diffusion_policy_3d.model.common.normalizer import SingleFieldLinearNormalizer
from diffusion_policy_3d.common.pytorch_util import dict_apply, dict_apply_reduce, dict_apply_split
import numpy as np


def get_range_normalizer_from_stat(stat, output_max=1, output_min=-1, range_eps=1e-7):
    # -1, 1 normalization
    input_max = stat['max']
    input_min = stat['min']
    input_range = input_max - input_min
    ignore_dim = input_range < range_eps
    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def get_image_range_normalizer():
    scale = np.array([2], dtype=np.float32)
    offset = np.array([-1], dtype=np.float32)
    stat = {
        'min': np.array([0], dtype=np.float32),
        'max': np.array([1], dtype=np.float32),
        'mean': np.array([0.5], dtype=np.float32),
        'std': np.array([np.sqrt(1/12)], dtype=np.float32)
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def get_identity_normalizer_from_stat(stat):
    scale = np.ones_like(stat['min'])
    offset = np.zeros_like(stat['min'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def robomimic_abs_action_normalizer_from_stat(stat, rotation_transformer):
    result = dict_apply_split(
        stat, lambda x: {
            'pos': x[...,:3],
            'rot': x[...,3:6],
            'gripper': x[...,6:]
    })

    def get_pos_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        # -1, 1 normalization
        input_max = stat['max']
        input_min = stat['min']
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

        return {'scale': scale, 'offset': offset}, stat

    def get_rot_param_info(stat):
        example = rotation_transformer.forward(stat['mean'])
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info
    
    def get_gripper_param_info(stat):
        example = stat['max']
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info

    pos_param, pos_info = get_pos_param_info(result['pos'])
    rot_param, rot_info = get_rot_param_info(result['rot'])
    gripper_param, gripper_info = get_gripper_param_info(result['gripper'])

    param = dict_apply_reduce(
        [pos_param, rot_param, gripper_param], 
        lambda x: np.concatenate(x,axis=-1))
    info = dict_apply_reduce(
        [pos_info, rot_info, gripper_info], 
        lambda x: np.concatenate(x,axis=-1))

    return SingleFieldLinearNormalizer.create_manual(
        scale=param['scale'],
        offset=param['offset'],
        input_stats_dict=info
    )


def robomimic_abs_action_only_normalizer_from_stat(stat):
    result = dict_apply_split(
        stat, lambda x: {
            'pos': x[...,:3],
            'other': x[...,3:]
    })

    def get_pos_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        # -1, 1 normalization
        input_max = stat['max']
        input_min = stat['min']
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

        return {'scale': scale, 'offset': offset}, stat

    
    def get_other_param_info(stat):
        example = stat['max']
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info

    pos_param, pos_info = get_pos_param_info(result['pos'])
    other_param, other_info = get_other_param_info(result['other'])

    param = dict_apply_reduce(
        [pos_param, other_param], 
        lambda x: np.concatenate(x,axis=-1))
    info = dict_apply_reduce(
        [pos_info, other_info], 
        lambda x: np.concatenate(x,axis=-1))

    return SingleFieldLinearNormalizer.create_manual(
        scale=param['scale'],
        offset=param['offset'],
        input_stats_dict=info
    )


def robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat):
    Da = stat['max'].shape[-1]
    Dah = Da // 2
    result = dict_apply_split(
        stat, lambda x: {
            'pos0': x[...,:3],
            'other0': x[...,3:Dah],
            'pos1': x[...,Dah:Dah+3],
            'other1': x[...,Dah+3:]
    })

    def get_pos_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        # -1, 1 normalization
        input_max = stat['max']
        input_min = stat['min']
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

        return {'scale': scale, 'offset': offset}, stat

    
    def get_other_param_info(stat):
        example = stat['max']
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info

    pos0_param, pos0_info = get_pos_param_info(result['pos0'])
    pos1_param, pos1_info = get_pos_param_info(result['pos1'])
    other0_param, other0_info = get_other_param_info(result['other0'])
    other1_param, other1_info = get_other_param_info(result['other1'])

    param = dict_apply_reduce(
        [pos0_param, other0_param, pos1_param, other1_param], 
        lambda x: np.concatenate(x,axis=-1))
    info = dict_apply_reduce(
        [pos0_info, other0_info, pos1_info, other1_info], 
        lambda x: np.concatenate(x,axis=-1))

    return SingleFieldLinearNormalizer.create_manual(
        scale=param['scale'],
        offset=param['offset'],
        input_stats_dict=info
    )


def array_to_stats(arr: np.ndarray):
    stat = {
        'min': np.min(arr, axis=0),
        'max': np.max(arr, axis=0),
        'mean': np.mean(arr, axis=0),
        'std': np.std(arr, axis=0)
    }
    return stat

def get_action_normalizer_from_stat(stat):
    """
    Action normalizer for [Δx, Δy, Δz, rotmat9(9), hand_joint(D)].
    Position and hand are scaled, rotation kept identity-normalized.
    """
    result = {
        'pos': {k: v[..., :3] for k, v in stat.items()},
        'rotmat9': {k: v[..., 3:12] for k, v in stat.items()},
        'hand': {k: v[..., 12:] for k, v in stat.items()}
    }

    pos_norm = get_range_normalizer_from_stat(result['pos'])
    rot_norm = get_identity_normalizer_from_stat(result['rotmat9'])
    hand_norm = get_range_normalizer_from_stat(result['hand'])

    scale = np.concatenate([
        pos_norm.params_dict['scale'].detach().cpu().numpy(),
        rot_norm.params_dict['scale'].detach().cpu().numpy(),
        hand_norm.params_dict['scale'].detach().cpu().numpy()
    ], axis=-1)
    offset = np.concatenate([
        pos_norm.params_dict['offset'].detach().cpu().numpy(),
        rot_norm.params_dict['offset'].detach().cpu().numpy(),
        hand_norm.params_dict['offset'].detach().cpu().numpy()
    ], axis=-1)

    input_stats = {
        k: np.concatenate([result['pos'][k], result['rotmat9'][k], result['hand'][k]], axis=-1)
        for k in ['min', 'max', 'mean', 'std']
    }
    return SingleFieldLinearNormalizer.create_manual(scale, offset, input_stats)

def get_pointcloud_normalizer_from_pc(pc: np.ndarray):
    """
    Compute a normalizer for point clouds with shape (..., N, D).
    Normalizes only the XYZ dimensions globally across all samples and points to [-1, 1].
    If D > 3, the other dimensions (e.g. segmentation mask, intensity) are left unnormalized.
    """
    # flatten everything except feature dim (D)
    flat_pc = pc.reshape(-1, pc.shape[-1])

    if flat_pc.shape[-1] > 3:
        # compute stats across all points and all samples
        pos_stat = array_to_stats(flat_pc[:, :3])  # global stats for x,y,z
        mask_stat = array_to_stats(flat_pc[:, 3:])  # global stats for mask/intensity/extra dims

        # build normalizers for position and others
        pc_pos_norm = get_range_normalizer_from_stat(pos_stat)
        pc_mask_norm = get_identity_normalizer_from_stat(mask_stat)

        # combine parameters
        scale = np.concatenate([
            pc_pos_norm.params_dict["scale"].detach().cpu().numpy(),
            pc_mask_norm.params_dict["scale"].detach().cpu().numpy()
        ], axis=-1)
        offset = np.concatenate([
            pc_pos_norm.params_dict["offset"].detach().cpu().numpy(),
            pc_mask_norm.params_dict["offset"].detach().cpu().numpy()
        ], axis=-1)

        # merge input stats dictionaries
        input_stats = {
            k: np.concatenate([pos_stat[k], mask_stat[k]], axis=-1)
            for k in ["min", "max", "mean", "std"]
        }

        return SingleFieldLinearNormalizer.create_manual(scale, offset, input_stats)

    else:
        # flatten all points and samples for (x,y,z)
        flat_xyz = flat_pc.reshape(-1, 3)
        stat = array_to_stats(flat_xyz)
        return get_range_normalizer_from_stat(stat)
