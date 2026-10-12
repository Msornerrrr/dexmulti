"""
Train the object-centric DP3 baseline (vendored iDP3 fork, see ../README.md).

Usage (from baselines/object_centric_dp3/):
    python idp3/train.py task=object_centric_xyzc data_name=grasp_pull_xyzc \
        hydra.run.dir=outputs/grasp_pull_xyzc pc_encoder=multi_stage_pointnet ...

DexMulti release changes vs. the fork's train.py: no os.chdir() to the repo root
(paths are relative to the launch CWD), default --config-name=idp3, explicit
hydra version_base="1.1" (same run-dir chdir behaviour the paper runs had).
"""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import hydra
from omegaconf import OmegaConf

from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

os.environ['WANDB_SILENT'] = "True"

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base="1.1",
    config_path=str(pathlib.Path(__file__).resolve().parent.joinpath(
        'diffusion_policy_3d', 'config')),
    config_name="idp3",
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
