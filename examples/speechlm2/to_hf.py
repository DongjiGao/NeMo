# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file

from nemo.core.config import hydra_runner
from nemo.utils.model_utils import import_class_by_path


@dataclass
class HfExportConfig:
    # Name of the model class to be imported, e.g. nemo.collections.speechlm2.models.SALM
    class_path: str

    # Path to PyTorch Lightning checkpoint file (normal ckpt) or directory (distributed ckpt)
    ckpt_path: str

    # Path to the experiment's config, used to instantiate the model class.
    ckpt_config: str

    # Path where we should save the HuggingFace Hub compatible checkpoint
    output_dir: str

    # Dtype used for stored parameters
    dtype: str = "bfloat16"


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str):
    if Path(checkpoint_path).is_dir():
        from torch.distributed.checkpoint import load

        state_dict = {"state_dict": model.state_dict()}
        load(state_dict, checkpoint_id=checkpoint_path)
        model.load_state_dict(state_dict["state_dict"])
    else:
        ckpt_data = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt_data["state_dict"])


def setup_distributed_with_strategy(strategy_cfg: dict):
    """Initialize torch.distributed and create a device mesh by reusing AutomodelParallelStrategy.

    Instantiates the strategy from the trainer config, initializes the process
    group, and calls :meth:`strategy.create_device_mesh` to build the mesh
    with the same parallelism sizes that were used during training.

    Returns:
        Tuple of ``(device_mesh, moe_mesh)``.
    """
    import hydra
    import torch.distributed as dist

    from nemo.utils.trainer_utils import _resolve_automodel_configs

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    strategy = hydra.utils.instantiate(strategy_cfg)
    _resolve_automodel_configs(strategy)
    return strategy.create_device_mesh()


def consolidate_state_dict(model: torch.nn.Module):
    """Gather a full (non-sharded) state dict from a model with DTensor parameters."""
    from torch.distributed.tensor import DTensor

    consolidated = {}
    for key, value in model.state_dict().items():
        if isinstance(value, DTensor):
            consolidated[key] = value.full_tensor().cpu()
        else:
            consolidated[key] = value.cpu()
    return consolidated


def _uses_automodel_parallel(strategy_cfg: dict) -> bool:
    """Check if the strategy config targets AutomodelParallelStrategy."""
    target = strategy_cfg.get("_target_", "")
    return "AutomodelParallelStrategy" in target


@hydra_runner(config_name="HfExportConfig", schema=HfExportConfig)
def main(cfg: HfExportConfig):
    """
    Read PyTorch Lightning checkpoint and export the model to HuggingFace Hub format.
    The resulting model can be then initialized via ModelClass.from_pretrained(path).

    Also supports distributed checkpoints for models trained with FSDP2/TP
    via AutomodelParallelStrategy.  Parallelism sizes (tp_size, pp_size, etc.)
    are read automatically from the ``trainer.strategy`` section of the
    experiment config (``ckpt_config``).

    When the checkpoint is a distributed checkpoint (a directory), launch this
    script via ``torchrun`` with the same number of GPUs used for training.

    Examples:
        # Single-file checkpoint (no parallelism needed):
        python to_hf.py \\
            class_path=nemo.collections.speechlm2.models.SALM \\
            ckpt_path=/path/to/checkpoint.ckpt \\
            ckpt_config=/path/to/config.yaml \\
            output_dir=/path/to/hf_output

        # Distributed checkpoint (parallelism read from config automatically):
        torchrun --nproc-per-node=8 to_hf.py \\
            class_path=nemo.collections.speechlm2.models.SALM \\
            ckpt_path=/path/to/distributed_ckpt_dir \\
            ckpt_config=/path/to/config.yaml \\
            output_dir=/path/to/hf_output
    """
    full_cfg = OmegaConf.to_container(OmegaConf.load(cfg.ckpt_config), resolve=True)
    model_cfg = full_cfg["model"]
    model_cfg["torch_dtype"] = cfg.dtype
    cls = import_class_by_path(cfg.class_path)

    strategy_cfg = full_cfg.get("trainer", {}).get("strategy", {})
    is_distributed = Path(cfg.ckpt_path).is_dir() and _uses_automodel_parallel(strategy_cfg)

    if is_distributed:
        import torch.distributed as dist

        device_mesh, moe_mesh = setup_distributed_with_strategy(strategy_cfg)

        # Don't call configure_model() inside __init__ — we set device_mesh first.
        model_cfg["init_configure_model"] = False
        model = cls(model_cfg)
        model._device_mesh = device_mesh
        model._moe_mesh = moe_mesh
        model.configure_model()

        load_checkpoint(model, cfg.ckpt_path)

        # Consolidate DTensors to regular tensors and save on rank 0.
        consolidated = consolidate_state_dict(model)
        if dist.get_rank() == 0:
            save_hf_checkpoint(model, consolidated, cfg)

        dist.barrier()
        dist.destroy_process_group()
    else:
        model_cfg["init_configure_model"] = True
        model = cls(model_cfg)
        load_checkpoint(model, cfg.ckpt_path)
        model = model.to(getattr(torch, cfg.dtype))
        model.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()
