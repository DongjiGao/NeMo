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
from pathlib import Path
from typing import Any, Dict, Optional, Union

from huggingface_hub import CONFIG_NAME, PyTorchModelHubMixin
from huggingface_hub.hub_mixin import DataclassInstance
from omegaconf import DictConfig, OmegaConf
from transformers.utils import cached_file

SAFETENSORS_SINGLE_FILE = "model.safetensors"


class HFHubMixin(
    PyTorchModelHubMixin,
    library_name="NeMo",
    repo_url="https://github.com/NVIDIA/NeMo",
    docs_url="https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit",
):
    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[dict] = None,
        resume_download: Optional[bool] = None,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs,
    ):
        """
        Load Pytorch pretrained weights and return the loaded model.
        Wrapper over PyTorchModelHubMixin that auto-handles config in **model_kwargs.

        Supports distributed model-parallel loading via ``device_mesh``:

            >>> from nemo.collections.speechlm2.parts.parallel import setup_distributed
            >>> strategy = setup_distributed(tp_size=2)
            >>> model = SALM.from_pretrained(
            ...     "nvidia/salm-model",
            ...     device_mesh=strategy.device_mesh,
            ...     distributed_config=strategy.distributed_config,
            ...     moe_config=strategy.moe_config,
            ...     moe_mesh=strategy.moe_mesh,
            ... )
        """
        # Pop distributed kwargs before they reach the constructor.
        device_mesh = model_kwargs.pop("device_mesh", None)
        distributed_config = model_kwargs.pop("distributed_config", None)
        moe_config = model_kwargs.pop("moe_config", None)
        moe_mesh = model_kwargs.pop("moe_mesh", None)
        torch_dtype = model_kwargs.pop("torch_dtype", None)

        _cached_file_kwargs = dict(
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            _raise_exceptions_for_gated_repo=False,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )

        resolved_config_file = cached_file(model_id, CONFIG_NAME, **_cached_file_kwargs)
        if resolved_config_file is None:
            raise RuntimeError(f"Missing {CONFIG_NAME} file for {model_id=}")
        model_kwargs['cfg'] = OmegaConf.to_container(OmegaConf.load(resolved_config_file))
        # The setting below tells the model's __init__ not to load the original pretrained weights
        # for individual children modules.
        # To illustrate: if you trained a new model M using a pretrained ASR and a pretrained LLM,
        # this setting skips loading the original pretrained ASR and LLM weights, and loads the
        # final trained model weights directly.
        model_kwargs['cfg']['pretrained_weights'] = False

        if device_mesh is None:
            # Non-distributed: existing flow unchanged
            if torch_dtype is not None:
                model_kwargs['cfg']['torch_dtype'] = (
                    torch_dtype if isinstance(torch_dtype, str) else str(torch_dtype).replace("torch.", "")
                )
            return super()._from_pretrained(
                model_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                map_location=map_location,
                strict=strict,
                **model_kwargs,
            )

        # --- Distributed flow ---
        # Delegate to a module-level function so that ``cls(...)`` is not called
        # from a frame that has ``__class__`` in its closure (which our classmethod
        # has due to the ``super()`` call above).  Lightning's
        # ``save_hyperparameters()`` walks the call stack and mistakes such frames
        # for ``__init__`` frames, causing a ``KeyError: 'self'``.
        return _distributed_from_pretrained(
            cls=cls,
            model_id=model_id,
            model_kwargs=model_kwargs,
            torch_dtype=torch_dtype,
            device_mesh=device_mesh,
            distributed_config=distributed_config,
            moe_config=moe_config,
            moe_mesh=moe_mesh,
            cached_file_kwargs=_cached_file_kwargs,
        )

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        *,
        config: Optional[Union[dict, "DataclassInstance"]] = None,
        repo_id: Optional[str] = None,
        push_to_hub: bool = False,
        model_card_kwargs: Optional[Dict[str, Any]] = None,
        **push_to_hub_kwargs,
    ) -> Optional[str]:
        """
        Save weights in local directory.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the model weights and configuration will be saved.
            config (`dict` or `DataclassInstance`, *optional*):
                Model configuration specified as a key/value dictionary or a dataclass instance.
                If not provided, we will automatically serialize attribute ``model.cfg``.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Huggingface Hub after saving it.
            repo_id (`str`, *optional*):
                ID of your repository on the Hub. Used only if `push_to_hub=True`. Will default to the folder name if
                not provided.
            model_card_kwargs (`Dict[str, Any]`, *optional*):
                Additional arguments passed to the model card template to customize the model card.
            push_to_hub_kwargs:
                Additional key word arguments passed along to the [`~ModelHubMixin.push_to_hub`] method.
        Returns:
            `str` or `None`: url of the commit on the Hub if `push_to_hub=True`, `None` otherwise.
        """
        if config is None:
            config = getattr(self, "cfg")
            if isinstance(config, DictConfig):
                config = OmegaConf.to_container(self.cfg)
        return super().save_pretrained(
            save_directory=save_directory,
            config=config,
            repo_id=repo_id,
            push_to_hub=push_to_hub,
            model_card_kwargs=model_card_kwargs,
            **push_to_hub_kwargs,
        )


def _distributed_from_pretrained(
    cls,
    model_id,
    model_kwargs,
    torch_dtype,
    device_mesh,
    distributed_config,
    moe_config,
    moe_mesh,
    cached_file_kwargs,
):
    """Create a distributed model instance outside of a classmethod frame.

    Lightning's ``save_hyperparameters()`` walks the call stack looking for
    ``__init__`` frames.  Our ``_from_pretrained`` classmethod has ``__class__``
    in its closure (due to a ``super()`` call), which Lightning mistakes for an
    ``__init__`` frame, causing ``KeyError: 'self'``.  By moving the constructor
    call here (a plain module-level function), the problematic frame is avoided.
    """
    model_kwargs['cfg']['init_configure_model'] = False
    if torch_dtype is not None:
        model_kwargs['cfg']['torch_dtype'] = (
            torch_dtype if isinstance(torch_dtype, str) else str(torch_dtype).replace("torch.", "")
        )

    # 1. Create instance (tokenizer only; llm=None, perception=None)
    instance = cls(**model_kwargs)

    # 2. Build parallelized architecture
    instance.configure_model(
        device_mesh=device_mesh,
        distributed_config=distributed_config,
        moe_config=moe_config,
        moe_mesh=moe_mesh,
    )

    # 3. Load weights
    weight_file = cached_file(model_id, SAFETENSORS_SINGLE_FILE, **cached_file_kwargs)
    if weight_file is None:
        raise RuntimeError(f"Missing {SAFETENSORS_SINGLE_FILE} file for {model_id=}")
    _load_state_dict_with_dtensors(instance, str(weight_file))

    return instance


def _load_state_dict_with_dtensors(model, weight_file):
    """Load a full (non-sharded) safetensors file into a model with DTensor parameters.

    Plain ``load_state_dict`` fails when the model has DTensor parameters
    (from FSDP2/TP) because in-place copy from ``torch.Tensor`` to ``DTensor``
    is not supported.

    This function converts each tensor to a DTensor matching the corresponding
    model parameter's mesh and placements via ``distribute_tensor``.  Since
    every rank already holds the full tensor, ``distribute_tensor`` is a cheap
    local slice — no communication needed.

    Two optimizations over the naive approach:

    1. DTensor metadata is read from ``named_parameters()`` / ``named_buffers()``
       instead of ``model.state_dict()``, avoiding FSDP2 state-dict hooks.
    2. Tensors are streamed one-at-a-time via ``safe_open`` instead of loading
       the entire file at once, capping peak CPU memory at ~1 tensor.
    """
    from itertools import chain

    from safetensors import safe_open
    from torch.distributed.tensor import DTensor, distribute_tensor

    # 1. Collect DTensor specs from parameters directly — no FSDP2 hooks.
    dtensor_specs = {}
    for name, param in chain(model.named_parameters(), model.named_buffers()):
        if isinstance(param, DTensor):
            dtensor_specs[name] = (param.device_mesh, param.placements)

    # 2. Stream tensors one at a time via safe_open (memory-mapped).
    #    For each tensor: load full to CPU -> distribute_tensor transfers
    #    only the local shard (1/N) to GPU -> full CPU tensor freed.
    state_dict = {}
    with safe_open(weight_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if key in dtensor_specs:
                mesh, placements = dtensor_specs[key]
                state_dict[key] = distribute_tensor(tensor, mesh, placements)
            else:
                state_dict[key] = tensor

    # 3. Load — types match (DTensor->DTensor), copy succeeds.
    model.load_state_dict(state_dict, strict=False)
