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
        import safetensors.torch

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

        # 3. Load weights — DTensor-aware load_state_dict distributes tensors
        weight_file = cached_file(model_id, SAFETENSORS_SINGLE_FILE, **_cached_file_kwargs)
        if weight_file is None:
            raise RuntimeError(f"Missing {SAFETENSORS_SINGLE_FILE} file for {model_id=}")
        state_dict = safetensors.torch.load_file(str(weight_file))
        instance.load_state_dict(state_dict, strict=False)

        return instance

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
