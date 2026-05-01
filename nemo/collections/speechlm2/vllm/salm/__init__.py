# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""vLLM plugin registration for NeMo Speech LM (SALM) models.

Registers ``NeMoSpeechLMConfig`` and the single
``NeMoSpeechLMForConditionalGeneration`` model class with vLLM via the
``vllm.general_plugins`` entry point.

A single model class covers every supported backbone family (standard
transformer like Qwen3 / Parakeet-TDT, hybrid Mamba+MoE like NemotronH).
Backbone-specific behavior is selected at instantiation time by
``backends.make_backend(config)``; vLLM's runtime ``ModelConfig.is_hybrid``
property gates the hybrid KV-cache path on ``text_config.layer_types``,
which ``config.py`` populates appropriately for transformer backbones.
"""

_PKG = "nemo.collections.speechlm2.vllm.salm"


def register():
    """Register the NeMo Speech LM model and config with vLLM."""
    from transformers import AutoConfig

    from nemo.collections.speechlm2.vllm.salm.config import NeMoSpeechLMConfig

    AutoConfig.register("nemo_speechlm", NeMoSpeechLMConfig)

    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    _CONFIG_REGISTRY["nemo_speechlm"] = NeMoSpeechLMConfig

    from vllm.model_executor.models.registry import ModelRegistry

    ModelRegistry.register_model(
        "NeMoSpeechLMForConditionalGeneration",
        f"{_PKG}.model:NeMoSpeechLMForConditionalGeneration",
    )

    _apply_backend_patches()


def _apply_backend_patches():
    """Apply runtime patches needed for vLLM compatibility.

    Called at plugin registration time. All patches here must be
    pickle-safe (no closures) since vLLM spawns EngineCore as a
    subprocess.
    """
    try:
        from transformers import AutoConfig as _AC

        _nhc = _AC.from_pretrained(
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            trust_remote_code=True,
        )
        NHConfigCls = type(_nhc)
        _orig_getattr = getattr(NHConfigCls, "__getattr__", None)

        def _patched_getattr(self, name):
            if name == "rms_norm_eps":
                return getattr(self, "layer_norm_epsilon", 1e-5)
            if _orig_getattr:
                return _orig_getattr(self, name)
            raise AttributeError(name)

        NHConfigCls.__getattr__ = _patched_getattr
    except Exception:
        # Best-effort: the patch only matters for NemotronH backbones. If the
        # NemotronH config class can't be reached (network offline, model not
        # cached, transformers signature changed), other backbones still load
        # fine -- silently skip and let plugin registration succeed.
        pass
