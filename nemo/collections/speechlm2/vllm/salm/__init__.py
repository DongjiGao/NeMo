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
decoder-only LLMs like Qwen3, hybrid Mamba+MoE like NemotronH).
Backbone-specific behavior is selected at instantiation time.

``register()`` additionally installs the StreamingSTT session-retention
scheduler patch (see ``streaming_scheduler``); because ``register()`` is the
``vllm.general_plugins`` entry point it runs in every vLLM process, including
the EngineCore process that owns the scheduler.
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

    # Install the StreamingSTT session-retention scheduler patch. Runs in every
    # vLLM process (including EngineCore) because ``register()`` is the
    # ``vllm.general_plugins`` entry point. Idempotent, version-guarded, and a
    # no-op for requests that do not opt in via ``extra_args``.
    from nemo.collections.speechlm2.vllm.salm.streaming_scheduler import install_streaming_session_patch

    install_streaming_session_patch()
