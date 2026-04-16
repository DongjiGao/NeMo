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

"""Configuration for NeMo Speech LM models in vLLM.

Provides ``NeMoSpeechLMConfig``, a HuggingFace-compatible config class
that wraps the LLM backbone's text config with NeMo-specific fields
(perception, audio_locator_tag, etc.).  The checkpoint's ``config.json``
determines which LLM backbone and encoder are used; hybrid (Mamba+MoE)
vs standard transformer backends are auto-detected.
"""

from transformers import AutoConfig, PretrainedConfig

_HYBRID_ARCHITECTURES = frozenset({
    "NemotronHForCausalLM",
    "NemotronHybridForCausalLM",
})


def _is_hybrid_backend(architectures: list[str]) -> bool:
    return bool(set(architectures) & _HYBRID_ARCHITECTURES)


class NeMoSpeechLMConfig(PretrainedConfig):
    """HuggingFace config for NeMo Speech LM multimodal models.

    Wraps a pretrained LLM config (e.g. NemotronH, Qwen3) with
    additional fields for the speech perception module.  Hybrid vs
    standard transformer is auto-detected from ``pretrained_llm``.
    """

    model_type = "nemo_speechlm"

    def __init__(
        self,
        perception: dict | None = None,
        pretrained_llm: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        pretrained_asr: str = "nvidia/canary-1b-v2",
        audio_locator_tag: str = "<|audio|>",
        prompt_format: str = "nemotron-nano-v3",
        pretrained_weights: bool = True,
        lora: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.perception = perception or {}
        self.pretrained_llm = pretrained_llm
        self.pretrained_asr = pretrained_asr
        self.audio_locator_tag = audio_locator_tag
        self.prompt_format = prompt_format
        self.pretrained_weights = pretrained_weights
        self.lora = lora

        self.text_config = AutoConfig.from_pretrained(
            pretrained_llm, trust_remote_code=True
        )

        raw_archs = getattr(self.text_config, "architectures", [])
        self.is_hybrid = _is_hybrid_backend(raw_archs)

        if self.is_hybrid:
            self.text_config.architectures = ["NemotronHForCausalLM"]
            if (
                not hasattr(self.text_config, "total_num_kv_heads")
                or self.text_config.total_num_kv_heads is None
            ):
                self.text_config.total_num_kv_heads = getattr(
                    self.text_config, "num_key_value_heads", 2
                )
            if not hasattr(self.text_config, "rms_norm_eps"):
                self.text_config.rms_norm_eps = getattr(
                    self.text_config, "layer_norm_epsilon", 1e-5
                )

        self.text_config.vocab_size = self.text_config.vocab_size + 10

    @property
    def llm_architectures(self) -> list[str]:
        """Return the LLM backbone architectures list."""
        return getattr(self.text_config, "architectures", [])

    def get_text_config(self, decoder=False) -> PretrainedConfig:
        return self.text_config

    _ATTR_ALIASES = {
        "rms_norm_eps": "layer_norm_epsilon",
        "layer_norm_eps": "layer_norm_epsilon",
    }

    def __getattr__(self, name):
        if name.startswith("_") or name in (
            "perception",
            "pretrained_llm",
            "pretrained_asr",
            "audio_locator_tag",
            "prompt_format",
            "pretrained_weights",
            "text_config",
            "_ATTR_ALIASES",
            "lora",
            "is_hybrid",
        ):
            raise AttributeError(name)
        alias = self._ATTR_ALIASES.get(name, name) if self.is_hybrid else name
        try:
            return getattr(self.text_config, alias)
        except AttributeError:
            if alias != name:
                try:
                    return getattr(self.text_config, name)
                except AttributeError:
                    pass
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute '{name}'"
            )
