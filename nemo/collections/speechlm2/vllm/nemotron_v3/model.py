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

"""Inference-only NeMo Speech LM model for vLLM.

Architecture: NeMo speech encoder (e.g. FastConformer) + projection + LLM.
Two concrete classes share a common base:

  NeMoSpeechLMForConditionalGeneration
      Hybrid backends (Mamba+MoE, e.g. NemotronH).

  NeMoSpeechLMStdForConditionalGeneration
      Standard transformer backends (e.g. Qwen3, Parakeet-TDT).
      Includes inline LoRA merging for PEFT checkpoints.

Requires NeMo toolkit for the audio encoder:
    pip install nemo_toolkit[asr]
"""

import re
from collections.abc import Iterable, Mapping
from typing import Annotated, Literal

import torch
from torch import nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    IsHybrid,
    MultiModalEmbeddings,
    SupportsMambaPrefixCaching,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.processing.dummy_inputs import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

logger = init_logger(__name__)

_AUDIO_PLACEHOLDER = "<|audio|>"
_SAMPLING_RATE = 16000
_MAX_AUDIO_DURATION_S = 40.0


# ── Helpers ─────────────────────────────────────────────────────────


def _ensure_special_tokens(tokenizer):
    special = [_AUDIO_PLACEHOLDER]
    existing = set(tokenizer.get_vocab().keys())
    to_add = [t for t in special if t not in existing]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})


def _load_nemo_perception(perception_cfg: dict, output_dim: int) -> nn.Module:
    try:
        from nemo.collections.speechlm2.modules import AudioPerceptionModule
        from omegaconf import DictConfig
    except ImportError as e:
        raise ImportError(
            "NeMo is required for the audio encoder. "
            "Install with: pip install nemo_toolkit[asr]"
        ) from e

    cfg = DictConfig(perception_cfg)
    if "output_dim" not in cfg:
        cfg.output_dim = output_dim
    perception = AudioPerceptionModule(cfg)
    perception.eval()
    return perception


def _pad_vocab_tensor(tensor: torch.Tensor, target_vocab: int) -> torch.Tensor:
    if tensor.shape[0] < target_vocab:
        pad = torch.zeros(
            target_vocab - tensor.shape[0],
            *tensor.shape[1:],
            dtype=tensor.dtype,
        )
        tensor = torch.cat([tensor, pad], dim=0)
    return tensor


# ── Multimodal plumbing (shared by both model classes) ──────────────


class NeMoSpeechLMAudioInputs(TensorSchema):
    type: Literal["audio_features"] = "audio_features"
    audio_signal: Annotated[
        torch.Tensor | list[torch.Tensor], TensorShape("b", "t")
    ]
    audio_signal_length: Annotated[torch.Tensor, TensorShape("b")]


class NeMoSpeechLMProcessingInfo(BaseProcessingInfo):

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(
            target_sr=_SAMPLING_RATE,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_max_audio_tokens(self) -> int:
        return self._estimate_audio_tokens(self.get_max_audio_len())

    def get_max_audio_len(self) -> int:
        return int(_MAX_AUDIO_DURATION_S * _SAMPLING_RATE)

    @staticmethod
    def _estimate_audio_tokens(audio_length_samples: int) -> int:
        n_fft = 512
        hop_length = 160
        stft_pad = n_fft // 2
        fbank_len = (
            (audio_length_samples + 2 * stft_pad - n_fft) // hop_length
        )
        kernel, stride, repeat = 3, 2, 3
        add_pad = 1 + 1 - kernel
        length = float(fbank_len)
        for _ in range(repeat):
            length = (length + add_pad) / stride + 1.0
        return max(1, int(length))


class NeMoSpeechLMMultiModalProcessor(
    BaseMultiModalProcessor[NeMoSpeechLMProcessingInfo],
):

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            audio_signal=MultiModalFieldConfig.batched("audio"),
            audio_signal_length=MultiModalFieldConfig.batched("audio"),
        )

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> list[PromptUpdate]:
        def get_replacement(item_idx: int):
            audios = mm_items.get_items("audio", AudioProcessorItems)
            audio = audios.get(item_idx)
            n_tokens = self.info._estimate_audio_tokens(audio.shape[-1])
            repl_full = _AUDIO_PLACEHOLDER * n_tokens
            return PromptUpdateDetails.select_text(
                repl_full, _AUDIO_PLACEHOLDER
            )

        return [
            PromptReplacement(
                modality="audio",
                target=_AUDIO_PLACEHOLDER,
                replacement=get_replacement,
            )
        ]

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        _ensure_special_tokens(tokenizer)
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", [])

        if audios:
            audio_list: list[torch.Tensor] = []
            audio_lengths: list[int] = []
            parts = re.split(
                f"({re.escape(_AUDIO_PLACEHOLDER)})", prompt
            )
            audio_idx = 0
            for i, part in enumerate(parts):
                if part == _AUDIO_PLACEHOLDER and audio_idx < len(audios):
                    audio = audios[audio_idx]
                    audio_tensor = (
                        audio
                        if isinstance(audio, torch.Tensor)
                        else torch.as_tensor(audio, dtype=torch.float32)
                    )
                    if audio_tensor.dim() > 1:
                        audio_tensor = audio_tensor.squeeze()
                    n_tokens = self.info._estimate_audio_tokens(
                        audio_tensor.shape[-1]
                    )
                    parts[i] = _AUDIO_PLACEHOLDER * n_tokens
                    audio_list.append(audio_tensor)
                    audio_lengths.append(audio_tensor.shape[-1])
                    audio_idx += 1

            prompt = "".join(parts)

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        result = BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        if audios:
            result["audio_signal"] = audio_list
            result["audio_signal_length"] = torch.tensor(audio_lengths)
        return result


class NeMoSpeechLMDummyInputsBuilder(
    BaseDummyInputsBuilder[NeMoSpeechLMProcessingInfo],
):

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        return {
            "audio": self._get_dummy_audios(
                length=self.info.get_max_audio_len(),
                num_audios=num_audios,
            )
        }

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        return "Transcribe the following: " + _AUDIO_PLACEHOLDER * num_audios


# ── Shared base class ───────────────────────────────────────────────


class _NeMoSpeechLMBase(nn.Module):
    """Common perception loading, audio processing, forward, and logits."""

    def _init_perception(
        self, config, vllm_config: VllmConfig
    ):
        llm_hidden = config.text_config.hidden_size
        with self._mark_tower_model(vllm_config, {"audio"}):
            self.perception = _load_nemo_perception(
                config.perception, output_dim=llm_hidden
            )
            self.perception = self.perception.to(torch.float32)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return _AUDIO_PLACEHOLDER
        return None

    # ── audio processing ──

    def _parse_audio_input(
        self, **kwargs
    ) -> NeMoSpeechLMAudioInputs | None:
        audio_signal = kwargs.pop("audio_signal", None)
        if audio_signal is None:
            return None
        audio_signal_length = kwargs.pop("audio_signal_length", None)

        if isinstance(audio_signal, list):
            max_len = max(a.shape[-1] for a in audio_signal)
            padded = [
                torch.nn.functional.pad(a, (0, max_len - a.shape[-1]))
                for a in audio_signal
            ]
            audio_signal = torch.stack(padded, dim=0)

        if audio_signal_length is None:
            audio_signal_length = torch.tensor(
                [audio_signal.shape[-1]] * audio_signal.shape[0]
            )
        elif not isinstance(audio_signal_length, torch.Tensor):
            audio_signal_length = torch.tensor(audio_signal_length)

        return NeMoSpeechLMAudioInputs(
            audio_signal=audio_signal,
            audio_signal_length=audio_signal_length,
        )

    def _process_audio(
        self, audio_input: NeMoSpeechLMAudioInputs
    ) -> tuple[torch.Tensor, ...]:
        device = next(self.perception.parameters()).device
        self.perception = self.perception.to(device)

        audio_signal = audio_input.audio_signal
        if isinstance(audio_signal, list):
            audio_signal = torch.stack(audio_signal, dim=0)
        audio_signal = audio_signal.to(device=device, dtype=torch.float32)
        audio_lengths = audio_input.audio_signal_length.to(device=device)

        with torch.no_grad():
            audio_embeds, audio_embed_lens = self.perception(
                input_signal=audio_signal,
                input_signal_length=audio_lengths,
            )

        audio_embeds = audio_embeds.to(torch.bfloat16)

        return tuple(
            audio_embeds[i, : audio_embed_lens[i]]
            for i in range(audio_embeds.shape[0])
        )

    def embed_multimodal(self, **kwargs) -> MultiModalEmbeddings:
        audio_input = self._parse_audio_input(**kwargs)
        if audio_input is None:
            return []
        return self._process_audio(audio_input)

    # ── forward / logits ──

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="perception.proj",
            tower_model="perception.encoder",
        )

    # ── weight loading helpers ──

    def _load_perception_weights(
        self, perception_weights: dict[str, torch.Tensor]
    ) -> set[str]:
        float32_weights = {
            k: v.float() for k, v in perception_weights.items()
        }
        self.perception.load_state_dict(float32_weights, strict=False)
        self.perception = self.perception.to(torch.float32)
        return {"perception." + k for k in perception_weights}

    @staticmethod
    def _split_perception_llm(
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> tuple[dict[str, torch.Tensor], list[tuple[str, torch.Tensor]]]:
        perception: dict[str, torch.Tensor] = {}
        llm: list[tuple[str, torch.Tensor]] = []
        for name, tensor in weights:
            if "._extra_state" in name:
                continue
            if name.startswith("perception."):
                perception[name[len("perception."):]] = tensor
            else:
                llm.append((name, tensor))
        return perception, llm


# ── Hybrid (NemotronH / Mamba+MoE) variant ─────────────────────────


@MULTIMODAL_REGISTRY.register_processor(
    NeMoSpeechLMMultiModalProcessor,
    info=NeMoSpeechLMProcessingInfo,
    dummy_inputs=NeMoSpeechLMDummyInputsBuilder,
)
class NeMoSpeechLMForConditionalGeneration(
    _NeMoSpeechLMBase,
    SupportsMultiModal,
    SupportsPP,
    IsHybrid,
    SupportsMambaPrefixCaching,
):
    """NeMo Speech LM with hybrid Mamba+MoE backbone (e.g. NemotronH)."""

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        from vllm.model_executor.models.nemotron_h import (
            NemotronHForCausalLM,
        )
        return NemotronHForCausalLM.get_mamba_state_dtype_from_config(
            vllm_config
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        from vllm.model_executor.models.nemotron_h import (
            NemotronHForCausalLM,
        )
        return NemotronHForCausalLM.get_mamba_state_shape_from_config(
            vllm_config
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        from vllm.model_executor.models.nemotron_h import (
            NemotronHForCausalLM,
        )
        return NemotronHForCausalLM.get_mamba_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["NemotronHForCausalLM"],
            )

        self._init_perception(config, vllm_config)
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _nemo_to_hf_llm_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """NeMo -> HuggingFace NemotronH weight name mapping."""
        target_vocab = getattr(
            self.config.text_config, "vocab_size", None
        )
        for name, tensor in weights:
            hf_name = name.replace("llm.model.", "backbone.")
            hf_name = hf_name.replace("llm.lm_head", "lm_head")
            if hf_name == "backbone.norm.weight":
                hf_name = "backbone.norm_f.weight"

            if hf_name.endswith(".experts.down_projs"):
                prefix = hf_name.replace(".experts.down_projs", "")
                for i in range(tensor.shape[0]):
                    yield (
                        f"{prefix}.experts.{i}.down_proj.weight",
                        tensor[i].t(),
                    )
            elif hf_name.endswith(".experts.gate_and_up_projs"):
                prefix = hf_name.replace(
                    ".experts.gate_and_up_projs", ""
                )
                for i in range(tensor.shape[0]):
                    yield (
                        f"{prefix}.experts.{i}.up_proj.weight",
                        tensor[i].t(),
                    )
            elif hf_name in (
                "backbone.embed_tokens.weight",
                "lm_head.weight",
            ):
                if target_vocab:
                    tensor = _pad_vocab_tensor(tensor, target_vocab)
                yield (hf_name, tensor)
            else:
                yield (hf_name, tensor)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        perception_weights, llm_raw = self._split_perception_llm(weights)
        loaded_perception = self._load_perception_weights(perception_weights)

        hf_weights = self._nemo_to_hf_llm_weights(llm_raw)
        combined = (
            ("language_model." + n, t) for n, t in hf_weights
        )

        loader = AutoWeightsLoader(self)
        loaded_llm = loader.load_weights(combined)

        return loaded_llm | loaded_perception


# ── Standard transformer variant ────────────────────────────────────


@MULTIMODAL_REGISTRY.register_processor(
    NeMoSpeechLMMultiModalProcessor,
    info=NeMoSpeechLMProcessingInfo,
    dummy_inputs=NeMoSpeechLMDummyInputsBuilder,
)
class NeMoSpeechLMStdForConditionalGeneration(
    _NeMoSpeechLMBase,
    SupportsMultiModal,
    SupportsPP,
):
    """NeMo Speech LM with standard transformer backbone (e.g. Qwen3).

    Handles both PEFT-wrapped checkpoints (``llm.base_model.model.model.``)
    and plain NeMo checkpoints (``llm.model.``).  LoRA weights are merged
    inline in float32 during loading.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        llm_archs = config.llm_architectures

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=llm_archs,
            )

        self._init_perception(config, vllm_config)
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    # ── LoRA merge ──

    @staticmethod
    def _normalize_lora_name(name: str) -> tuple[str, str]:
        """Classify a weight name and return (normalized_name, kind).

        Returns kind: "lora_a", "lora_b", or "base".
        Normalizes both PEFT and NeMo LoRA naming to a common base key.
          PEFT:  q_proj.lora_A.default.weight → (q_proj.weight, lora_a)
          NeMo:  q_proj.lora_A.weight         → (q_proj.weight, lora_a)
          PEFT:  q_proj.base_layer.weight      → (q_proj.weight, base)
          Plain: q_proj.weight                 → (q_proj.weight, base)
        """
        import re
        m = re.match(r"(.+)\.lora_(A|B)(?:\.default)?\.weight$", name)
        if m:
            return m.group(1) + ".weight", "lora_" + m.group(2).lower()
        if ".base_layer." in name:
            return name.replace(".base_layer.", "."), "base"
        return name, "base"

    @staticmethod
    def _merge_lora_weights(
        weights: list[tuple[str, torch.Tensor]],
        lora_cfg: dict | None,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Merge LoRA A/B into base weights in float32.

        Handles both PEFT and NeMo LoRA naming conventions.
        """
        has_lora = any(
            ".lora_A." in n or ".lora_B." in n for n, _ in weights
        )
        if not has_lora:
            yield from weights
            return

        if not lora_cfg:
            lora_cfg = {"r": 128, "lora_alpha": 256}
        scaling = lora_cfg.get("lora_alpha", 1) / lora_cfg.get("r", 1)

        normalize = NeMoSpeechLMStdForConditionalGeneration._normalize_lora_name
        base: dict[str, torch.Tensor] = {}
        lora_a: dict[str, torch.Tensor] = {}
        lora_b: dict[str, torch.Tensor] = {}

        for name, tensor in weights:
            key, kind = normalize(name)
            if kind == "lora_a":
                lora_a[key] = tensor
            elif kind == "lora_b":
                lora_b[key] = tensor
            else:
                base[key] = tensor

        merged_count = 0
        for key, tensor in base.items():
            if key in lora_a and key in lora_b:
                orig_dtype = tensor.dtype
                a = lora_a.pop(key).float()
                b = lora_b.pop(key).float()
                tensor = (tensor.float() + scaling * (b @ a)).to(orig_dtype)
                merged_count += 1
            yield (key, tensor)

        logger.info(
            "Merged %d LoRA weight pairs (scaling=%.4f)",
            merged_count,
            scaling,
        )
        if lora_a:
            logger.warning("Unmerged LoRA keys: %s", list(lora_a.keys()))

    # ── weight name mapping ──

    def _nemo_to_hf_llm_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Map NeMo/PEFT weight names to HuggingFace standard format.

        Handles:
          - PEFT-wrapped: ``llm.base_model.model.model.`` -> ``model.``
          - Plain NeMo:   ``llm.model.`` -> ``model.``
          - Standalone:   ``embed_tokens.weight`` -> ``model.embed_tokens.weight``
          - LM head:      ``llm.lm_head.weight`` -> ``lm_head.weight``
          - Weight tying:  if no embed_tokens, duplicate lm_head as embed_tokens
        """
        target_vocab = getattr(
            self.config.text_config, "vocab_size", None
        )

        has_embed_tokens = False
        lm_head_tensor = None

        for name, tensor in weights:
            if name == "embed_tokens.weight":
                if target_vocab:
                    tensor = _pad_vocab_tensor(tensor, target_vocab)
                yield ("model.embed_tokens.weight", tensor)
                has_embed_tokens = True
                continue

            hf = name
            if hf.startswith("llm.base_model.model."):
                hf = hf[len("llm.base_model.model."):]
            elif hf.startswith("llm."):
                hf = hf[len("llm."):]

            if hf.startswith("lm_head") or hf.startswith("model.embed_tokens"):
                if target_vocab:
                    tensor = _pad_vocab_tensor(tensor, target_vocab)
                if hf.startswith("model.embed_tokens"):
                    has_embed_tokens = True

            if hf == "lm_head.weight":
                lm_head_tensor = tensor

            yield (hf, tensor)

        if not has_embed_tokens and lm_head_tensor is not None:
            logger.info("No embed_tokens found; duplicating lm_head (weight tying)")
            if target_vocab:
                lm_head_tensor = _pad_vocab_tensor(lm_head_tensor, target_vocab)
            yield ("model.embed_tokens.weight", lm_head_tensor)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        perception_weights, llm_raw = self._split_perception_llm(weights)
        loaded_perception = self._load_perception_weights(perception_weights)

        lora_cfg = getattr(self.config, "lora", None)
        merged = list(self._merge_lora_weights(llm_raw, lora_cfg))
        hf_weights = self._nemo_to_hf_llm_weights(merged)
        combined = (
            ("language_model." + n, t) for n, t in hf_weights
        )

        loader = AutoWeightsLoader(self)
        loaded_llm = loader.load_weights(combined)

        return loaded_llm | loaded_perception
