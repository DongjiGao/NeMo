# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Single registered model class for the NeMo Speech LM (SALM) vLLM plugin.

Architecture: NeMo speech encoder (e.g. FastConformer) + projection + LLM.

A single ``NeMoSpeechLMForConditionalGeneration`` covers every supported
backbone family. Backbone-specific behavior (architecture choice, weight
rename rules, optional LoRA merge, mamba state passthroughs) lives in
``backends.py`` and is selected once at ``__init__`` time via
``make_backend(config)``. The class declares ``IsHybrid`` /
``SupportsMambaPrefixCaching`` so vLLM's hybrid KV-cache allocator picks up
NemotronH backbones; for transformer backbones the runtime
``ModelConfig.is_hybrid`` property returns False because ``config.py``
populates ``text_config.layer_types`` with all-attention markers (vLLM's
granite-4.0-micro escape hatch).

Requires NeMo toolkit for the audio encoder:
    pip install 'nemo-toolkit[asr]'
"""

from collections.abc import Iterable, Mapping
from typing import Any, ClassVar, Literal

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import (
    IsHybrid,
    MultiModalEmbeddings,
    SupportsMambaPrefixCaching,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from nemo.utils import logging

from nemo.collections.speechlm2.parts.encoder_chunking import encode_audio_with_optional_chunking
from nemo.collections.speechlm2.vllm.salm.audio import (
    _SAMPLING_RATE,
    NeMoSpeechLMAudioInputs,
    NeMoSpeechLMDummyInputsBuilder,
    NeMoSpeechLMMultiModalProcessor,
    NeMoSpeechLMProcessingInfo,
    _apply_encoder_quantization,
    _load_nemo_perception,
    _prepare_prequantized_encoder,
    _maybe_mount_independent_speaker_encoder,
    _maybe_mount_pe_encoder,
)
from nemo.collections.speechlm2.vllm.salm.backends import HybridBackend, make_backend
from nemo.collections.speechlm2.vllm.salm.config import _AUDIO_PLACEHOLDER
from nemo.collections.speechlm2.vllm.salm.ctc_timestamps import (
    WORKER_ALIGN_METHOD,
    active_encoder,
    align_request,
    install_worker_align_method,
    pending_row_ids,
    register_encoder,
    require_v1_model_runner,
    set_default_retention,
    speaker_logprob_weight,
    store_timestamp_inputs,
)

_AUDIO_INPUT_DTYPE = torch.float32
_PERCEPTION_DTYPE = torch.bfloat16

# The prompt the reported ASR-leaderboard numbers were measured with. Serving has
# to render the same one or its WER will not match the published figures: the
# "Omit accidental repetitions" clause is what suppresses runaway decoding, and
# dropping the /no_think system turn lets the model emit reasoning instead of a
# transcript.
_TRANSCRIBE_PROMPT = (
    "Produce a verbatim transcript of the audio. Preserve named entities, "
    "abbreviations, numbers, dates, measurements, acronyms, and technical terms "
    "as clearly as possible. Keep the same language as spoken and do not "
    "translate. Omit accidental repetitions."
)

# The multi-speaker (t-SOT) prompt of the multispeaker-sot benchmark. Speaker
# segments need the <spk:N> turn tags only this prompt produces; the verbatim
# prompt above yields untagged text, which aligns as a single speaker.
_SOT_PROMPT = (
    "Transcribe this audio. Include all spoken words and verbal fillers. Write numbers as spoken. "
    "Use no punctuation and no capitalization. Start every speaker turn with a speaker tag such as "
    "<spk:0>, <spk:1>, <spk:2>, etc.; assign one stable tag per speaker and keep speaker identities "
    "consistent. Output only the speaker-tagged transcript."
)

# runtime_compat enforces locator placement for the chat path: text, one ASCII
# space, then a final <|audio|>. get_generation_prompt bypasses chat rendering,
# so it has to reproduce that layout itself or the two entry points would
# disagree on the prompt for the same model.
_AUDIO_LAST = True

# Nemotron Speech is English-first. Declaring only what we have measured keeps
# validate_language from silently promising languages we have not evaluated;
# other codes still pass with a warning through get_other_languages.
_SUPPORTED_LANGUAGES: Mapping[str, str] = {"en": "english"}

# Our timestamps come from CTC alignment rather than the token stream, and the
# captured rows live in the engine process, so the serving layer has to fetch the
# alignment from there (transcription_worker_method) and hand it to the hooks.
# That plumbing is an upstream addition; against a stock vLLM the hooks are
# either never called or called without it, and the request would fail deep in
# response assembly with a misleading "did not contain a valid diarized
# transcript". Advertising the capability only when the plumbing exists turns
# that into a clean, up-front "not supported for this model" instead.
try:  # pragma: no cover - depends on the installed vLLM
    from vllm.model_executor.models.interfaces import (
        SupportsTranscription as _SupportsTranscriptionProto,
    )

    _TIMESTAMP_PLUMBING = hasattr(_SupportsTranscriptionProto, "transcription_worker_method")
except Exception:  # noqa: BLE001
    _TIMESTAMP_PLUMBING = False


def _is_parallel_expert_encoder(module: nn.Module) -> bool:
    """Recognize the shared speaker-aware encoder contract without importing ASR at plugin import time."""
    return bool(getattr(module, "supports_external_speaker_targets", False)) and callable(
        getattr(module, "online_inference", None)
    )


@MULTIMODAL_REGISTRY.register_processor(
    NeMoSpeechLMMultiModalProcessor,
    info=NeMoSpeechLMProcessingInfo,
    dummy_inputs=NeMoSpeechLMDummyInputsBuilder,
)
class NeMoSpeechLMForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    IsHybrid,
    SupportsMambaPrefixCaching,
    SupportsTranscription,
):
    """Backbone-agnostic NeMo SpeechLM. Composition with a backend handles per-backbone details."""

    supported_languages: ClassVar[Mapping[str, str]] = _SUPPORTED_LANGUAGES
    supports_transcription: ClassVar[Literal[True]] = True

    # Timings come from CTC forced alignment over encoder frames rather than from
    # timestamp tokens, so segment-timestamp parsing stays off: the server would
    # otherwise try to read timestamps out of the decoded token stream, where
    # this model emits none.
    supports_segment_timestamp: ClassVar[bool] = False
    supports_word_timestamp: ClassVar[bool] = _TIMESTAMP_PLUMBING
    supports_diarized_transcription: ClassVar[bool] = _TIMESTAMP_PLUMBING
    transcription_worker_method: ClassVar[str | None] = WORKER_ALIGN_METHOD if _TIMESTAMP_PLUMBING else None
    # The <spk:N> turn tags are special tokens, so default decoding would strip
    # them before alignment.
    keep_special_tokens_for_diarization: ClassVar[bool] = True

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: Any, task_type: Literal["transcribe", "translate"]
    ) -> Any:
        """Describe audio handling for the /v1/audio/transcriptions endpoint.

        Chunking is disabled: a ParallelExpertEncoder runs its own
        context-preserving window over the full audio, and letting the endpoint
        pre-split would both duplicate that and fragment the CTC frame grid the
        timestamps are derived from.
        """
        from vllm.config import SpeechToTextConfig

        return SpeechToTextConfig(
            sample_rate=_SAMPLING_RATE,
            max_audio_clip_s=None,
            min_energy_split_window_size=None,
        )

    @classmethod
    def get_generation_prompt(cls, stt_params: Any) -> Any:
        """Render the transcription prompt for one audio chunk.

        ``diarized_json`` requests get the t-SOT prompt, which is what makes the
        model emit speaker tags; everything else gets the measured verbatim prompt.

        Returns a text prompt rather than token ids on purpose: the multimodal
        processor splits on ``<|audio|>`` and expands each locator into the
        estimated audio-token count, so pre-tokenizing here would skip that
        expansion and the embedding merge would not line up.
        """
        try:
            from vllm.tokenizers import cached_tokenizer_from_config
        except ImportError:  # older vLLM layout
            from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config

        task_type = getattr(stt_params, "task_type", "transcribe")
        if task_type != "transcribe":
            raise ValueError(f"NeMo SpeechLM supports transcription only, got task_type={task_type!r}.")

        diarized = getattr(stt_params, "response_format", None) == "diarized_json"
        text = _SOT_PROMPT if diarized else _TRANSCRIBE_PROMPT
        content = f"{text} {_AUDIO_PLACEHOLDER}" if _AUDIO_LAST else f"{_AUDIO_PLACEHOLDER}\n{text}"

        tokenizer = cached_tokenizer_from_config(stt_params.model_config)
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Older templates do not accept enable_thinking; the /no_think
            # directive in the prompt text is the fallback guard.
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )

        return {"prompt": prompt, "multi_modal_data": {"audio": stt_params.audio}}

    @classmethod
    def get_word_timestamps(cls, text: str, request_output: Any = None, worker_output: Any = None) -> Any:
        """Return per-word timings for a finished transcription.

        Returns ``None`` rather than an empty list when nothing was captured, so
        the response omits ``words`` instead of asserting the audio had none.
        """
        from vllm.entrypoints.speech_to_text.transcription.protocol import (
            TranscriptionWord,
        )

        words = cls._aligned_words(text, request_output, worker_output)
        if not words:
            return None
        return [
            TranscriptionWord(word=word["word"], start=word["start"], end=word["end"])
            for word in words
        ]

    @classmethod
    def parse_diarized_transcript(cls, text: str, request_output: Any = None, worker_output: Any = None) -> Any:
        """Group aligned words into speaker-attributed segments.

        The speaker labels come from the model's own ``<spk:N>`` t-SOT tags, while
        the boundaries come from CTC alignment, so a segment closes whenever the
        speaker changes rather than on punctuation or silence.
        """
        from vllm.model_executor.models.interfaces import DiarizedTranscriptionSegment

        words = cls._aligned_words(text, request_output, worker_output)
        if not words:
            return []

        segments: list[DiarizedTranscriptionSegment] = []
        run: list[dict] = []

        def flush() -> None:
            if not run:
                return
            segments.append(
                DiarizedTranscriptionSegment(
                    start=run[0]["start"],
                    end=run[-1]["end"],
                    speaker=run[0]["speaker"],
                    text=" ".join(word["word"] for word in run).strip(),
                )
            )

        for word in words:
            if run and word["speaker"] != run[0]["speaker"]:
                flush()
                run = []
            run.append(word)
        flush()
        return segments

    @classmethod
    def _aligned_words(cls, text: str, request_output: Any, worker_output: Any = None) -> list[dict]:
        """Return the aligned words for a request.

        Under ``vllm serve`` the engine already aligned them through
        ``transcription_worker_method``; aligning here only works when the
        engine shares this process, as with an in-process ``LLM``.
        """
        if worker_output is not None:
            return list(worker_output["words"])
        if request_output is None or not getattr(request_output, "request_id", None):
            return []
        return align_request(request_output.request_id, text)["words"]

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return _AUDIO_PLACEHOLDER
        return None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.encoder_chunk_size_seconds = getattr(config, "encoder_chunk_size_seconds", None)

        backend = make_backend(config)
        self._backend = backend

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=backend.architectures(),
            )

        with self._mark_tower_model(vllm_config, {"audio"}):
            self.perception = _load_nemo_perception(config.perception)
            pe_encoder_path = getattr(config, "pe_encoder_path", None)
            pe_encoder_config = getattr(config, "pe_encoder_config", None)
            speaker_encoder = getattr(config, "speaker_encoder", None)
            has_pe_encoder = pe_encoder_path not in (
                None,
                "",
                False,
            ) or pe_encoder_config not in (
                None,
                {},
                "",
                False,
            )
            has_speaker_encoder = speaker_encoder not in (None, {}, "", False)
            if has_pe_encoder and has_speaker_encoder:
                raise ValueError("ParallelExpertEncoder and speaker_encoder are mutually exclusive.")
            if has_speaker_encoder:
                _maybe_mount_independent_speaker_encoder(
                    self.perception,
                    speaker_encoder,
                    self.encoder_chunk_size_seconds,
                )
                self._uses_pe_encoder = False
            else:
                _maybe_mount_pe_encoder(
                    self.perception,
                    pe_encoder_path,
                    pe_encoder_config,
                    getattr(config, "pe_encoder_overrides", None),
                )
                self._uses_pe_encoder = _is_parallel_expert_encoder(getattr(self.perception, "encoder", None))

            # After mounting, so the tag that marks the ASR branch already exists.
            # The swap itself is deferred to the first forward, which is what lets
            # this run before weights are loaded.
            _apply_encoder_quantization(self.perception, getattr(config, "encoder_quantization", None))

        # The server calls this for every timestamped request, adapter or not;
        # without one it returns no words and the request fails with a clean 400.
        install_worker_align_method()
        self._maybe_enable_ctc_timestamps(getattr(config, "ctc_timestamps", None), vllm_config)

        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def _maybe_enable_ctc_timestamps(self, ctc_config: Any, vllm_config: VllmConfig) -> None:
        """Arm CTC timestamp capture when the checkpoint ships an adapter path.

        Driven by a ``ctc_timestamps`` block in the checkpoint config, mirroring
        how ``encoder_quantization`` travels, so serving needs no extra flags.
        Off unless configured: capture costs throughput and retains rows, so a
        deployment that does not want timestamps should not pay for them.
        """
        if not ctc_config:
            return
        adapter_path = (
            ctc_config.get("adapter_path") if isinstance(ctc_config, dict) else getattr(ctc_config, "adapter_path", None)
        )
        if not adapter_path:
            return
        if not self._uses_pe_encoder:
            raise ValueError(
                "ctc_timestamps.adapter_path is set, but CTC timestamps need a ParallelExpertEncoder perception encoder."
            )

        from nemo.collections.speechlm2.parts.ctc_timestamp_utils import get_ctc_timestamp_aligner
        from nemo.collections.speechlm2.vllm.salm.ctc_timestamps import install_encoder_cache_binding

        encoder = self.perception.encoder
        if not getattr(encoder, "supports_ctc_timestamp_inputs", False):
            raise ValueError(f"{type(encoder).__name__} cannot produce CTC timestamp inputs.")
        require_v1_model_runner(vllm_config)
        weight = speaker_logprob_weight(ctc_config)
        encoder.ctc_timestamp_model_path = adapter_path
        device = next(encoder.parameters()).device
        # Load now rather than on the first timestamped request, so a bad artifact
        # path fails at startup instead of mid-serve.
        aligner = get_ctc_timestamp_aligner(encoder, adapter_path, device)
        aligner.speaker_logprob_weight = weight

        set_default_retention(vllm_config.scheduler_config.max_num_seqs)
        register_encoder(lambda: self.perception.encoder)
        install_encoder_cache_binding(lambda: self.perception.encoder)
        logging.info("[NeMoSpeechLM] CTC timestamps enabled from checkpoint config: %s", adapter_path)

    # ── audio processing ──

    def _parse_audio_input(
        self,
        audio_signal: torch.Tensor | list[torch.Tensor] | None = None,
        audio_signal_length: torch.Tensor | None = None,
        **kwargs,
    ) -> NeMoSpeechLMAudioInputs | None:
        if audio_signal is None:
            return None

        if not isinstance(audio_signal_length, torch.Tensor):
            raise ValueError(
                "audio_signal_length must be a torch.Tensor; got " f"{type(audio_signal_length).__name__}."
            )

        if isinstance(audio_signal, list):
            max_len = max(a.shape[-1] for a in audio_signal)
            padded = [torch.nn.functional.pad(a, (0, max_len - a.shape[-1])) for a in audio_signal]
            audio_signal = torch.stack(padded, dim=0)

        return NeMoSpeechLMAudioInputs(
            audio_signal=audio_signal,
            audio_signal_length=audio_signal_length,
        )

    def _process_audio(self, audio_input: NeMoSpeechLMAudioInputs) -> tuple[torch.Tensor, ...]:
        # Real device placement happens at init via _mark_tower_model +
        # get_mm_mapping; this .to() is a no-op guard kept for paranoia.
        device = next(self.perception.parameters()).device
        self.perception = self.perception.to(device)

        audio_signal = audio_input.audio_signal
        if isinstance(audio_signal, list):
            audio_signal = torch.stack(audio_signal, dim=0)
        audio_signal = audio_signal.to(device=device, dtype=_AUDIO_INPUT_DTYPE)
        audio_lengths = audio_input.audio_signal_length.to(device=device)

        # Mirrors training (``encode_audio_with_optional_chunking``): when the
        # checkpoint was trained with a chunked encoder (e.g. SALMAutomodel
        # default 30 s), long audios are split into chunks before the perception
        # forward and the per-chunk embeddings are concatenated. ``None``
        # disables chunking and runs a single forward over the full batch.
        # A ParallelExpertEncoder instead runs its own context-preserving online
        # inference over the full audio, so it bypasses the chunking helper.
        with torch.no_grad():
            if self._uses_pe_encoder:
                encoder = self.perception.encoder
                with encoder.online_inference():
                    if active_encoder() is None:
                        audio_embs, audio_emb_lens = self.perception(
                            input_signal=audio_signal, input_signal_length=audio_lengths
                        )
                    else:
                        # vLLM passes only tensors here, so the timestamp inputs are
                        # stored under placeholders that ctc_timestamps renames to
                        # each item's mm_hash once the runner announces it.
                        row_ids = pending_row_ids(audio_signal.shape[0])
                        audio_embs, audio_emb_lens, timestamp_inputs = self.perception(
                            input_signal=audio_signal,
                            input_signal_length=audio_lengths,
                            return_ctc_timestamp_inputs=True,
                        )
                        store_timestamp_inputs(row_ids, timestamp_inputs, audio_lengths.double() / _SAMPLING_RATE)
                audio_embeds = [emb[:emblen] for emb, emblen in zip(audio_embs, audio_emb_lens)]
            else:
                audio_embeds = encode_audio_with_optional_chunking(
                    self.perception,
                    audio_signal,
                    audio_lengths,
                    chunk_size_seconds=self.encoder_chunk_size_seconds,
                    sampling_rate=_SAMPLING_RATE,
                )

        return tuple(emb.to(_PERCEPTION_DTYPE) for emb in audio_embeds)

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
        return self.language_model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="perception.proj",
            tower_model="perception.encoder",
        )

    # ── weight loading ──

    def _load_perception_weights(self, perception_weights: dict[str, torch.Tensor]) -> set[str]:
        self.perception = self.perception.to(_PERCEPTION_DTYPE)
        # Between the cast and the load on purpose. The cast is dtype-blind and
        # would turn fp8 buffers back into bfloat16, while the load needs the fp8
        # parameters to already exist or it reports weight_scale as unexpected.
        _prepare_prequantized_encoder(
            self.perception, getattr(self.config, "encoder_quantization", None)
        )
        incompatible = self.perception.load_state_dict(perception_weights, strict=False)

        from nemo.collections.speechlm2.modules.perception import IndependentDualEncoder

        requires_exact_architecture = (
            isinstance(getattr(self.perception, "encoder", None), IndependentDualEncoder) or self._uses_pe_encoder
        )
        if requires_exact_architecture:
            missing = [name for name in incompatible.missing_keys if not name.endswith("._extra_state")]
            unexpected = [name for name in incompatible.unexpected_keys if not name.endswith("._extra_state")]
            if missing or unexpected:
                raise RuntimeError(
                    "Speech encoder checkpoint does not exactly match its exported architecture: "
                    f"missing={missing}, unexpected={unexpected}."
                )
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
                perception[name[len("perception.") :]] = tensor
            elif name.startswith("llm.mtp."):
                pass  # MTP draft-head weights; loaded by the speculative draft model, not here
            elif name.startswith("mtp."):
                raise ValueError(
                    f"Unsupported bare MTP tensor {name!r}; NeMo SpeechLM exports must store draft weights "
                    f"under the 'llm.mtp.*' namespace."
                )
            else:
                llm.append((name, tensor))
        return perception, llm

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        perception_weights, llm_raw = self._split_perception_llm(weights)
        loaded_perception = self._load_perception_weights(perception_weights)

        preprocessed = self._backend.preprocess_llm_weights(llm_raw)
        hf_weights = self._backend.nemo_to_hf_llm_weights(preprocessed)
        combined = (("language_model." + n, t) for n, t in hf_weights)

        loader = AutoWeightsLoader(self)
        loaded_llm = loader.load_weights(combined)

        return loaded_llm | loaded_perception

    # ── vLLM IsHybrid mamba state classmethods ──
    #
    # Reached only when vLLM's ``ModelConfig.is_hybrid`` returns True at
    # runtime (NemotronH backbones). For transformer backbones the
    # ``text_config.layer_types`` shim in ``config.py`` flips ``is_hybrid``
    # off at runtime so vLLM never calls into these.

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig) -> Any:
        return HybridBackend.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig) -> Any:
        return HybridBackend.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls) -> Any:
        return HybridBackend.get_mamba_state_copy_func()
