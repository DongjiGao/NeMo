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

"""Unit tests for the vLLM NeMo Speech LM (SALM) plugin.

Covers plugin registration, config loading + escape-hatch wiring, special
token handling, backend selection, and StreamingSTT session decoding
(including SOU/EOU turn boundaries) -- without requiring GPU or model
weights.
"""

import asyncio
import importlib.util
from types import SimpleNamespace

import pytest

try:
    from nemo.collections.speechlm2.vllm.salm import config as _config_module

    NeMoSpeechLMConfig = _config_module.NeMoSpeechLMConfig

    _HAS_CONFIG = True
except (ImportError, RuntimeError):
    _HAS_CONFIG = False

_HAS_VLLM = importlib.util.find_spec("vllm") is not None
_DEFAULT_CONFIG_KWARGS = {
    "pretrained_llm": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "pretrained_asr": "nvidia/canary-1b-v2",
    "audio_locator_tag": "<|audio|>",
    "prompt_format": "nemotron-nano-v3",
    "pretrained_weights": True,
}


@pytest.mark.skipif(not _HAS_CONFIG, reason="NeMoSpeechLMConfig not available")
class TestNeMoSpeechLMConfig:
    """Tests for NeMoSpeechLMConfig."""

    @pytest.fixture(autouse=True)
    def mock_backbone_config(self, monkeypatch):
        def from_pretrained(model_name: str, trust_remote_code: bool = True):
            if "Nemotron" in model_name:
                return SimpleNamespace(
                    architectures=["NemotronHybridForCausalLM"],
                    hidden_size=2048,
                    vocab_size=131072,
                    num_hidden_layers=4,
                    num_key_value_heads=2,
                    layer_norm_epsilon=1e-5,
                )
            return SimpleNamespace(
                architectures=["Qwen3ForCausalLM"],
                hidden_size=2048,
                vocab_size=151936,
                num_hidden_layers=4,
                rms_norm_eps=1e-6,
            )

        monkeypatch.setattr(_config_module.AutoConfig, "from_pretrained", from_pretrained)

    def test_model_type(self):
        assert NeMoSpeechLMConfig.model_type == "nemo_speechlm"

    def test_default_construction_for_hf_serialization(self):
        """HF internally constructs a no-arg config when serializing configs."""
        cfg = NeMoSpeechLMConfig()
        assert cfg.pretrained_llm is None
        assert cfg.pretrained_asr is None
        assert cfg.audio_locator_tag is None
        assert cfg.prompt_format is None
        assert cfg.pretrained_weights is None
        assert cfg.llm_architectures == []
        assert cfg.get_text_config() is cfg.text_config

    def test_streaming_markers_default_none(self):
        """streaming_markers is optional and defaults to None for non-streaming checkpoints."""
        assert NeMoSpeechLMConfig().streaming_markers is None
        assert NeMoSpeechLMConfig(**_DEFAULT_CONFIG_KWARGS).streaming_markers is None

    def test_streaming_markers_round_trip(self):
        """A streaming checkpoint may carry its turn-template markers in the config."""
        markers = {"chunk_size": 14, "blank_token_id": 151669, "eos_id": 151645}
        cfg = NeMoSpeechLMConfig(**{**_DEFAULT_CONFIG_KWARGS, "streaming_markers": markers})
        assert cfg.streaming_markers == markers

    def test_loads_text_config(self):
        """Config should load a text_config from the pretrained LLM."""
        cfg = NeMoSpeechLMConfig(**_DEFAULT_CONFIG_KWARGS)
        assert cfg.text_config is not None
        assert hasattr(cfg.text_config, "hidden_size")
        assert cfg.get_text_config() is cfg.text_config

    def test_hybrid_backbone_aliases_for_vllm(self):
        cfg = NeMoSpeechLMConfig(**_DEFAULT_CONFIG_KWARGS)
        assert cfg.is_hybrid is True
        assert cfg.llm_architectures == ["NemotronHForCausalLM"]
        assert cfg.text_config.total_num_kv_heads == cfg.text_config.num_key_value_heads
        assert cfg.text_config.rms_norm_eps == cfg.text_config.layer_norm_epsilon

    @pytest.mark.parametrize(
        "architectures, expected_is_hybrid",
        [
            (["NemotronHForCausalLM"], True),
            (["NemotronHybridForCausalLM"], True),
            (["Qwen3ForCausalLM"], False),
            (["LlamaForCausalLM"], False),
            (["Qwen2ForCausalLM"], False),
        ],
    )
    def test_is_hybrid_backend_helper(self, architectures, expected_is_hybrid):
        """``_is_hybrid_backend`` should match the documented hybrid allow-list."""
        from nemo.collections.speechlm2.vllm.salm.config import _is_hybrid_backend

        assert _is_hybrid_backend(architectures) is expected_is_hybrid

    @pytest.mark.parametrize(
        "backbone_archs, expected_is_hybrid",
        [
            (["NemotronHForCausalLM"], True),
            (["NemotronHybridForCausalLM"], True),
            (["Qwen3ForCausalLM"], False),
        ],
    )
    def test_is_hybrid_set_from_backbone_architectures(self, monkeypatch, backbone_archs, expected_is_hybrid):
        """``cfg.is_hybrid`` is driven by the backbone HF config's ``architectures``."""

        def from_pretrained(model_name: str, trust_remote_code: bool = True):
            kwargs = dict(
                architectures=backbone_archs,
                hidden_size=2048,
                vocab_size=131072,
                num_hidden_layers=4,
            )
            if expected_is_hybrid:
                kwargs.update(num_key_value_heads=2, layer_norm_epsilon=1e-5)
            else:
                kwargs.update(rms_norm_eps=1e-6)
            return SimpleNamespace(**kwargs)

        monkeypatch.setattr(_config_module.AutoConfig, "from_pretrained", from_pretrained)

        cfg = NeMoSpeechLMConfig(**_DEFAULT_CONFIG_KWARGS)
        assert cfg.is_hybrid is expected_is_hybrid

    def test_hybrid_backbone_does_not_set_layer_types_shim(self):
        """Hybrid backbones must NOT have layer_types overridden -- the runtime
        is_hybrid escape hatch only fires when every layer is 'attention'."""
        cfg = NeMoSpeechLMConfig(**_DEFAULT_CONFIG_KWARGS)
        assert cfg.is_hybrid is True
        assert getattr(cfg.text_config, "layer_types", None) is None

    def test_transformer_backbone_engages_layer_types_shim(self):
        """Non-hybrid backbones get layer_types=['attention']*N so vLLM's
        ModelConfig.is_hybrid property returns False at runtime even though
        the model class declares IsHybrid (needed for NemotronH path)."""
        cfg = NeMoSpeechLMConfig(
            **{
                **_DEFAULT_CONFIG_KWARGS,
                "pretrained_llm": "Qwen/Qwen3-1.7B",
            }
        )
        assert cfg.is_hybrid is False
        assert cfg.text_config.layer_types == ["attention"] * 4

    def test_custom_pretrained_llm(self):
        """Config should accept different LLM backbones."""
        cfg = NeMoSpeechLMConfig(
            **{
                **_DEFAULT_CONFIG_KWARGS,
                "pretrained_llm": "Qwen/Qwen3-1.7B",
            }
        )
        assert cfg.pretrained_llm == "Qwen/Qwen3-1.7B"
        assert cfg.text_config is not None
        assert cfg.llm_architectures == ["Qwen3ForCausalLM"]

    def test_audio_locator_tag_default_accepted(self):
        cfg = NeMoSpeechLMConfig(**_DEFAULT_CONFIG_KWARGS)
        assert cfg.audio_locator_tag == "<|audio|>"

    def test_audio_locator_tag_custom_rejected(self):
        """Plugin only supports ``<|audio|>``; mismatched checkpoints fail at load time."""
        with pytest.raises(ValueError, match="audio_locator_tag"):
            NeMoSpeechLMConfig(
                **{
                    **_DEFAULT_CONFIG_KWARGS,
                    "audio_locator_tag": "<|custom_audio|>",
                }
            )

    @pytest.mark.parametrize(
        "field",
        [
            "pretrained_llm",
            "pretrained_asr",
            "audio_locator_tag",
            "prompt_format",
            "pretrained_weights",
        ],
    )
    def test_required_exported_fields(self, field):
        kwargs = dict(_DEFAULT_CONFIG_KWARGS)
        kwargs.pop(field)
        with pytest.raises(ValueError, match=field):
            NeMoSpeechLMConfig(**kwargs)

    def test_unknown_attr_raises(self):
        cfg = NeMoSpeechLMConfig(**_DEFAULT_CONFIG_KWARGS)
        with pytest.raises(AttributeError):
            _ = cfg.nonexistent_attribute_xyz

    def test_encoder_chunk_size_seconds_default_none(self):
        """Legacy checkpoints without a chunk size keep the single-pass encoder path."""
        cfg = NeMoSpeechLMConfig(**_DEFAULT_CONFIG_KWARGS)
        assert cfg.encoder_chunk_size_seconds is None

    def test_encoder_chunk_size_seconds_round_trips(self):
        """Chunk size set in config.json (e.g. SALMAutomodel default 30 s) survives load."""
        cfg = NeMoSpeechLMConfig(
            **{
                **_DEFAULT_CONFIG_KWARGS,
                "encoder_chunk_size_seconds": 30.0,
            }
        )
        assert cfg.encoder_chunk_size_seconds == 30.0

    def test_encoder_chunk_size_seconds_default_init_inert(self):
        """No-arg default init must still expose ``encoder_chunk_size_seconds=None``."""
        cfg = NeMoSpeechLMConfig()
        assert cfg.encoder_chunk_size_seconds is None


@pytest.mark.skipif(not (_HAS_CONFIG and _HAS_VLLM), reason="NeMoSpeechLMConfig or vLLM not available")
class TestBackendSelection:
    """Tests for ``backends.make_backend`` dispatch on hybrid/transformer configs."""

    @pytest.fixture(autouse=True)
    def mock_backbone_config(self, monkeypatch):
        def from_pretrained(model_name: str, trust_remote_code: bool = True):
            if "Nemotron" in model_name:
                return SimpleNamespace(
                    architectures=["NemotronHybridForCausalLM"],
                    hidden_size=2048,
                    vocab_size=131072,
                    num_hidden_layers=4,
                    num_key_value_heads=2,
                    layer_norm_epsilon=1e-5,
                )
            return SimpleNamespace(
                architectures=["Qwen3ForCausalLM"],
                hidden_size=2048,
                vocab_size=151936,
                num_hidden_layers=4,
                rms_norm_eps=1e-6,
            )

        monkeypatch.setattr(_config_module.AutoConfig, "from_pretrained", from_pretrained)

    def test_hybrid_config_picks_hybrid_backend(self):
        from nemo.collections.speechlm2.vllm.salm.backends import HybridBackend, make_backend

        cfg = NeMoSpeechLMConfig(**_DEFAULT_CONFIG_KWARGS)
        backend = make_backend(cfg)
        assert isinstance(backend, HybridBackend)
        assert backend.architectures() == ["NemotronHForCausalLM"]

    def test_transformer_config_picks_transformer_backend(self):
        from nemo.collections.speechlm2.vllm.salm.backends import TransformerBackend, make_backend

        cfg = NeMoSpeechLMConfig(
            **{
                **_DEFAULT_CONFIG_KWARGS,
                "pretrained_llm": "Qwen/Qwen3-1.7B",
            }
        )
        backend = make_backend(cfg)
        assert isinstance(backend, TransformerBackend)
        assert backend.architectures() == ["Qwen3ForCausalLM"]


@pytest.mark.skipif(not _HAS_VLLM, reason="vLLM not installed")
class TestSpecialTokens:
    """Tests for special token handling."""

    def test_adds_missing_token(self):
        from unittest.mock import MagicMock

        from nemo.collections.speechlm2.vllm.salm.audio import _ensure_special_tokens

        tokenizer = MagicMock()
        # MagicMock auto-creates attributes as truthy; set the memoization
        # sentinel explicitly so the once-per-tokenizer guard does not short-circuit.
        tokenizer._salm_special_tokens_ensured = False
        tokenizer.get_vocab.return_value = {}
        _ensure_special_tokens(tokenizer)
        tokenizer.add_special_tokens.assert_called_once()

    def test_skips_existing_token(self):
        from unittest.mock import MagicMock

        from nemo.collections.speechlm2.vllm.salm.audio import _ensure_special_tokens

        tokenizer = MagicMock()
        tokenizer._salm_special_tokens_ensured = False
        tokenizer.get_vocab.return_value = {"<|audio|>": 99}
        _ensure_special_tokens(tokenizer)
        tokenizer.add_special_tokens.assert_not_called()

    def test_memoizes_after_first_call(self):
        """The once-per-tokenizer guard avoids re-materializing the vocab each chunk."""
        from unittest.mock import MagicMock

        from nemo.collections.speechlm2.vllm.salm.audio import _ensure_special_tokens

        tokenizer = MagicMock()
        tokenizer._salm_special_tokens_ensured = False
        tokenizer.get_vocab.return_value = {}
        _ensure_special_tokens(tokenizer)
        _ensure_special_tokens(tokenizer)
        tokenizer.add_special_tokens.assert_called_once()
        assert tokenizer.get_vocab.call_count == 1

    def test_placeholder_str(self):
        from nemo.collections.speechlm2.vllm.salm.model import NeMoSpeechLMForConditionalGeneration

        assert NeMoSpeechLMForConditionalGeneration.get_placeholder_str("audio", 0) == "<|audio|>"
        assert NeMoSpeechLMForConditionalGeneration.get_placeholder_str("image", 0) is None


@pytest.mark.skipif(not _HAS_VLLM, reason="vLLM not installed")
class TestAudioProcessing:
    """Tests for audio encoding with a tiny perception module."""

    def test_data_parser_normalizes_audio(self, monkeypatch):
        from nemo.collections.speechlm2.vllm.salm.audio import NeMoSpeechLMProcessingInfo

        info = object.__new__(NeMoSpeechLMProcessingInfo)
        monkeypatch.setattr(info, "_get_expected_hidden_size", lambda: 2048)

        parser = info.get_data_parser()

        assert parser.audio_resampler.target_sr == 16000
        assert parser.target_channels == 1

    def test_processing_info_has_no_audio_duration_limit(self):
        from nemo.collections.speechlm2.vllm.salm.audio import NeMoSpeechLMProcessingInfo

        info = object.__new__(NeMoSpeechLMProcessingInfo)

        assert not hasattr(info, "get_max_audio_len")
        assert not hasattr(info, "get_max_audio_tokens")

    def test_dummy_inputs_use_profiling_audio_length(self):
        from nemo.collections.speechlm2.vllm.salm.audio import (
            NeMoSpeechLMDummyInputsBuilder,
            NeMoSpeechLMProcessingInfo,
        )

        info = object.__new__(NeMoSpeechLMProcessingInfo)
        builder = object.__new__(NeMoSpeechLMDummyInputsBuilder)
        builder.info = info

        result = builder.get_dummy_mm_data(seq_len=0, mm_counts={"audio": 1}, mm_options={})

        assert result["audio"][0].shape[-1] == 40 * 16000

    def test_dummy_inputs_use_requested_audio_length(self, monkeypatch):
        from nemo.collections.speechlm2.vllm.salm.audio import NeMoSpeechLMDummyInputsBuilder

        builder = object.__new__(NeMoSpeechLMDummyInputsBuilder)
        builder.info = SimpleNamespace(_get_encoder_chunk_size_seconds=lambda: None)
        monkeypatch.setattr(
            builder,
            "_get_dummy_audios",
            lambda length, num_audios: [SimpleNamespace(length=length) for _ in range(num_audios)],
        )

        result = builder.get_dummy_mm_data(
            seq_len=0,
            mm_counts={"audio": 1},
            mm_options={"audio": SimpleNamespace(length=12345)},
        )

        assert result["audio"][0].length == 12345

    def test_dummy_inputs_cap_requested_audio_length_to_text_budget(self, monkeypatch):
        from nemo.collections.speechlm2.vllm.salm.audio import (
            _DUMMY_AUDIO_TEXT_TOKEN_RESERVE,
            NeMoSpeechLMDummyInputsBuilder,
            NeMoSpeechLMProcessingInfo,
        )

        target_audio_tokens = 4
        max_audio_len = NeMoSpeechLMProcessingInfo._samples_for_audio_tokens(target_audio_tokens)
        builder = object.__new__(NeMoSpeechLMDummyInputsBuilder)
        builder.info = SimpleNamespace(_get_encoder_chunk_size_seconds=lambda: None)
        monkeypatch.setattr(
            builder,
            "_get_dummy_audios",
            lambda length, num_audios: [SimpleNamespace(length=length) for _ in range(num_audios)],
        )

        result = builder.get_dummy_mm_data(
            seq_len=_DUMMY_AUDIO_TEXT_TOKEN_RESERVE + target_audio_tokens,
            mm_counts={"audio": 1},
            mm_options={"audio": SimpleNamespace(length=max_audio_len + 16000)},
        )

        assert result["audio"][0].length == max_audio_len

    def test_dummy_inputs_large_seq_len_uses_max_audio_cap(self, monkeypatch):
        from nemo.collections.speechlm2.vllm.salm.audio import (
            _DUMMY_AUDIO_MAX_DURATION_S,
            _SAMPLING_RATE,
            NeMoSpeechLMDummyInputsBuilder,
        )

        max_audio_len = int(_DUMMY_AUDIO_MAX_DURATION_S * _SAMPLING_RATE)
        builder = object.__new__(NeMoSpeechLMDummyInputsBuilder)
        builder.info = SimpleNamespace(_get_encoder_chunk_size_seconds=lambda: None)
        monkeypatch.setattr(
            builder,
            "_get_dummy_audios",
            lambda length, num_audios: [SimpleNamespace(length=length) for _ in range(num_audios)],
        )

        result = builder.get_dummy_mm_data(
            seq_len=10_000_000,
            mm_counts={"audio": 1},
            mm_options={"audio": SimpleNamespace(length=max_audio_len + 16000)},
        )

        assert result["audio"][0].length == max_audio_len

    def test_call_hf_processor_requires_matching_placeholder_count(self):
        from nemo.collections.speechlm2.vllm.salm.audio import NeMoSpeechLMMultiModalProcessor

        processor = object.__new__(NeMoSpeechLMMultiModalProcessor)
        processor.info = SimpleNamespace(
            get_tokenizer=_FakeTokenizer,
            _estimate_audio_tokens=lambda samples, chunk_size_seconds=None: 2,
            _get_encoder_chunk_size_seconds=lambda: None,
        )

        with pytest.raises(ValueError, match="placeholders"):
            processor._call_hf_processor(
                prompt="Transcribe this audio",
                mm_data={"audios": [[0.0] * 16000]},
                mm_kwargs={},
                tok_kwargs={},
            )

    def test_call_hf_processor_emits_true_audio_lengths(self):
        import torch

        from nemo.collections.speechlm2.vllm.salm.audio import NeMoSpeechLMMultiModalProcessor

        processor = object.__new__(NeMoSpeechLMMultiModalProcessor)
        processor.info = SimpleNamespace(
            get_tokenizer=_FakeTokenizer,
            _estimate_audio_tokens=lambda samples, chunk_size_seconds=None: 2,
            _get_encoder_chunk_size_seconds=lambda: None,
        )

        result = processor._call_hf_processor(
            prompt="Transcribe: <|audio|>",
            mm_data={"audios": [[0.0] * 12345]},
            mm_kwargs={},
            tok_kwargs={},
        )

        assert len(result["audio_signal"]) == 1
        assert result["audio_signal"][0].shape[-1] == 12345
        assert torch.equal(result["audio_signal_length"], torch.tensor([12345]))

    def test_perception_forward(self):
        """A small NeMo perception module should encode dummy audio to embeddings."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        from nemo.collections.speechlm2.vllm.salm.audio import _load_nemo_perception

        perception_cfg = {
            "output_dim": 256,
            "encoder": {
                "_target_": "nemo.collections.asr.modules.ConformerEncoder",
                "feat_in": 128,
                "feat_out": -1,
                "n_layers": 2,
                "d_model": 256,
                "subsampling": "dw_striding",
                "subsampling_factor": 8,
                "subsampling_conv_channels": 64,
                "ff_expansion_factor": 4,
                "self_attention_model": "rel_pos",
                "n_heads": 4,
                "conv_kernel_size": 9,
                "conv_norm_type": "batch_norm",
                "dropout": 0.0,
                "dropout_pre_encoder": 0.0,
                "dropout_emb": 0.0,
                "dropout_att": 0.0,
            },
            "modality_adapter": {
                "_target_": "nemo.collections.speechlm2.modules.perception.IdentityConnector",
                "d_model": 256,
            },
            "preprocessor": {
                "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
                "sample_rate": 16000,
                "normalize": "per_feature",
                "window_size": 0.025,
                "window_stride": 0.01,
                "window": "hann",
                "features": 128,
                "n_fft": 512,
                "log": True,
                "frame_splicing": 1,
                "dither": 0.0,
                "pad_to": 0,
                "pad_value": 0.0,
            },
        }

        perception = _load_nemo_perception(perception_cfg)
        perception = perception.to("cuda", dtype=torch.float32)

        dummy_audio = torch.randn(1, 16000, device="cuda")
        audio_len = torch.tensor([16000], device="cuda")

        with torch.no_grad():
            embeds, embed_lens = perception(input_signal=dummy_audio, input_signal_length=audio_len)

        assert embeds.ndim == 3
        assert embeds.shape[0] == 1
        assert embeds.shape[2] == 256
        assert embed_lens[0] > 0


@pytest.mark.skipif(not _HAS_VLLM, reason="vLLM not installed")
class TestPluginRegistration:
    """Tests for plugin registration with vLLM."""

    def test_register_config(self, monkeypatch):
        """register() should add nemo_speechlm to vLLM's config registry."""
        from transformers import AutoConfig

        from nemo.collections.speechlm2.vllm.salm import register

        monkeypatch.setattr(
            AutoConfig, "from_pretrained", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError())
        )

        register()

        from vllm.transformers_utils.config import _CONFIG_REGISTRY

        assert "nemo_speechlm" in _CONFIG_REGISTRY

    def test_register_model(self, monkeypatch):
        """register() should make NeMoSpeechLMForConditionalGeneration importable.

        The plugin now registers a single architecture name; the obsolete
        ``NeMoSpeechLMHybridForConditionalGeneration`` no longer appears.
        """
        from transformers import AutoConfig

        from nemo.collections.speechlm2.vllm.salm import register

        monkeypatch.setattr(
            AutoConfig, "from_pretrained", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError())
        )

        register()

        from vllm.model_executor.models.registry import ModelRegistry

        from nemo.collections.speechlm2.vllm.salm.model import NeMoSpeechLMForConditionalGeneration

        assert "NeMoSpeechLMForConditionalGeneration" in ModelRegistry.get_supported_archs()
        assert NeMoSpeechLMForConditionalGeneration is not None

    def test_register_does_not_patch_fast_tokenizer(self, monkeypatch):
        from transformers import AutoConfig, PreTrainedTokenizerFast

        from nemo.collections.speechlm2.vllm.salm import register

        monkeypatch.setattr(
            AutoConfig, "from_pretrained", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError())
        )

        assert "_orig_batch_encode_plus" not in PreTrainedTokenizerFast.__dict__
        register()
        assert "_orig_batch_encode_plus" not in PreTrainedTokenizerFast.__dict__

    def test_register_does_not_load_backbone_config(self, monkeypatch):
        from unittest.mock import Mock

        from transformers import AutoConfig

        from nemo.collections.speechlm2.vllm.salm import register

        from_pretrained = Mock(side_effect=AssertionError("register() must not load remote backbone configs"))
        monkeypatch.setattr(AutoConfig, "from_pretrained", from_pretrained)

        register()

        from_pretrained.assert_not_called()


class _FakeTokenizer:
    def __init__(self):
        self.added_special_tokens = None

    def get_vocab(self):
        return {}

    def add_special_tokens(self, tokens):
        self.added_special_tokens = tokens

    def encode(self, prompt, add_special_tokens=True):
        return list(range(max(1, len(prompt.split()))))


class TestStreamingMarkers:
    """Tests for StreamingMarkers (no vLLM / GPU required)."""

    _MARKERS = {
        "chunk_size": 14,
        "user_header_ids": [151644, 872, 198],
        "uf_ah_ids": [151645, 198, 151644, 77091, 198],
        "asst_footer_ids": [151645, 198],
        "blank_token_id": 151669,
        "eos_id": 151645,
        "has_blank": True,
        "audio_id": 151670,
        "system_prompt": "Transcribe the audio into text.",
    }

    def test_from_dict(self):
        from nemo.collections.speechlm2.vllm.salm.streaming_session import StreamingMarkers

        m = StreamingMarkers.from_dict(self._MARKERS)
        assert m.chunk_size == 14
        assert m.blank_token_id == 151669
        assert m.eos_id == 151645
        assert m.audio_id == 151670
        assert m.asst_footer_ids == [151645, 198]
        assert m.user_header_ids == [151644, 872, 198]

    def test_from_config(self):
        from nemo.collections.speechlm2.vllm.salm.streaming_session import StreamingMarkers

        m = StreamingMarkers.from_config(SimpleNamespace(streaming_markers=self._MARKERS))
        assert m.chunk_size == 14
        assert m.blank_token_id == 151669

    def test_from_config_missing_raises(self):
        from nemo.collections.speechlm2.vllm.salm.streaming_session import StreamingMarkers

        with pytest.raises(ValueError, match="streaming_markers"):
            StreamingMarkers.from_config(SimpleNamespace(streaming_markers=None))


class _FakeSamplingParams:
    """Stand-in for ``vllm.SamplingParams`` (records the session's decode config)."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeStreamingEngine:
    """Async engine double replaying a canned list of DELTA token groups."""

    def __init__(self, deltas):
        self.deltas = deltas
        self.stream_inputs = []
        self.abort_calls = []
        self.reset_calls = 0

    async def generate(self, inputs, sampling_params, request_id):
        async for stream_input in inputs:
            self.stream_inputs.append(stream_input)
        for index, token_ids in enumerate(self.deltas):
            yield SimpleNamespace(
                outputs=[SimpleNamespace(token_ids=token_ids, finish_reason=None)],
                finished=index == len(self.deltas) - 1,
            )

    async def abort(self, request_id):
        """Awaitable abort -- the session must also accept a synchronous one."""
        self.abort_calls.append(request_id)

    def reset_mm_cache(self):
        """Synchronous reset -- the session must also accept an awaitable one."""
        self.reset_calls += 1


class _FakeStreamingTokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return [11, 12]

    def decode(self, token_ids, skip_special_tokens=True):
        words = {42: " hello ", 43: "world"}
        return "".join(words.get(token_id, f"token-{token_id}") for token_id in token_ids)


_LEGACY_STREAMING_MARKERS = {
    "chunk_size": 14,
    "blank_token_id": 100,
    "eos_id": 101,
    "audio_id": 102,
    "asst_footer_ids": [101],
    "user_header_ids": [20],
    "uf_ah_ids": [21],
    "has_blank": True,
}

_TTM_STREAMING_MARKERS = {
    "chunk_size": 2,
    "blank_token_id": 100,
    "eos_id": 101,
    "audio_id": 102,
    "sou_id": 103,
    "eou_id": 104,
    "asst_footer_ids": [101],
    "user_header_ids": [],
    "uf_ah_ids": [105],
    "has_blank": True,
    "frame_length_seconds": 0.08,
}


class TestStreamingBoundaryMarkers:
    """SOU/EOU marker parsing and validation (no vLLM / GPU required)."""

    def test_legacy_markers_have_no_boundaries(self):
        from nemo.collections.speechlm2.vllm.salm.streaming_session import StreamingMarkers

        m = StreamingMarkers.from_dict(_LEGACY_STREAMING_MARKERS)
        assert m.has_boundaries is False
        assert m.sou_id is None
        assert m.eou_id is None

    def test_flat_boundary_pair_is_parsed(self):
        from nemo.collections.speechlm2.vllm.salm.streaming_session import StreamingMarkers

        m = StreamingMarkers.from_dict(_TTM_STREAMING_MARKERS)
        assert m.has_boundaries is True
        assert (m.sou_id, m.eou_id) == (103, 104)
        assert m.frame_length_seconds == 0.08

    def test_nested_boundary_pair_is_parsed(self):
        from nemo.collections.speechlm2.vllm.salm.streaming_session import StreamingMarkers

        markers = {k: v for k, v in _TTM_STREAMING_MARKERS.items() if k not in ("sou_id", "eou_id")}
        markers["boundary_markers"] = {"sou_id": 103, "eou_id": 104}

        m = StreamingMarkers.from_dict(markers)
        assert m.has_boundaries is True
        assert (m.sou_id, m.eou_id) == (103, 104)

    def test_partial_boundary_pair_is_rejected(self):
        from nemo.collections.speechlm2.vllm.salm.streaming_session import StreamingMarkers

        with pytest.raises(ValueError, match="both be set"):
            StreamingMarkers.from_dict({**_TTM_STREAMING_MARKERS, "eou_id": None})

    def test_colliding_boundary_ids_are_rejected(self):
        from nemo.collections.speechlm2.vllm.salm.streaming_session import StreamingMarkers

        with pytest.raises(ValueError, match="distinct"):
            StreamingMarkers.from_dict({**_TTM_STREAMING_MARKERS, "eou_id": 101})

    def test_noblank_sentinel_does_not_collide(self):
        """``blank_token_id=-1`` is a sentinel for noblank checkpoints, not a vocab id."""
        from nemo.collections.speechlm2.vllm.salm.streaming_session import StreamingMarkers

        m = StreamingMarkers.from_dict({**_TTM_STREAMING_MARKERS, "blank_token_id": -1, "has_blank": False})
        assert m.has_boundaries is True

    def test_non_positive_frame_length_is_rejected(self):
        from nemo.collections.speechlm2.vllm.salm.streaming_session import StreamingMarkers

        with pytest.raises(ValueError, match="frame_length_seconds"):
            StreamingMarkers.from_dict({**_TTM_STREAMING_MARKERS, "frame_length_seconds": 0.0})


class TestStreamingSTTSessionBoundaries:
    """Turn-boundary (TTM) session behavior against a fake vLLM engine.

    The session only touches a handful of vLLM symbols, all imported lazily, so
    the tests install minimal fakes instead of requiring a vLLM install.
    """

    @pytest.fixture
    def streaming_session(self, monkeypatch):
        import sys
        from types import ModuleType

        # Import the module (and its package) against the real environment first,
        # so the fakes below only stand in for the lazily imported call sites.
        from nemo.collections.speechlm2.vllm.salm import streaming_session as module

        def fake_module(name: str, **attrs) -> ModuleType:
            mod = ModuleType(name)
            for attr, value in attrs.items():
                setattr(mod, attr, value)
            monkeypatch.setitem(sys.modules, name, mod)
            return mod

        fake_module("vllm", SamplingParams=_FakeSamplingParams)
        fake_module("vllm.sampling_params", RequestOutputKind=SimpleNamespace(DELTA="DELTA"))
        engine_module = fake_module("vllm.engine")
        engine_module.protocol = fake_module(
            "vllm.engine.protocol", StreamingInput=lambda **kwargs: SimpleNamespace(**kwargs)
        )
        fake_module("vllm.inputs", TokensPrompt=lambda **kwargs: SimpleNamespace(**kwargs))
        return module

    def _session(self, module, markers_dict, *, deltas, **kwargs):
        """Build a session over a fake engine replaying ``deltas``."""
        engine = _FakeStreamingEngine(deltas)
        session = module.StreamingSTTSession(
            engine,
            _FakeStreamingTokenizer(),
            module.StreamingMarkers.from_dict(markers_dict),
            **kwargs,
        )
        return session, engine

    def test_legacy_checkpoint_skips_boundary_cleanup(self, streaming_session):
        session, engine = self._session(streaming_session, _LEGACY_STREAMING_MARKERS, deltas=[[42, 100]])

        result = asyncio.run(session.transcribe_with_events([object()], "legacy-request"))

        assert result.text == "hello"
        assert result.token_ids == [42, 100]
        assert result.boundary_events == []
        assert result.ended_by_eou is False
        assert result.request_aborted is False
        assert result.mm_cache_reset is False
        assert engine.abort_calls == []
        assert engine.reset_calls == 0
        # Legacy stop-token behavior is unchanged: blank + EOS, nothing else.
        assert session._chunk_sp.stop_token_ids == [100, 101]

    def test_transcribe_stays_transcript_only(self, streaming_session):
        """``transcribe()`` keeps its ``str`` return type on both paths."""
        legacy, _ = self._session(streaming_session, _LEGACY_STREAMING_MARKERS, deltas=[[42, 100]])
        ttm, _ = self._session(streaming_session, _TTM_STREAMING_MARKERS, deltas=[[103, 42, 104]])

        assert asyncio.run(legacy.transcribe([object()], "legacy")) == "hello"
        assert asyncio.run(ttm.transcribe([object()], "ttm")) == "hello"

    def test_transcribe_many_returns_strings_on_both_paths(self, streaming_session):
        legacy, _ = self._session(streaming_session, _LEGACY_STREAMING_MARKERS, deltas=[[42, 100]])
        ttm, ttm_engine = self._session(streaming_session, _TTM_STREAMING_MARKERS, deltas=[[103, 42, 104]])

        assert asyncio.run(legacy.transcribe_many([[object()], [object()]])) == ["hello", "hello"]
        assert asyncio.run(ttm.transcribe_many([[object()], [object()]])) == ["hello", "hello"]
        assert ttm_engine.reset_calls == 1

    def test_eou_is_retained_observed_and_closes_request(self, streaming_session):
        session, engine = self._session(streaming_session, _TTM_STREAMING_MARKERS, deltas=[[103, 42, 104, 43]])
        callbacks = []

        async def on_boundary(event):
            callbacks.append(event)

        result = asyncio.run(session.transcribe_with_events([object()], "ttm-request", on_boundary=on_boundary))

        assert result.text == "hello"
        # The EOU token is kept; tokens after it in the same DELTA are dropped.
        assert result.token_ids == [103, 42, 104]
        assert [event.boundary_type for event in result.boundary_events] == ["sou", "eou"]
        assert [event.token_index for event in result.boundary_events] == [0, 2]
        assert callbacks == result.boundary_events
        assert result.boundary_events[-1].chunk_end_audio_seconds == pytest.approx(0.16)
        assert result.ended_by_eou is True
        assert result.request_aborted is True
        assert result.mm_cache_reset is True
        assert engine.abort_calls == ["ttm-request"]
        assert engine.reset_calls == 1
        # EOU must stay visible to the consumer, so it is never a vLLM stop token.
        assert 104 not in session._chunk_sp.stop_token_ids

    def test_sync_boundary_callback_is_supported(self, streaming_session):
        session, _ = self._session(streaming_session, _TTM_STREAMING_MARKERS, deltas=[[103, 104]])
        callbacks = []

        result = asyncio.run(session.transcribe_with_events([object()], "sync-callback", on_boundary=callbacks.append))

        assert callbacks == result.boundary_events

    def test_eou_cache_reset_can_be_disabled(self, streaming_session):
        session, engine = self._session(
            streaming_session, _TTM_STREAMING_MARKERS, deltas=[[104]], reset_mm_cache_on_eou=False
        )

        result = asyncio.run(session.transcribe_with_events([object()], "no-reset"))

        assert result.ended_by_eou is True
        assert result.request_aborted is True
        assert result.mm_cache_reset is False
        assert engine.abort_calls == ["no-reset"]
        assert engine.reset_calls == 0

    def test_observe_only_mode_continues_after_eou(self, streaming_session):
        session, engine = self._session(
            streaming_session, _TTM_STREAMING_MARKERS, deltas=[[103, 42, 104, 43, 101]], close_on_eou=False
        )

        result = asyncio.run(session.transcribe_with_events([object()], "observe-only"))

        assert result.text == "hello world"
        assert result.token_ids == [103, 42, 104, 43, 101]
        assert [event.boundary_type for event in result.boundary_events] == ["sou", "eou"]
        assert result.ended_by_eou is False
        assert result.request_aborted is False
        assert result.mm_cache_reset is False
        assert engine.abort_calls == []
        assert engine.reset_calls == 0

    def test_coalesced_deltas_keep_boundary_chunk_times(self, streaming_session):
        """One DELTA may carry several chunks' stop tokens; timestamps must follow them."""
        session, _ = self._session(streaming_session, _TTM_STREAMING_MARKERS, deltas=[[100, 100, 103, 101, 100, 104]])

        result = asyncio.run(session.transcribe_with_events([object()] * 5, "coalesced"))

        assert [event.chunk_index for event in result.boundary_events] == [2, 4]
        assert [event.chunk_end_audio_seconds for event in result.boundary_events] == [
            pytest.approx(0.48),
            pytest.approx(0.8),
        ]

    def test_concurrent_batch_defers_single_cache_reset(self, streaming_session):
        session, engine = self._session(streaming_session, _TTM_STREAMING_MARKERS, deltas=[[103, 104]])

        results = asyncio.run(
            session.transcribe_many_with_events([[object()], [object()]], tag="batch", concurrency=2)
        )

        assert sorted(engine.abort_calls) == ["batch_0", "batch_1"]
        # One engine-global reset for the whole batch, after every session ended.
        assert engine.reset_calls == 1
        assert all(result.ended_by_eou for result in results)
        assert all(result.mm_cache_reset for result in results)

    def test_concurrent_batch_without_eou_does_not_reset(self, streaming_session):
        session, engine = self._session(streaming_session, _TTM_STREAMING_MARKERS, deltas=[[103, 42, 100]])

        results = asyncio.run(session.transcribe_many_with_events([[object()], [object()]], tag="batch"))

        assert engine.abort_calls == []
        assert engine.reset_calls == 0
        assert not any(result.ended_by_eou for result in results)


class TestStreamingSchedulerPatch:
    """Tests for the StreamingSTT session-retention scheduler patch."""

    def test_retain_flag_constant(self):
        from nemo.collections.speechlm2.vllm.salm.streaming_constants import RETAIN_FLAG

        assert RETAIN_FLAG == "streaming_stt_retain_until_blank"

    @pytest.mark.skipif(not _HAS_VLLM, reason="vLLM not installed")
    def test_install_is_idempotent(self):
        from vllm.v1.core.sched.scheduler import Scheduler

        from nemo.collections.speechlm2.vllm.salm.streaming_scheduler import install_streaming_session_patch

        install_streaming_session_patch()
        assert getattr(Scheduler, "_nemo_streaming_stt_patched", False) is True
        patched = Scheduler._update_request_as_session
        install_streaming_session_patch()  # second call must be a no-op
        assert Scheduler._update_request_as_session is patched
