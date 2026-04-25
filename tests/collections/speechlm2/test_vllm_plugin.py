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

"""Unit tests for the vLLM NeMo Speech LM plugin.

Tests plugin registration, config loading, and special token handling
without requiring GPU or model weights.
"""

import importlib.util

import pytest

try:
    from nemo.collections.speechlm2.vllm.nemotron_v3.config import NeMoSpeechLMConfig

    _HAS_CONFIG = True
except (ImportError, RuntimeError):
    _HAS_CONFIG = False

_HAS_VLLM = importlib.util.find_spec("vllm") is not None


@pytest.mark.skipif(not _HAS_CONFIG, reason="NeMoSpeechLMConfig not available")
class TestNeMoSpeechLMConfig:
    """Tests for NeMoSpeechLMConfig."""

    def test_model_type(self):
        assert NeMoSpeechLMConfig.model_type == "nemo_speechlm"

    def test_loads_text_config(self):
        """Config should load a text_config from the pretrained LLM."""
        cfg = NeMoSpeechLMConfig()
        assert cfg.text_config is not None
        assert hasattr(cfg.text_config, "hidden_size")
        assert cfg.get_text_config() is cfg.text_config

    def test_custom_pretrained_llm(self):
        """Config should accept different LLM backbones."""
        cfg = NeMoSpeechLMConfig(pretrained_llm="Qwen/Qwen3-1.7B")
        assert cfg.pretrained_llm == "Qwen/Qwen3-1.7B"
        assert cfg.text_config is not None

    def test_audio_locator_tag_configurable(self):
        cfg = NeMoSpeechLMConfig(audio_locator_tag="<|custom_audio|>")
        assert cfg.audio_locator_tag == "<|custom_audio|>"

    def test_unknown_attr_raises(self):
        cfg = NeMoSpeechLMConfig()
        with pytest.raises(AttributeError):
            _ = cfg.nonexistent_attribute_xyz


@pytest.mark.skipif(not _HAS_VLLM, reason="vLLM not installed")
class TestSpecialTokens:
    """Tests for special token handling."""

    def test_adds_missing_token(self):
        from unittest.mock import MagicMock

        from nemo.collections.speechlm2.vllm.nemotron_v3.model import _ensure_special_tokens

        tokenizer = MagicMock()
        tokenizer.get_vocab.return_value = {}
        _ensure_special_tokens(tokenizer)
        tokenizer.add_special_tokens.assert_called_once()

    def test_skips_existing_token(self):
        from unittest.mock import MagicMock

        from nemo.collections.speechlm2.vllm.nemotron_v3.model import _ensure_special_tokens

        tokenizer = MagicMock()
        tokenizer.get_vocab.return_value = {"<|audio|>": 99}
        _ensure_special_tokens(tokenizer)
        tokenizer.add_special_tokens.assert_not_called()


@pytest.mark.skipif(not _HAS_VLLM, reason="vLLM not installed")
class TestAudioProcessing:
    """Tests for audio encoding with a tiny perception module."""

    def test_perception_forward(self):
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        """A small NeMo perception module should encode dummy audio to embeddings."""
        from nemo.collections.speechlm2.vllm.nemotron_v3.model import _load_nemo_perception

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

    def test_register_config(self):
        """register() should add nemo_speechlm to vLLM's config registry."""
        from nemo.collections.speechlm2.vllm.nemotron_v3 import register

        register()

        from vllm.transformers_utils.config import _CONFIG_REGISTRY

        assert "nemo_speechlm" in _CONFIG_REGISTRY

    def test_register_model(self):
        """register() should make NeMoSpeechLMForConditionalGeneration importable."""
        from nemo.collections.speechlm2.vllm.nemotron_v3 import register

        register()

        from nemo.collections.speechlm2.vllm.nemotron_v3.model import NeMoSpeechLMForConditionalGeneration

        assert NeMoSpeechLMForConditionalGeneration is not None
