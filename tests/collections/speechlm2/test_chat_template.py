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

"""Validate that PromptFormatter.to_jinja() produces templates matching encode_dialog().

For each formatter that implements to_jinja(), renders the Jinja template
with sample messages and compares the output against what the Python
PromptFormatter would produce (text-level comparison). This catches
drift between training prompt format and inference chat template.
"""

import pytest
from jinja2 import Template


def _render_jinja(template_str: str, messages: list[dict], add_generation_prompt: bool = True) -> str:
    return Template(template_str).render(messages=messages, add_generation_prompt=add_generation_prompt)


def _load_module_direct(name, filepath):
    """Load a Python module directly from file, bypassing package __init__.py."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-load formatter base class (needed by subclasses)
import os as _os

_PROMPTS_DIR = _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "nemo", "collections", "common", "prompts")
_PROMPTS_DIR = _os.path.normpath(_PROMPTS_DIR)
_formatter_mod = _load_module_direct(
    "nemo.collections.common.prompts.formatter", _os.path.join(_PROMPTS_DIR, "formatter.py")
)
_qwen_mod = _load_module_direct("nemo.collections.common.prompts.qwen", _os.path.join(_PROMPTS_DIR, "qwen.py"))
_nano_mod = _load_module_direct(
    "nemo.collections.common.prompts.nemotron_nano_v3", _os.path.join(_PROMPTS_DIR, "nemotron_nano_v3.py")
)


def _get_qwen_template():
    return _qwen_mod.QwenPromptFormatter.to_jinja()


def _get_nano_template(**kwargs):
    return _nano_mod.NemotronNanoV3PromptFormatter.to_jinja(**kwargs)


class TestQwenChatTemplate:
    """Validate QwenPromptFormatter.to_jinja() against training format."""

    @pytest.fixture
    def template(self):
        return _get_qwen_template()

    def test_text_only_user(self, template):
        result = _render_jinja(template, [{"role": "user", "content": "Hello"}])
        assert result == "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"

    def test_multimodal_user(self, template):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe:"},
                    {"type": "input_audio", "input_audio": {"data": "base64"}},
                ],
            }
        ]
        result = _render_jinja(template, messages)
        assert "<|audio|>" in result
        assert result == "<|im_start|>user\nTranscribe: <|audio|><|im_end|>\n<|im_start|>assistant\n"

    def test_no_generation_prompt(self, template):
        result = _render_jinja(template, [{"role": "user", "content": "Hi"}], add_generation_prompt=False)
        assert result == "<|im_start|>user\nHi<|im_end|>\n"
        assert "assistant" not in result

    def test_no_thinking_tags(self, template):
        result = _render_jinja(template, [{"role": "user", "content": "Test"}])
        assert "<think>" not in result


class TestNemotronNanoV3ChatTemplate:
    """Validate NemotronNanoV3PromptFormatter.to_jinja() against training format."""

    @pytest.fixture
    def template(self):
        return _get_nano_template(enable_thinking=False)

    @pytest.fixture
    def template_thinking(self):
        return _get_nano_template(enable_thinking=True)

    def test_auto_empty_system_turn(self, template):
        """NeMo training always inserts empty system turn if missing."""
        result = _render_jinja(template, [{"role": "user", "content": "Transcribe:"}])
        assert result.startswith("<|im_start|>system\n<|im_end|>\n")

    def test_explicit_system_no_duplicate(self, template):
        """Providing a system message should not create a duplicate."""
        result = _render_jinja(
            template,
            [
                {"role": "system", "content": ""},
                {"role": "user", "content": "Transcribe:"},
            ],
        )
        assert result.count("<|im_start|>system") == 1

    def test_thinking_disabled(self, template):
        result = _render_jinja(template, [{"role": "user", "content": "Test"}])
        assert "<think></think>" in result
        assert "<think>\n" not in result

    def test_thinking_enabled(self, template_thinking):
        result = _render_jinja(template_thinking, [{"role": "user", "content": "Test"}])
        assert "<think>\n" in result
        assert "<think></think>" not in result

    def test_multimodal_with_audio(self, template):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe the following:"},
                    {"type": "input_audio", "input_audio": {"data": "base64"}},
                ],
            }
        ]
        result = _render_jinja(template, messages)
        assert "<|audio|>" in result
        assert "Transcribe the following: <|audio|>" in result

    def test_full_inference_format(self, template):
        """Full format matching NeMo training: empty system + user + gen prompt."""
        result = _render_jinja(
            template,
            [{"role": "user", "content": [{"type": "text", "text": "Transcribe:"}, {"type": "audio", "audio": {}}]}],
        )
        expected = (
            "<|im_start|>system\n<|im_end|>\n"
            "<|im_start|>user\nTranscribe: <|audio|><|im_end|>\n"
            "<|im_start|>assistant\n<think></think>"
        )
        assert result == expected


class TestToJinjaBaseClass:
    """Validate base PromptFormatter.to_jinja() behavior."""

    def test_base_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="does not support Jinja"):
            _formatter_mod.PromptFormatter.to_jinja()

    def test_custom_audio_token(self):
        tmpl = _qwen_mod.QwenPromptFormatter.to_jinja(audio_token="<audio>")
        result = _render_jinja(
            tmpl,
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hi"}, {"type": "input_audio", "input_audio": {}}],
                }
            ],
        )
        assert "<audio>" in result
        assert "<|audio|>" not in result

    def test_nano_enable_thinking_kwarg(self):
        tmpl_off = _nano_mod.NemotronNanoV3PromptFormatter.to_jinja(enable_thinking=False)
        tmpl_on = _nano_mod.NemotronNanoV3PromptFormatter.to_jinja(enable_thinking=True)
        assert tmpl_off != tmpl_on

    def test_qwen_and_nano_use_same_bot_eot(self):
        """Templates use module-level constants, not hardcoded strings."""
        assert _qwen_mod.QWEN_BOT == "<|im_start|>"
        assert _nano_mod.NANO_BOT == "<|im_start|>"
        qwen_tmpl = _qwen_mod.QwenPromptFormatter.to_jinja()
        nano_tmpl = _nano_mod.NemotronNanoV3PromptFormatter.to_jinja()
        assert _qwen_mod.QWEN_BOT in qwen_tmpl
        assert _nano_mod.NANO_BOT in nano_tmpl
