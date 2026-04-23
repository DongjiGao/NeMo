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
import torch

from nemo.collections.common.prompts.qwen import QwenPromptFormatter


def test_qwen_prompt_formatter_training(bpe_tokenizer):
    formatter = QwenPromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids", "answer_ids", "mask"}
    # fmt: off
    # The test tokenizer inserts an extra space, but it was verified that AutoTokenizer("Qwen/Qwen3-1.7B") doesn't.
    # Matches ``Qwen/Qwen3-1.7B`` chat_template with ``enable_thinking=False``
    # (the SpeechLM reasoning-off mode). ``<think>\n\n</think>\n\n`` is the
    # empty reasoning block Qwen3's jinja injects.
    assert bpe_tokenizer.ids_to_text(ans["input_ids"].tolist()) == '<|im_start|>user\nTEST<|im_end|>\n <|im_start|>assistant\n<think>\n\n</think>\n\nTEST<|im_end|>\n'
    assert bpe_tokenizer.ids_to_text(ans["context_ids"].tolist()) == '<|im_start|>user\nTEST<|im_end|>\n'
    assert bpe_tokenizer.ids_to_text(ans["answer_ids"].tolist()) == '<|im_start|>assistant\n<think>\n\n</think>\n\nTEST<|im_end|>\n'
    assert torch.is_tensor(ans["mask"])
    # fmt: on


def test_qwen_prompt_formatter_inference(bpe_tokenizer):
    formatter = QwenPromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids"}
    # fmt: off
    # The test tokenizer inserts an extra space, but it was verified that AutoTokenizer("Qwen/Qwen3-1.7B") doesn't.
    assert ans["input_ids"].tolist() == ans["context_ids"].tolist()
    # Matches ``Qwen/Qwen3-1.7B`` chat_template with
    # ``enable_thinking=False, add_generation_prompt=True``.
    assert bpe_tokenizer.ids_to_text(ans["input_ids"].tolist()) == '<|im_start|>user\nTEST<|im_end|>\n <|im_start|>assistant\n<think>\n\n</think>\n\n'
    # fmt: on
