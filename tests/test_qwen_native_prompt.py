from __future__ import annotations

from asr.qwen_native import build_transcription_prompt, prepare_transcription_inputs


class FakeProcessor:
    def __init__(self) -> None:
        self.transcription_request = None
        self.chat_template_request = None

    def apply_transcription_request(self, **kwargs):
        self.transcription_request = kwargs
        return {"official": True}

    def apply_chat_template(self, conversations, **kwargs):
        self.chat_template_request = {
            "conversations": conversations,
            "kwargs": kwargs,
        }
        if kwargs.get("tokenize"):
            return {"chat_template": True}
        system_text = conversations[0]["content"][0]["text"]
        return (
            "<|im_start|>system\n"
            f"{system_text}<|im_end|>\n"
            "<|im_start|>user\n"
            "<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
            "<|im_start|>assistant\n"
        )


def test_prepare_transcription_inputs_without_context_uses_official_helper():
    processor = FakeProcessor()

    result = prepare_transcription_inputs(
        processor,
        audio=["a.wav"],
        contexts=[""],
        language="Japanese",
    )

    assert result == {"official": True}
    assert processor.transcription_request == {
        "audio": ["a.wav"],
        "language": ["Japanese"],
    }
    assert processor.chat_template_request is None


def test_prepare_transcription_inputs_with_context_uses_tokenized_chat_template():
    processor = FakeProcessor()

    result = prepare_transcription_inputs(
        processor,
        audio=["a.wav"],
        contexts=["小那海あや"],
        language="Japanese",
    )

    assert result == {"chat_template": True}
    request = processor.chat_template_request
    assert request["kwargs"] == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "processor_kwargs": {"padding": True},
    }
    messages = request["conversations"][0]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"][0]["text"] == "Japanese\nContext hint: 小那海あや"
    assert messages[1]["content"][0]["audio"] == "a.wav"


def test_build_transcription_prompt_does_not_use_legacy_language_prefill():
    prompt = build_transcription_prompt(
        FakeProcessor(),
        context="小那海あや",
        language="Japanese",
    )

    assert "Context hint: 小那海あや" in prompt
    assert "language Japanese<asr_text>" not in prompt
