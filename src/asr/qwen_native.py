from __future__ import annotations

from typing import Any


def build_transcription_messages(
    audio: Any,
    *,
    language: str | None = None,
) -> list[dict[str, Any]]:
    """Build Qwen3-ASR chat messages following the official request shape.

    Runtime transcription uses ``apply_transcription_request`` directly. This
    helper is kept for SFT prompt masking, where the prompt string is needed.
    """

    system_parts = []
    if language:
        system_parts.append(str(language).strip())
    messages: list[dict[str, Any]] = []
    if system_parts:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": "\n".join(system_parts)}],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [{"type": "audio", "audio": audio}],
        }
    )
    return messages


def build_transcription_prompt(
    processor: Any,
    *,
    language: str | None = None,
) -> str:
    messages = build_transcription_messages(
        "",
        language=language,
    )
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def prepare_transcription_inputs(
    processor: Any,
    *,
    audio: list[Any],
    language: str | None,
):
    languages = [language] * len(audio) if language else None
    return processor.apply_transcription_request(
        audio=audio,
        language=languages,
    )


def move_processor_inputs(inputs: Any, *, device: Any, dtype: Any) -> dict[str, Any]:
    import torch

    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif torch.is_floating_point(value):
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved
