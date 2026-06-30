from __future__ import annotations

from typing import Any


def build_transcription_messages(
    audio: Any,
    *,
    context: str = "",
    language: str | None = None,
) -> list[dict[str, Any]]:
    """Build Qwen3-ASR chat messages following the official request shape.

    Transformers' native ``apply_transcription_request`` currently exposes a
    language hint but no separate recognition context. For context-bearing
    requests, keep the same chat shape as the official helper: one system
    message, one audio user message, then the assistant generation prompt. The
    context is a weak textual hint, not a hotword or forced vocabulary API.
    """

    system_parts = []
    if language:
        system_parts.append(str(language).strip())
    context = str(context or "").strip()
    if context:
        system_parts.append(f"Context hint: {context}")
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
    context: str = "",
    language: str | None = None,
) -> str:
    messages = build_transcription_messages(
        "",
        context=context,
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
    contexts: list[str],
    language: str | None,
):
    if len(audio) != len(contexts):
        raise ValueError(
            f"audio/context count mismatch: audio={len(audio)}, contexts={len(contexts)}"
        )
    if not any(str(context or "").strip() for context in contexts):
        languages = [language] * len(audio) if language else None
        return processor.apply_transcription_request(
            audio=audio,
            language=languages,
        )

    conversations = [
        build_transcription_messages(
            audio_item,
            context=str(context or ""),
            language=language,
        )
        for audio_item, context in zip(audio, contexts, strict=True)
    ]
    return processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        processor_kwargs={"padding": True},
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
