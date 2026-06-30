from __future__ import annotations

from typing import Any


def build_transcription_prompt(
    processor: Any,
    *,
    context: str = "",
    language: str | None = None,
) -> str:
    """Build the context-aware Qwen3-ASR prompt used by the former wrapper.

    Transformers' native ``apply_transcription_request`` currently exposes a
    language hint but no separate recognition context. For context-bearing
    requests, preserve the model's original prompt contract: context in the
    system message and an optional language/asr-text assistant prefill.
    """

    messages = [
        {"role": "system", "content": context or ""},
        {
            "role": "user",
            "content": [{"type": "audio", "audio": ""}],
        },
    ]
    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    if language:
        prompt += f"language {language}<asr_text>"
    return prompt


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

    prompts = [
        build_transcription_prompt(
            processor,
            context=str(context or ""),
            language=language,
        )
        for context in contexts
    ]
    return processor(
        text=prompts,
        audio=audio,
        return_tensors="pt",
        padding=True,
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
