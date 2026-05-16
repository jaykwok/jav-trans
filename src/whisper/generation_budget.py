from __future__ import annotations

from dataclasses import dataclass
from typing import Any


POLICY_VERSION = 1


@dataclass(frozen=True)
class GenerationBudget:
    policy_version: int
    configured_max_new_tokens: int
    effective_max_new_tokens: int
    model_max_target_positions: int | None
    decoder_prefix_tokens: int
    prompt_tokens_original: int
    prompt_tokens_kept: int
    prompt_tokens_dropped: int
    min_effective_new_tokens: int
    clipped_max_new_tokens: bool
    clipped_prompt_tokens: bool

    def as_dict(self) -> dict[str, int | bool | None]:
        return {
            "policy_version": self.policy_version,
            "configured_max_new_tokens": self.configured_max_new_tokens,
            "effective_max_new_tokens": self.effective_max_new_tokens,
            "model_max_target_positions": self.model_max_target_positions,
            "decoder_prefix_tokens": self.decoder_prefix_tokens,
            "prompt_tokens_original": self.prompt_tokens_original,
            "prompt_tokens_kept": self.prompt_tokens_kept,
            "prompt_tokens_dropped": self.prompt_tokens_dropped,
            "min_effective_new_tokens": self.min_effective_new_tokens,
            "clipped_max_new_tokens": self.clipped_max_new_tokens,
            "clipped_prompt_tokens": self.clipped_prompt_tokens,
        }


def _int_or_none(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _count_forced_decoder_tokens(forced_decoder_ids: Any) -> int:
    if not forced_decoder_ids:
        return 0
    count = 0
    try:
        for item in forced_decoder_ids:
            if item is None:
                continue
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                if item[1] is not None:
                    count += 1
            else:
                count += 1
    except TypeError:
        return 0
    return count


def _token_count(token_ids: Any) -> int:
    if token_ids is None:
        return 0
    numel = getattr(token_ids, "numel", None)
    if callable(numel):
        try:
            return int(numel())
        except (TypeError, ValueError):
            return 0
    try:
        return len(token_ids)
    except TypeError:
        return 0


def resolve_model_max_target_positions(model: Any) -> int | None:
    config = getattr(model, "config", None)
    value = _int_or_none(getattr(config, "max_target_positions", None))
    if value is not None:
        return value
    generation_config = getattr(model, "generation_config", None)
    return _int_or_none(getattr(generation_config, "max_length", None))


def _tail_slice(token_ids: Any, keep_tokens: int) -> Any:
    if token_ids is None or keep_tokens <= 0:
        return None
    count = _token_count(token_ids)
    if count <= keep_tokens:
        return token_ids
    try:
        return token_ids[..., -keep_tokens:]
    except Exception:
        return token_ids


def generation_budget_policy_version() -> int:
    return POLICY_VERSION


def apply_generation_budget(
    *,
    model: Any,
    generate_kwargs: dict[str, Any],
    prompt_ids: Any = None,
    min_effective_new_tokens: int = 64,
    decoder_start_tokens: int = 1,
) -> tuple[dict[str, Any], Any, GenerationBudget]:
    configured_max_new_tokens = max(1, int(generate_kwargs.get("max_new_tokens") or 1))
    min_effective_new_tokens = max(1, int(min_effective_new_tokens or 1))
    model_max = resolve_model_max_target_positions(model)
    forced_count = _count_forced_decoder_tokens(generate_kwargs.get("forced_decoder_ids"))
    decoder_prefix_tokens = max(0, int(decoder_start_tokens)) + forced_count
    prompt_original = _token_count(prompt_ids)

    if model_max is None:
        budget = GenerationBudget(
            policy_version=POLICY_VERSION,
            configured_max_new_tokens=configured_max_new_tokens,
            effective_max_new_tokens=configured_max_new_tokens,
            model_max_target_positions=None,
            decoder_prefix_tokens=decoder_prefix_tokens,
            prompt_tokens_original=prompt_original,
            prompt_tokens_kept=prompt_original,
            prompt_tokens_dropped=0,
            min_effective_new_tokens=min_effective_new_tokens,
            clipped_max_new_tokens=False,
            clipped_prompt_tokens=False,
        )
        kwargs = dict(generate_kwargs)
        kwargs["max_new_tokens"] = configured_max_new_tokens
        if prompt_ids is not None:
            kwargs["prompt_ids"] = prompt_ids
        return kwargs, prompt_ids, budget

    absolute_room = max(1, model_max - decoder_prefix_tokens)
    target_new_tokens = min(configured_max_new_tokens, absolute_room)
    target_new_tokens = max(1, target_new_tokens)
    protected_new_tokens = min(max(1, min_effective_new_tokens), target_new_tokens)
    prompt_room = max(0, model_max - decoder_prefix_tokens - protected_new_tokens)
    prompt_keep = min(prompt_original, prompt_room)
    clipped_prompt = prompt_original > prompt_keep
    effective_prompt_ids = _tail_slice(prompt_ids, prompt_keep) if prompt_keep else None
    prompt_kept = _token_count(effective_prompt_ids)
    remaining_room = max(1, model_max - decoder_prefix_tokens - prompt_kept)
    effective_max_new_tokens = max(1, min(configured_max_new_tokens, remaining_room))

    kwargs = dict(generate_kwargs)
    kwargs["max_new_tokens"] = effective_max_new_tokens
    if effective_prompt_ids is not None:
        kwargs["prompt_ids"] = effective_prompt_ids
    else:
        kwargs.pop("prompt_ids", None)

    budget = GenerationBudget(
        policy_version=POLICY_VERSION,
        configured_max_new_tokens=configured_max_new_tokens,
        effective_max_new_tokens=effective_max_new_tokens,
        model_max_target_positions=model_max,
        decoder_prefix_tokens=decoder_prefix_tokens,
        prompt_tokens_original=prompt_original,
        prompt_tokens_kept=prompt_kept,
        prompt_tokens_dropped=max(0, prompt_original - prompt_kept),
        min_effective_new_tokens=min_effective_new_tokens,
        clipped_max_new_tokens=effective_max_new_tokens < configured_max_new_tokens,
        clipped_prompt_tokens=clipped_prompt,
    )
    return kwargs, effective_prompt_ids, budget
