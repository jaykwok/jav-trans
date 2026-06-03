from __future__ import annotations

from types import SimpleNamespace

import torch

from asr.generation_budget import apply_generation_budget


def test_generation_budget_preserves_decoder_window_with_prompt():
    model = SimpleNamespace(config=SimpleNamespace(max_target_positions=448))
    prompt_ids = torch.arange(180).reshape(1, 180)
    kwargs = {
        "max_new_tokens": 444,
        "forced_decoder_ids": [(1, 50265), (2, 50359)],
    }

    effective_kwargs, effective_prompt, budget = apply_generation_budget(
        model=model,
        generate_kwargs=kwargs,
        prompt_ids=prompt_ids,
        min_effective_new_tokens=64,
    )

    assert budget.model_max_target_positions == 448
    assert budget.decoder_prefix_tokens == 3
    assert budget.effective_max_new_tokens == 265
    assert budget.prompt_tokens_kept == 180
    assert budget.clipped_max_new_tokens is True
    assert budget.clipped_prompt_tokens is False
    assert int(effective_prompt.numel()) == 180
    assert (
        budget.decoder_prefix_tokens
        + budget.prompt_tokens_kept
        + effective_kwargs["max_new_tokens"]
        == 448
    )


def test_generation_budget_clips_prompt_before_min_generation_space():
    model = SimpleNamespace(config=SimpleNamespace(max_target_positions=448))
    prompt_ids = torch.arange(500).reshape(1, 500)

    effective_kwargs, effective_prompt, budget = apply_generation_budget(
        model=model,
        generate_kwargs={"max_new_tokens": 444},
        prompt_ids=prompt_ids,
        min_effective_new_tokens=64,
    )

    assert budget.prompt_tokens_original == 500
    assert budget.prompt_tokens_kept == 383
    assert budget.prompt_tokens_dropped == 117
    assert budget.clipped_prompt_tokens is True
    assert effective_kwargs["max_new_tokens"] == 64
    assert torch.equal(effective_prompt, prompt_ids[..., -383:])


def test_generation_budget_without_model_limit_keeps_configured_values():
    model = SimpleNamespace(config=SimpleNamespace(max_target_positions=None))
    prompt_ids = torch.arange(12).reshape(1, 12)

    effective_kwargs, effective_prompt, budget = apply_generation_budget(
        model=model,
        generate_kwargs={"max_new_tokens": 128},
        prompt_ids=prompt_ids,
        min_effective_new_tokens=64,
    )

    assert budget.model_max_target_positions is None
    assert budget.effective_max_new_tokens == 128
    assert budget.prompt_tokens_kept == 12
    assert effective_kwargs["max_new_tokens"] == 128
    assert effective_kwargs["prompt_ids"] is prompt_ids
    assert effective_prompt is prompt_ids
