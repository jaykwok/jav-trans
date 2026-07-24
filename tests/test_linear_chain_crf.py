from __future__ import annotations

import itertools

import numpy as np
import pytest

from boundary.linear_chain_crf import BinaryLinearChainCrf


def _enumerated_scores(emissions, transitions):
    torch = pytest.importorskip("torch")
    sequences = list(itertools.product(range(2), repeat=int(emissions.shape[0])))
    scores = []
    for sequence in sequences:
        score = emissions[0, sequence[0]]
        for frame in range(1, len(sequence)):
            score = (
                score
                + transitions[sequence[frame - 1], sequence[frame]]
                + emissions[frame, sequence[frame]]
            )
        scores.append(score)
    return sequences, torch.stack(scores)


def test_binary_crf_matches_bruteforce_nll_viterbi_and_marginals() -> None:
    torch = pytest.importorskip("torch")
    crf = BinaryLinearChainCrf()
    with torch.no_grad():
        crf.transitions.copy_(torch.tensor([[0.3, -0.4], [-0.2, 0.5]]))
    emissions = torch.tensor(
        [[[0.2, -0.1], [-0.3, 0.6], [0.7, -0.2]]], dtype=torch.float32
    )
    tags = torch.tensor([[0, 1, 0]], dtype=torch.long)
    mask = torch.ones((1, 3), dtype=torch.bool)
    sequences, scores = _enumerated_scores(emissions[0], crf.transitions)
    expected_nll = (
        torch.logsumexp(scores, dim=0)
        - scores[sequences.index(tuple(tags[0].tolist()))]
    ) / 3.0

    actual_nll = crf.neg_log_likelihood(emissions, tags, mask)
    decoded = crf.decode(emissions, mask)
    marginal = crf.marginal_probabilities(emissions, mask)
    weights = torch.softmax(scores, dim=0)
    expected_marginal = torch.zeros((3, 2), dtype=torch.float32)
    for weight, sequence in zip(weights, sequences, strict=True):
        for frame, label in enumerate(sequence):
            expected_marginal[frame, label] += weight

    assert actual_nll.item() == pytest.approx(expected_nll.item(), abs=1e-6)
    assert decoded[0].tolist() == list(sequences[int(torch.argmax(scores).item())])
    torch.testing.assert_close(marginal[0], expected_marginal, atol=1e-6, rtol=1e-6)


def test_binary_crf_unsure_and_non_owner_frames_do_not_enter_score_or_normalization() -> None:
    torch = pytest.importorskip("torch")
    crf = BinaryLinearChainCrf()
    emissions = torch.tensor(
        [[[0.2, 0.1], [0.3, -0.2], [9.0, -9.0], [-0.1, 0.4], [0.6, -0.3]]],
        dtype=torch.float32,
    )
    tags = torch.tensor([[0, 0, -100, 1, 0]], dtype=torch.long)
    supervision = torch.tensor([[True, True, False, True, True]])
    reference = crf.neg_log_likelihood(emissions, tags, supervision)

    changed_emissions = emissions.clone()
    changed_emissions[0, 2] = torch.tensor([-1000.0, 1000.0])
    changed_tags = tags.clone()
    changed_tags[0, 2] = 777
    changed = crf.neg_log_likelihood(
        changed_emissions, changed_tags, supervision
    )

    assert changed.item() == pytest.approx(reference.item(), abs=1e-7)


def test_binary_crf_runtime_mask_requires_a_prefix() -> None:
    torch = pytest.importorskip("torch")
    crf = BinaryLinearChainCrf()
    emissions = torch.zeros((1, 3, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="valid prefix"):
        crf.decode(emissions, torch.tensor([[True, False, True]]))
    with pytest.raises(ValueError, match="valid prefix"):
        crf.marginal_probabilities(
            emissions, torch.tensor([[True, False, True]])
        )
