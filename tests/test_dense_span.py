from __future__ import annotations

import pytest

from boundary.dense_span import BinaryDenseSpanDecoder


def _decoder(*, context: int = 12):
    return BinaryDenseSpanDecoder(
        hidden_size=4,
        rank=2,
        duration_hidden_size=4,
        context_window_frames=context,
    )


def test_dense_span_zero_gate_is_exact_frame_argmax_without_length_filter() -> None:
    torch = pytest.importorskip("torch")
    decoder = _decoder(context=12)
    encoded = torch.randn((1, 9, 4), generator=torch.Generator().manual_seed(41))
    frame_logits = torch.tensor(
        [
            [
                [3.0, -1.0],
                [2.0, -0.5],
                [-1.0, 2.0],
                [-0.5, 3.0],
                [2.5, -1.0],
                [-1.0, 2.5],
                [-0.5, 2.0],
                [-1.0, 3.0],
                [2.0, -1.0],
            ]
        ]
    )
    mask = torch.ones((1, 9), dtype=torch.bool)

    scores = decoder.span_scores(encoded, frame_logits, mask)
    decoded = decoder.decode(scores, mask)

    assert scores.shape == (1, 2, 9, 9)
    torch.testing.assert_close(
        scores[0, :, 0, 8], frame_logits[0].sum(dim=0), rtol=0.0, atol=0.0
    )
    assert torch.isfinite(scores[0, :, 0, 8]).all()
    assert torch.isneginf(scores[0, :, 8, 0]).all()
    torch.testing.assert_close(
        torch.tanh(decoder.span_residual_gate), torch.zeros(())
    )
    torch.testing.assert_close(decoded, torch.argmax(frame_logits, dim=-1))


def test_dense_span_exact_dp_can_select_one_complete_learned_span() -> None:
    torch = pytest.importorskip("torch")
    decoder = _decoder(context=5)
    frame_logits = torch.tensor(
        [[[0.0, 3.0], [0.0, 3.0], [3.0, 0.0], [0.0, 3.0], [0.0, 3.0]]]
    )
    encoded = torch.zeros((1, 5, 4))
    mask = torch.ones((1, 5), dtype=torch.bool)
    scores = decoder.span_scores(encoded, frame_logits, mask)
    assert decoder.decode(scores, mask).tolist() == [[1, 1, 0, 1, 1]]

    learned_scores = scores.clone()
    learned_scores[0, 1, 0, 4] = 20.0
    assert decoder.decode(learned_scores, mask).tolist() == [[1, 1, 1, 1, 1]]


def test_dense_span_viterbi_matches_exhaustive_binary_paths() -> None:
    torch = pytest.importorskip("torch")
    decoder = _decoder(context=5)
    generator = torch.Generator().manual_seed(2407)
    scores = torch.full((2, 5, 5), -torch.inf)
    for start in range(5):
        scores[:, start, start:] = torch.randn(
            (2, 5 - start), generator=generator
        )
    with torch.no_grad():
        decoder.start_scores.copy_(torch.tensor([0.17, -0.11]))
        decoder.end_scores.copy_(torch.tensor([-0.07, 0.13]))
        decoder.transitions.copy_(torch.tensor([[0.0, -0.23], [0.19, 0.0]]))

    decoded = decoder._viterbi_path(scores)
    decoded_score = decoder._path_score(scores, decoded)
    truth = torch.tensor([0, 1, 1, 0, 1])
    loss_augmented = decoder._viterbi_path(scores, labels=truth)
    loss_augmented_score = (
        decoder._path_score(scores, loss_augmented)
        + float(loss_augmented.hamming_count)
    )

    exhaustive_scores = []
    exhaustive_augmented_scores = []
    for encoded_labels in range(1 << 5):
        labels = torch.tensor(
            [(encoded_labels >> frame) & 1 for frame in range(5)]
        )
        path = decoder._gold_path(labels)
        path_score = decoder._path_score(scores, path)
        exhaustive_scores.append(path_score)
        exhaustive_augmented_scores.append(
            path_score + torch.count_nonzero(labels != truth)
        )

    torch.testing.assert_close(decoded_score, torch.stack(exhaustive_scores).max())
    torch.testing.assert_close(
        loss_augmented_score,
        torch.stack(exhaustive_augmented_scores).max(),
    )


def test_dense_span_structured_loss_excludes_unsure_and_non_owner() -> None:
    torch = pytest.importorskip("torch")
    decoder = _decoder(context=6)
    scores = torch.zeros((1, 2, 6, 6), requires_grad=True)
    labels = torch.tensor([[1, -100, 1, 0, 1, 1]], dtype=torch.long)
    owner = torch.tensor([[True, True, True, True, True, False]])
    supervision = owner & (labels != -100)

    loss = decoder.structured_hinge(scores, labels, supervision)
    loss.backward()

    assert torch.isfinite(loss)
    assert float(loss.detach()) > 0.0
    assert scores.grad is not None
    assert torch.count_nonzero(scores.grad[:, :, 1, :]) == 0
    assert torch.count_nonzero(scores.grad[:, :, :, 1]) == 0
    assert torch.count_nonzero(scores.grad[:, :, 5, :]) == 0
    assert torch.count_nonzero(scores.grad[:, :, :, 5]) == 0
    # A cross-gap span must never enter either loss-augmented or gold scoring.
    assert torch.count_nonzero(scores.grad[:, :, 0, 4]) == 0


def test_dense_span_batch_padding_does_not_change_valid_lattice_or_decode() -> None:
    torch = pytest.importorskip("torch")
    decoder = _decoder(context=9).eval()
    generator = torch.Generator().manual_seed(74)
    encoded = torch.randn((2, 9, 4), generator=generator)
    frame_logits = torch.randn((2, 9, 2), generator=generator)
    mask = torch.tensor(
        [
            [True, True, True, True, True, False, False, False, False],
            [True, True, True, True, True, True, True, True, True],
        ]
    )

    batched_scores = decoder.span_scores(encoded, frame_logits, mask)
    batched_labels = decoder.decode(batched_scores, mask)
    singleton_scores = decoder.span_scores(
        encoded[:1, :5], frame_logits[:1, :5], mask[:1, :5]
    )
    singleton_labels = decoder.decode(singleton_scores, mask[:1, :5])

    torch.testing.assert_close(
        batched_scores[0, :, :5, :5], singleton_scores[0], rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(batched_labels[0, :5], singleton_labels[0])
