"""The two feature-basis mechanisms must be exactly what they claim.

`FrameStore` gained a second access mode because the widest basis could not be
memory-mapped on a 15.9 GiB host without driving machine-wide available RAM to
zero. A loader that returned subtly different arrays would silently change every
metric, so the two modes are pinned to bit-equality.

The learnable PTM projector is initialised as an identity slice specifically so
that "learned projection vs leading-N truncation" is a single-variable
comparison: at step 0 the two must be indistinguishable, and any later
difference is attributable to training rather than to a different starting
point. That is the protocol the 07-06 island ablation used, and it is only
sound if the initialisation really is an identity slice.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

torch = pytest.importorskip("torch")

from tools.boundary.ja.train_typed_span_falsification import (  # noqa: E402
    FrameStore,
    build_model,
    compute_normalization,
)


def _make_store(tmp_path: Path, *, frames: int, ptm_dim: int, mfcc_dim: int = 40):
    width = ptm_dim + mfcc_dim
    rng = np.random.default_rng(3)
    data = rng.normal(scale=0.02, size=(frames, width)).astype(np.float16)
    data.tofile(tmp_path / "features.f16")
    (tmp_path / "features_summary.json").write_text(
        json.dumps(
            {
                "total_frames": frames,
                "width": width,
                "ptm_dim": ptm_dim,
                "mfcc_dim": mfcc_dim,
            }
        ),
        encoding="utf-8",
    )
    np.savez(
        tmp_path / "frame_labels.npz",
        speech=np.zeros(frames, dtype=np.int8),
        type=np.zeros(frames, dtype=np.int8),
    )
    with (tmp_path / "index.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "example_id": "e0",
                    "partition": "train",
                    "provenance": "real_omni_joint",
                    "frame_offset": 0,
                    "frame_count": frames,
                }
            )
            + "\n"
        )
    return data


def test_pread_and_mmap_return_identical_frames(tmp_path: Path) -> None:
    data = _make_store(tmp_path, frames=200, ptm_dim=64)
    mapped = FrameStore(tmp_path, loader="mmap")
    streamed = FrameStore(tmp_path, loader="pread")
    try:
        for start, count in ((0, 200), (0, 7), (37, 61), (199, 1)):
            np.testing.assert_array_equal(
                mapped.read(start, count), streamed.read(start, count)
            )
            np.testing.assert_array_equal(
                streamed.read(start, count),
                np.asarray(data[start : start + count], dtype=np.float32),
            )
    finally:
        mapped.close()
        streamed.close()


def test_empty_read_is_well_shaped(tmp_path: Path) -> None:
    _make_store(tmp_path, frames=32, ptm_dim=64)
    store = FrameStore(tmp_path, loader="pread")
    try:
        assert store.read(0, 0).shape == (0, store.width)
    finally:
        store.close()


def test_auto_loader_picks_mmap_for_a_small_file(tmp_path: Path) -> None:
    _make_store(tmp_path, frames=64, ptm_dim=64)
    store = FrameStore(tmp_path, loader="auto")
    try:
        assert store.loader == "mmap"
    finally:
        store.close()


def test_store_exposes_the_ptm_split(tmp_path: Path) -> None:
    _make_store(tmp_path, frames=16, ptm_dim=2048, mfcc_dim=40)
    store = FrameStore(tmp_path, loader="mmap")
    try:
        assert (store.ptm_dim, store.mfcc_dim, store.width) == (2048, 40, 2088)
    finally:
        store.close()


def test_projector_init_reproduces_truncation_exactly() -> None:
    """Step 0 of a learned 2048->N projector must equal raw-N truncation.

    Seeding alone will not line the two networks up: constructing the projector
    consumes RNG, so the shared layers would draw different weights. They are
    copied across instead, leaving the projector as the only difference.
    """
    projected = build_model(2088, 32, (1, 2), ptm_in=2048, projector_dim=128)
    truncated = build_model(168, 32, (1, 2), ptm_in=0, projector_dim=0)
    shared = {
        key: value
        for key, value in projected.state_dict().items()
        if not key.startswith("ptm_projector")
    }
    truncated.load_state_dict(shared)

    x_full = torch.randn(2, 40, 2088)
    x_trunc = torch.cat((x_full[..., :128], x_full[..., 2048:]), dim=-1)

    with torch.no_grad():
        a = projected(x_full)
        b = truncated(x_trunc)

    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dim", [128, 256, 512])
def test_projector_passes_mfcc_through_untouched(dim: int) -> None:
    model = build_model(2088, 32, (1,), ptm_in=2048, projector_dim=dim)
    x = torch.randn(1, 5, 2088)

    with torch.no_grad():
        head = model.ptm_projector(x[..., :2048])

    assert head.shape == (1, 5, dim)
    # The identity slice means the leading `dim` PTM columns survive verbatim.
    torch.testing.assert_close(head, x[..., :dim], rtol=1e-5, atol=1e-5)


def test_projector_is_trainable_and_moves_off_the_identity() -> None:
    model = build_model(2088, 32, (1,), ptm_in=2048, projector_dim=64)
    before = model.ptm_projector.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    x = torch.randn(2, 16, 2088)
    target = torch.randint(0, 2, (2, 16))
    loss = torch.nn.functional.cross_entropy(
        model(x).reshape(-1, 2), target.reshape(-1)
    )
    optimizer.zero_grad()
    loss.backward()
    assert model.ptm_projector.weight.grad is not None
    optimizer.step()

    assert not torch.equal(before, model.ptm_projector.weight.detach())


def test_no_projector_leaves_the_stem_width_alone() -> None:
    model = build_model(168, 32, (1,), ptm_in=0, projector_dim=0)
    assert model.ptm_projector is None
    assert model.stem.in_channels == 168


def test_streaming_normalization_matches_the_stacked_result() -> None:
    """Running sums must equal the naive mean/std they replaced.

    The stacked version held every sampled frame at once; at 2088 dims that is
    ~2.8 GB plus a transient copy from `np.concatenate`, which tripped the host
    memory cap on the wide-projector runs. The replacement is only safe if it is
    numerically the same statistic.
    """
    rng = np.random.default_rng(0)
    data = rng.normal(
        loc=[1.0, -2.0, 5.0], scale=[0.5, 2.0, 0.1], size=(4000, 3)
    ).astype(np.float32)

    class Store:
        width = 3

        def read(self, start: int, count: int) -> np.ndarray:
            return data[start : start + count].astype(np.float32)

    examples = [{"frame_offset": i * 400, "frame_count": 400} for i in range(10)]
    mean, std = compute_normalization(
        Store(), examples, sample_frames=10**9, seed=1
    )

    np.testing.assert_allclose(mean, data.mean(axis=0), atol=1e-4)
    np.testing.assert_allclose(std, data.std(axis=0), atol=1e-4)


def test_normalization_floors_a_degenerate_dimension() -> None:
    """A constant column must not divide the batch by ~0."""
    constant = np.zeros((256, 2), dtype=np.float32)
    constant[:, 1] = np.arange(256, dtype=np.float32)

    class Store:
        width = 2

        def read(self, start: int, count: int) -> np.ndarray:
            return constant[start : start + count]

    _mean, std = compute_normalization(
        Store(), [{"frame_offset": 0, "frame_count": 256}], sample_frames=256, seed=0
    )

    assert std[0] == 1.0
    assert std[1] > 1.0
