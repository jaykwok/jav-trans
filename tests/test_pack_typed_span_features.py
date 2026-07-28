"""Packing must apply the requested PTM basis exactly, or refuse.

Three bases share one packer: leading-N truncation, a learned 2048->N linear
projection, and full width. They are not interchangeable - the leading 128 dims
carry only ~6.7% of total PTM energy - so a basis silently applied to the wrong
input width would quietly change what every downstream arm is measuring while
still producing a plausible-looking memmap.

The projection convention is fixed by `sequence_features._projected_ptm_features`
as `(x - mean) @ components.T`; these tests pin it numerically rather than
trusting the two implementations to stay in step.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.datasets.pack_typed_span_features import pack  # noqa: E402

PTM_PROJECTION_SCHEMA = "speech_boundary_ja_ptm_projection_v1"


def _write_example(
    tmp_path: Path, *, frames: int, ptm_dim: int, mfcc_dim: int = 40, seed: int = 0
) -> tuple[Path, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ptm = rng.normal(scale=0.02, size=(frames, ptm_dim)).astype(np.float16)
    mfcc = rng.normal(size=(frames, mfcc_dim)).astype(np.float16)
    path = tmp_path / f"feat_{seed}.npz"
    np.savez(path, ptm=ptm, mfcc=mfcc)
    return path, ptm, mfcc


def _write_index(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "index.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def _write_projection(
    tmp_path: Path, *, input_dim: int, output_dim: int, seed: int = 7
) -> tuple[Path, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mean = rng.normal(scale=0.01, size=input_dim).astype(np.float32)
    components = rng.normal(
        scale=1.0 / np.sqrt(input_dim), size=(output_dim, input_dim)
    ).astype(np.float32)
    path = tmp_path / "projection.npz"
    np.savez(
        path,
        schema=np.array(PTM_PROJECTION_SCHEMA),
        mean=mean,
        components=components,
    )
    return path, mean, components


def _load_output(output: Path, summary: dict) -> np.ndarray:
    return np.memmap(
        output / "features.f16",
        dtype=np.float16,
        mode="r",
        shape=(summary["total_frames"], summary["width"]),
    )


def test_truncation_takes_the_leading_dims(tmp_path: Path) -> None:
    feature, ptm, mfcc = _write_example(tmp_path, frames=32, ptm_dim=2048)
    index = _write_index(
        tmp_path,
        [{"feature_path": str(feature), "frame_offset": 0, "frame_count": 32}],
    )
    output = tmp_path / "out"

    summary = pack(
        index_path=index, output=output, ptm_dim=128, mfcc_dim=40, workers=1
    )

    assert summary["basis"] == "truncation"
    assert summary["width"] == 168
    packed = _load_output(output, summary)
    assert np.array_equal(packed[:, :128], ptm[:, :128])
    assert np.array_equal(packed[:, 128:], mfcc)


def test_full_width_keeps_every_dim(tmp_path: Path) -> None:
    feature, ptm, _mfcc = _write_example(tmp_path, frames=16, ptm_dim=2048)
    index = _write_index(
        tmp_path,
        [{"feature_path": str(feature), "frame_offset": 0, "frame_count": 16}],
    )
    output = tmp_path / "out"

    summary = pack(
        index_path=index, output=output, ptm_dim=2048, mfcc_dim=40, workers=1
    )

    assert summary["width"] == 2088
    packed = _load_output(output, summary)
    assert np.array_equal(packed[:, :2048], ptm)


def test_projection_matches_the_canonical_formula(tmp_path: Path) -> None:
    feature, ptm, mfcc = _write_example(tmp_path, frames=24, ptm_dim=2048)
    projection, mean, components = _write_projection(
        tmp_path, input_dim=2048, output_dim=128
    )
    index = _write_index(
        tmp_path,
        [{"feature_path": str(feature), "frame_offset": 0, "frame_count": 24}],
    )
    output = tmp_path / "out"

    summary = pack(
        index_path=index,
        output=output,
        ptm_dim=999,  # must be overridden by the projection's output width
        mfcc_dim=40,
        workers=1,
        projection_path=str(projection),
    )

    assert summary["basis"] == "learned_projection"
    assert summary["ptm_dim"] == 128
    assert summary["width"] == 168
    assert summary["projection_input_dim"] == 2048
    assert summary["projection_digest"]

    expected = (ptm.astype(np.float32) - mean.reshape(1, -1)) @ components.T
    packed = _load_output(output, summary)
    np.testing.assert_allclose(
        np.asarray(packed[:, :128], dtype=np.float32), expected, atol=2e-2
    )
    assert np.array_equal(packed[:, 128:], mfcc)


def test_projection_refuses_a_truncated_input(tmp_path: Path) -> None:
    """The 128-dim cache cannot feed a 2048-dim projection.

    This is the mistake the full re-extraction exists to prevent; silently
    accepting it would project garbage and still emit a well-formed memmap.
    """
    feature, _ptm, _mfcc = _write_example(tmp_path, frames=16, ptm_dim=128)
    projection, _mean, _components = _write_projection(
        tmp_path, input_dim=2048, output_dim=128
    )
    index = _write_index(
        tmp_path,
        [{"feature_path": str(feature), "frame_offset": 0, "frame_count": 16}],
    )
    output = tmp_path / "out"

    summary = pack(
        index_path=index,
        output=output,
        ptm_dim=128,
        mfcc_dim=40,
        workers=1,
        projection_path=str(projection),
    )

    assert summary["packed"] == 0
    assert summary["ptm_too_narrow"] == 1


def test_truncation_refuses_a_too_narrow_input(tmp_path: Path) -> None:
    feature, _ptm, _mfcc = _write_example(tmp_path, frames=16, ptm_dim=64)
    index = _write_index(
        tmp_path,
        [{"feature_path": str(feature), "frame_offset": 0, "frame_count": 16}],
    )
    output = tmp_path / "out"

    summary = pack(
        index_path=index, output=output, ptm_dim=128, mfcc_dim=40, workers=1
    )

    assert summary["packed"] == 0
    assert summary["ptm_too_narrow"] == 1


def test_examples_land_at_their_declared_offsets(tmp_path: Path) -> None:
    first, ptm_a, _ = _write_example(tmp_path, frames=10, ptm_dim=256, seed=1)
    second, ptm_b, _ = _write_example(tmp_path, frames=7, ptm_dim=256, seed=2)
    index = _write_index(
        tmp_path,
        [
            {"feature_path": str(first), "frame_offset": 0, "frame_count": 10},
            {"feature_path": str(second), "frame_offset": 10, "frame_count": 7},
        ],
    )
    output = tmp_path / "out"

    summary = pack(
        index_path=index, output=output, ptm_dim=256, mfcc_dim=40, workers=2
    )

    assert summary["total_frames"] == 17
    packed = _load_output(output, summary)
    assert np.array_equal(packed[0:10, :256], ptm_a)
    assert np.array_equal(packed[10:17, :256], ptm_b)


def test_short_example_is_recorded_not_truncated(tmp_path: Path) -> None:
    feature, _ptm, _mfcc = _write_example(tmp_path, frames=8, ptm_dim=256)
    index = _write_index(
        tmp_path,
        [{"feature_path": str(feature), "frame_offset": 0, "frame_count": 40}],
    )
    output = tmp_path / "out"

    summary = pack(
        index_path=index, output=output, ptm_dim=256, mfcc_dim=40, workers=1
    )

    assert summary["packed"] == 0
    assert summary["too_few_frames"] == 1
