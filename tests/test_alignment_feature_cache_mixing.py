"""Mixing two feature caches (galgame + real JAV) for alignment training.

The 2026-07-31 pre-gate falsification traced back to a head trained on one domain
only, so the retrain mixes two caches. That makes shard names ambiguous: every
cache directory independently starts at `features_0000.npy`, and resolving a row
against the wrong directory would pair one domain's audio features with the
other domain's text. Nothing would crash - CTC would just fail to converge on
half the corpus and look like a hyperparameter problem.

These tests are cheap because the cache format is plain `.npy` + JSONL, so the
mixing logic can be checked without an encoder, a GPU, or torch.
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

from tools.align.train_ctc_aligner import FeatureCache  # noqa: E402

FEATURE_DIM = 4


def _make_cache(root: Path, name: str, *, fill: float, rows: list[dict]) -> Path:
    cache_dir = root / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    total = sum(int(row["frames"]) for row in rows)
    shard = np.full((total, FEATURE_DIM), fill, dtype=np.float16)
    np.save(cache_dir / "features_0000.npy", shard)
    with (cache_dir / "index.jsonl").open("w", encoding="utf-8") as handle:
        offset = 0
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "audio_id": row["audio_id"],
                        "shard": "features_0000.npy",
                        "offset": offset,
                        "frames": int(row["frames"]),
                        "text": row.get("text", "あ"),
                        "duration_s": row.get("duration_s", 1.0),
                        "partition": row.get("partition", "train"),
                    }
                )
                + "\n"
            )
            offset += int(row["frames"])
    return cache_dir


@pytest.fixture()
def caches(tmp_path: Path) -> tuple[Path, Path]:
    galgame = _make_cache(
        tmp_path,
        "galgame",
        fill=1.0,
        rows=[
            {"audio_id": "g0", "frames": 3, "duration_s": 2.0},
            {"audio_id": "g1", "frames": 2, "duration_s": 1.5, "partition": "val"},
        ],
    )
    real = _make_cache(
        tmp_path,
        "real",
        fill=7.0,
        rows=[{"audio_id": "r0", "frames": 4, "duration_s": 3.0}],
    )
    return galgame, real


def test_identically_named_shards_resolve_to_their_own_cache(caches):
    galgame, real = caches
    cache = FeatureCache([galgame, real])

    by_id = {row["audio_id"]: row for row in cache.rows}
    # Both caches wrote `features_0000.npy`; the fill value is the tell.
    assert np.all(cache.features(by_id["g0"]) == 1.0)
    assert np.all(cache.features(by_id["r0"]) == 7.0)
    assert cache.features(by_id["r0"]).shape == (4, FEATURE_DIM)


def test_rows_carry_the_domain_they_came_from(caches):
    galgame, real = caches
    cache = FeatureCache([galgame, real])

    assert {row["domain"] for row in cache.rows} == {"galgame", "real"}
    assert cache.domain_rows == {"galgame": 2, "real": 1}
    assert cache.domain_seconds["galgame"] == pytest.approx(3.5)
    assert cache.domain_seconds["real"] == pytest.approx(3.0)


def test_repeat_oversamples_train_rows_only(caches):
    galgame, real = caches
    cache = FeatureCache([galgame, real], [1, 3])

    ids = [row["audio_id"] for row in cache.rows]
    assert ids.count("r0") == 3
    assert ids.count("g0") == 1
    assert ids.count("g1") == 1

    # Oversampling a val row would weight the domains differently in train and
    # val, which makes the two numbers incomparable rather than merely noisy.
    repeated_val = FeatureCache([galgame], [5])
    val_ids = [row["audio_id"] for row in repeated_val.rows if row["audio_id"] == "g1"]
    assert len(val_ids) == 1

    # The reported totals count distinct clips, not copies, so a repeat does not
    # inflate "hours of real audio trained on".
    assert cache.domain_rows["real"] == 1


def test_repeat_count_must_line_up_with_cache_dirs(caches):
    galgame, real = caches
    with pytest.raises(SystemExit):
        FeatureCache([galgame, real], [2])


def test_explicit_domain_labels_override_the_directory_name(caches):
    galgame, real = caches
    cache = FeatureCache([galgame, real], None, ["galgame_clean", "real_jav"])

    assert cache.domain_rows == {"galgame_clean": 2, "real_jav": 1}
    assert {row["domain"] for row in cache.rows} == {"galgame_clean", "real_jav"}

    with pytest.raises(SystemExit):
        FeatureCache([galgame, real], None, ["only_one"])


def test_single_cache_still_works_without_repeats(caches):
    galgame, _ = caches
    cache = FeatureCache([galgame])
    assert len(cache.rows) == 2
    assert all(row["cache_index"] == 0 for row in cache.rows)
