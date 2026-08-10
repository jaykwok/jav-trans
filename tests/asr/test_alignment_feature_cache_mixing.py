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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.align.train_ctc_aligner import (  # noqa: E402
    FeatureCache,
    _cap_empty_train_rows,
    _collate,
)
from tools.align.build_alignment_features import (  # noqa: E402
    _assign_partitions,
    _is_held_out,
)

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


def test_empty_target_cap_is_per_domain_and_deterministic() -> None:
    rows = [
        {"audio_id": f"positive-{index}", "domain": "real", "text": "あ"}
        for index in range(7)
    ] + [
        {"audio_id": f"blank-{index}", "domain": "real", "text": ""}
        for index in range(20)
    ] + [
        {"audio_id": "gal-positive", "domain": "galgame", "text": "あ"}
    ]

    first, report = _cap_empty_train_rows(rows, maximum_fraction=0.30, seed=5)
    second, _ = _cap_empty_train_rows(rows, maximum_fraction=0.30, seed=5)

    assert [row["audio_id"] for row in first] == [row["audio_id"] for row in second]
    assert report["real"]["empty_kept"] == 3
    assert report["real"]["empty_fraction"] == pytest.approx(0.3)
    assert report["galgame"]["empty_kept"] == 0


def test_collate_accepts_a_batch_whose_targets_are_all_empty() -> None:
    torch = pytest.importorskip("torch")
    batch = [
        (np.ones((3, FEATURE_DIM), dtype=np.float32), []),
        (np.ones((2, FEATURE_DIM), dtype=np.float32), []),
    ]

    _, targets, _, target_lengths = _collate(batch, torch)

    assert targets.numel() == 0
    assert target_lengths.tolist() == [0, 0]


def test_feature_builder_preserves_fixed_group_partitions() -> None:
    rows = [
        {"audio_id": "a", "group": "game-a", "partition": "train"},
        {"audio_id": "b", "group": "game-a", "partition": "train"},
        {"audio_id": "c", "group": "game-b", "partition": "val"},
    ]

    report = _assign_partitions(
        rows, val_fraction=0.5, rng=np.random.default_rng(123)
    )

    assert [row["partition"] for row in rows] == ["train", "train", "val"]
    assert report["fixed_manifest_partitions"] is True
    assert report["groups_val"] == 1


def test_feature_builder_rejects_a_group_crossing_fixed_partitions() -> None:
    rows = [
        {"audio_id": "a", "group": "game-a", "partition": "train"},
        {"audio_id": "b", "group": "game-a", "partition": "val"},
    ]

    with pytest.raises(SystemExit, match="crosses train/val"):
        _assign_partitions(rows, val_fraction=0.5, rng=np.random.default_rng(123))


def test_feature_builder_holdout_matches_teacher_source_identity() -> None:
    held_out = {"original-galgame-id"}

    assert _is_held_out(
        {
            "audio_id": "original-galgame-id-full",
            "source_id": "original-galgame-id",
        },
        held_out,
    )
    assert not _is_held_out(
        {"audio_id": "other-full", "source_id": "other"}, held_out
    )


class TestOneFeasibilityJudgment:
    """Extraction, inference and the loss must agree on "the text fits".

    They did not. Three sites carried three different bounds - the extractor
    filtered on `frames < len(text)`, inference refused `frames < len(targets)+1`,
    and `zero_infinity=True` silently zeroed whatever reached the loss anyway.
    None of them counted the mandatory blank between adjacent identical
    characters, which Japanese produces constantly (`ああ`, `っっ`, `ーー`).

    The consequence was ordered badly: the *looser* check ran first, so a clip
    could pass extraction, train against a silently-zeroed loss, and then be
    refused at inference by the stricter one.
    """

    def test_the_extractor_uses_the_same_bound_as_inference(self) -> None:
        source = (
            PROJECT_ROOT / "tools" / "align" / "build_alignment_features.py"
        ).read_text(encoding="utf-8")
        assert "minimum_ctc_frames" in source
        # The bound that ignored repeated characters, in either of its spellings.
        assert "frames < len(row[" not in source
        assert "len(text) > duration" not in source

    def test_the_trainer_counts_what_zero_infinity_hides(self) -> None:
        """`zero_infinity` turns an infeasible row into a zero loss, not an
        error, so it dilutes every reported loss with no other symptom."""
        source = (
            PROJECT_ROOT / "tools" / "align" / "train_ctc_aligner.py"
        ).read_text(encoding="utf-8")
        assert "minimum_ctc_frames" in source
        assert "infeasible_rows" in source
        # Counted at the length CTC actually sees, not the cached frame count.
        assert 'int(row["frames"]) * args.upsample' in source

    def test_the_extractor_bound_rejects_a_clip_inference_would_refuse(self) -> None:
        """The property behind the source assertions: no clip may pass
        extraction and then be refused by `forced_align`."""
        from asr.alignment import minimum_ctc_frames

        # あああ: three frames of text, but five frames of CTC path.
        assert minimum_ctc_frames("あああ") == 5
        assert minimum_ctc_frames("あああ") > len("あああ")


class TestHeldOutGrouping:
    """Validation has to be a set of speakers, not a set of clips.

    The split was a per-clip random draw over a corpus of visual novel voice
    lines, so the same actor reading the same script style sat on both sides and
    the val loss reported memorisation as generalisation. Every architecture
    comparison that number decides is decided on a contaminated metric - which
    is why this had to be fixed before, not after, any A/B.
    """

    @staticmethod
    def _extractor():
        from tools.align import build_alignment_features

        return build_alignment_features

    def test_an_explicit_group_field_wins_when_the_manifest_has_one(self) -> None:
        build = self._extractor()
        key = build._group_key({"speaker": "aoi"}, 7, field="speaker", block=200)
        assert key == "speaker=aoi"

    def test_a_missing_group_field_is_refused_rather_than_ignored(self) -> None:
        """Falling back would produce a leaky split under a name that claims
        otherwise, and nothing downstream could tell."""
        build = self._extractor()
        with pytest.raises(SystemExit, match="group-field"):
            build._group_key({"audio_id": "x"}, 0, field="speaker", block=200)

    def test_without_a_field_consecutive_manifest_rows_group_together(self) -> None:
        """The galgame manifest keeps only ids, text and upstream order, and
        source order groups by game - the same property the extractor already
        relies on when it refuses to subsample by truncation."""
        build = self._extractor()
        keys = [
            build._group_key({"index": i}, i, field="", block=100) for i in range(250)
        ]
        assert keys[0] == keys[99] != keys[100]
        assert len(set(keys)) == 3

    def test_a_group_never_appears_on_both_sides_of_the_split(self) -> None:
        build = self._extractor()
        rows = [
            {"audio_id": str(i), "group": f"block={i // 50}"} for i in range(1000)
        ]
        build._assign_partitions(
            rows, val_fraction=0.1, rng=np.random.default_rng(0)
        )
        sides: dict[str, set[str]] = {}
        for row in rows:
            sides.setdefault(row["group"], set()).add(row["partition"])
        assert all(len(side) == 1 for side in sides.values())

    def test_the_val_share_lands_near_the_requested_fraction(self) -> None:
        """Whole groups overshoot; the split is not allowed to drift far."""
        build = self._extractor()
        rows = [{"audio_id": str(i), "group": f"block={i // 20}"} for i in range(2000)]
        report = build._assign_partitions(
            rows, val_fraction=0.05, rng=np.random.default_rng(1)
        )
        assert 0.05 <= report["val_fraction_actual"] <= 0.06
        assert report["groups_total"] == 100
        assert report["groups_val"] == 5

    def test_block_zero_restores_the_old_per_clip_behaviour(self) -> None:
        """Kept deliberately: the first cache was built this way and a rerun
        that wants to reproduce it must be able to say so out loud."""
        build = self._extractor()
        keys = [build._group_key({"index": i}, i, field="", block=0) for i in range(5)]
        assert len(set(keys)) == 5

    def test_the_cache_row_carries_its_group(self) -> None:
        """Without this the split cannot be audited or redrawn later - which is
        exactly the state the 30k-clip cache is in."""
        source = (
            PROJECT_ROOT / "tools" / "align" / "build_alignment_features.py"
        ).read_text(encoding="utf-8")
        assert '"group": row["group"]' in source
