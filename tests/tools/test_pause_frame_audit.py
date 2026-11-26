"""Contract tests for the real-domain safe-cut frame audit.

This audit exists because the two candidate explanations for the 07-31 pre-gate
falsification - "one posterior cannot serve two tasks" versus "the head is
mistrained for this domain" - predict the same observation, and nothing in the
repo could tell them apart. What separates them is the frame-resolution margin
between the blank rate on wordless voice and on words. So the tests here are
mostly about the two properties that make that number mean anything:

  * the labelling page must not show the reading it is used to judge, or the
    agreement is an artifact of the page;
  * `unsure` frames must be counted in neither direction, or an unresolved
    labelling question masquerades as a result.

The pages themselves are checked as text and as data rather than by driving a
browser, which is how the other audit pages in this repo are tested.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.audits import pause_frame_audit as adapter  # noqa: E402
from tools.audits import generate_pause_frame_audit_html as label_page  # noqa: E402
from tools.audits import generate_pause_frame_review_html as review_page  # noqa: E402


def _manifest(tmp_path: Path, windows: int = 2, frames: int = 20) -> Path:
    rows = [
        {
            "schema": adapter.MANIFEST_SCHEMA,
            "row_id": f"pause-{index:04d}",
            "audio": f"media/pause-{index:04d}.wav",
            "duration_s": round(frames * adapter.FRAME_HOP_S, 4),
            "frame_count": frames,
            "frame_hop_s": adapter.FRAME_HOP_S,
        }
        for index in range(windows)
    ]
    path = tmp_path / "manifest.jsonl"
    adapter.write_jsonl(path, rows)
    return path


def _labels(
    tmp_path: Path,
    row_id: str,
    spans: list[tuple[str, int, int]],
    frames: int = 20,
) -> Path:
    path = tmp_path / adapter.MANUAL_LABEL_FILENAME
    adapter.write_jsonl(
        path,
        [
            {
                "schema": adapter.MANUAL_LABEL_SCHEMA,
                "row_id": row_id,
                "frame_count": frames,
                "frame_hop_s": adapter.FRAME_HOP_S,
                "corrected_span_signature": "0" * 64,
                "segments": [
                    {
                        "label": label,
                        "start_frame": begin,
                        "end_frame": end,
                        "start_s": adapter.frames_to_seconds(begin),
                        "end_s": adapter.frames_to_seconds(end),
                    }
                    for label, begin, end in spans
                ],
                "note": "",
                "updated_at": "2026-08-05T20:00:00.000Z",
            }
        ],
    )
    return path


def _readings(tmp_path: Path, row_id: str, runs: list[tuple[int, int]]) -> Path:
    path = tmp_path / "head_readings.jsonl"
    adapter.write_jsonl(
        path,
        [
            {
                "row_id": row_id,
                "min_blank_s": 0.6,
                "blank_runs": [
                    [adapter.frames_to_seconds(a), adapter.frames_to_seconds(b)]
                    for a, b in runs
                ],
            }
        ],
    )
    return path


class TestFrameResolution:
    """The labels have to line up with the posterior, or nothing joins.

    `drop_span_words_v1` cannot answer this question precisely because its spans
    do not: median 7.47 s against a gate that decides at 38.5 ms.
    """

    def test_the_hop_is_the_heads_own_output_frame(self) -> None:
        from asr import alignment

        assert adapter.FRAME_HOP_S == pytest.approx(
            alignment.frame_to_seconds(1, upsample=2)
        )

    def test_frames_and_seconds_round_trip(self) -> None:
        for frame in (0, 1, 13, 26, 207):
            assert adapter.seconds_to_frame(adapter.frames_to_seconds(frame)) == frame

    def test_a_partial_trailing_frame_is_dropped(self) -> None:
        """Half a frame has no audio behind its second half; asking about it
        would be asking about silence the file does not contain."""
        window = adapter.PauseWindow(
            row_id="pause-0000",
            audio="media/x.wav",
            source_class="definite_keep",
            duration_s=adapter.FRAME_HOP_S * 10.6,
        )
        assert window.frame_count == 10


class TestBlindLabelling:
    """The page must not show what it is being used to judge."""

    def test_the_labelling_page_carries_no_model_output(self, tmp_path: Path) -> None:
        """Checked on the embedded payload, not on prose: the intro says the
        words "blank 游程" precisely to promise they are absent, so a substring
        scan over the whole document would fail on the promise itself."""
        summary = label_page.build(_manifest(tmp_path), tmp_path, "prompt")
        assert summary["shows_model_output"] is False
        text = (tmp_path / "index.html").read_text(encoding="utf-8")
        payload = text.split("const WINDOWS=", 1)[1].split(";const STORAGE_KEY", 1)[0]
        windows = json.loads(payload)
        # Exactly what the ear needs and nothing that could anchor it.
        assert all(
            set(window) == {"row_id", "audio", "frame_count"} for window in windows
        )
        for forbidden in ("blank_runs", "head_readings", "cut_point", "min_blank"):
            assert forbidden not in text, forbidden

    def test_the_source_class_never_reaches_the_page(self, tmp_path: Path) -> None:
        """`definite_keep` vs `definite_drop` is the sampling frame, and knowing
        which pool a window came from would answer the question for the ear."""
        label_page.build(_manifest(tmp_path), tmp_path, "prompt")
        text = (tmp_path / "index.html").read_text(encoding="utf-8")
        for forbidden in ("definite_keep", "definite_drop", "ambiguous_ignore"):
            assert forbidden not in text, forbidden
        manifest_text = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8")
        assert "source_class" not in manifest_text

    def test_every_window_starts_undecided(self, tmp_path: Path) -> None:
        """A default of `silence` would let an untouched window serialize as a
        real answer."""
        summary = label_page.build(_manifest(tmp_path), tmp_path, "prompt")
        assert summary["default_label"] == adapter.LABEL_UNSURE
        text = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert "label:'unsure',start_frame:0,end_frame:entry.frame_count" in text

    def test_the_page_offers_every_label_including_unsure(self, tmp_path: Path) -> None:
        label_page.build(_manifest(tmp_path), tmp_path, "prompt")
        text = (tmp_path / "index.html").read_text(encoding="utf-8")
        for label in adapter.PARTITION_LABELS:
            assert f'data-label="{label}"' in text
        assert adapter.LABEL_UNSURE in adapter.PARTITION_LABELS


class TestSplitAffordance:
    """The editor has to be able to produce a second interval.

    It could not, at first. `splitAt` was wired to `.pause-strip`, but the label
    segments are absolutely positioned buttons that tile that strip end to end
    and call `stopPropagation` so that clicking one auditions it. With every
    window starting as a single `unsure` span covering the whole width, the
    split handler was unreachable from the very first click - the page offered
    exactly one interval per window forever, which is the resolution this audit
    exists to avoid.
    """

    def test_the_split_target_is_not_the_strip_the_segments_cover(
        self, tmp_path: Path
    ) -> None:
        label_page.build(_manifest(tmp_path), tmp_path, "prompt")
        text = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert 'class="pause-ruler"' in text
        assert "ruler.onclick=" in text
        # The regression itself: nothing may hang a gesture off the covered strip.
        assert "strip.onclick=" not in text

    def test_the_segments_still_swallow_their_own_clicks(self, tmp_path: Path) -> None:
        """Which is correct - it is why the split needed somewhere else to live,
        not something to undo."""
        label_page.build(_manifest(tmp_path), tmp_path, "prompt")
        text = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert "event.stopPropagation();" in text

    def test_a_boundary_can_be_placed_at_the_playhead(self, tmp_path: Path) -> None:
        """The ear finds a boundary by listening, not by looking at a flat bar."""
        label_page.build(_manifest(tmp_path), tmp_path, "prompt")
        text = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert 'data-action="cut-here"' in text
        assert "splitAt(entry,(Number(audio.currentTime)||0)/totalSeconds)" in text

    def test_a_refused_split_says_so(self, tmp_path: Path) -> None:
        """A silent no-op reads the same as a handler that never fired, which is
        how the dead split went unnoticed."""
        label_page.build(_manifest(tmp_path), tmp_path, "prompt")
        text = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert "这里已经是一条边界了" in text

    def test_the_page_says_it_is_not_one_verdict_per_window(
        self, tmp_path: Path
    ) -> None:
        label_page.build(_manifest(tmp_path), tmp_path, "prompt")
        text = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert "不是整条打一个标" in text


class TestPartitionContract:
    """Frame-aligned, gapless, complete - enforced at save, not while editing."""

    def test_the_core_helper_is_told_which_labels_are_legal(self) -> None:
        """The core used to hardcode three boundary-chain names, which meant a
        second task could only reuse it by borrowing someone else's wording."""
        core = (
            PROJECT_ROOT / "tools" / "audits" / "review_page_core.py"
        ).read_text(encoding="utf-8")
        assert "function validateAuditPartition(segments,frameCount,labels)" in core
        assert "auditPartitionLabelSet" in core

    def test_the_page_validates_against_this_audits_labels(self, tmp_path: Path) -> None:
        label_page.build(_manifest(tmp_path), tmp_path, "prompt")
        text = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert "validateAuditPartition(annotation.segments,entry.frame_count,PAUSE_LABELS)" in text

    def test_an_all_unsure_partition_is_valid_but_not_complete(
        self, tmp_path: Path
    ) -> None:
        """It is a legal partition and a decided one are different things."""
        label_page.build(_manifest(tmp_path), tmp_path, "prompt")
        text = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert "segment.label!=='unsure'" in text

    def test_a_hole_in_the_partition_reads_as_unsure_not_as_a_guess(self) -> None:
        labels = adapter.expand_partition(
            [{"label": adapter.LABEL_WORD, "start_frame": 0, "end_frame": 4}], 10
        )
        assert labels[:4] == [adapter.LABEL_WORD] * 4
        assert labels[4:] == [adapter.LABEL_UNSURE] * 6


class TestSeparationMetric:
    """The number the audit exists to produce."""

    def test_the_margin_is_wordless_voice_minus_words(self) -> None:
        table = {
            adapter.LABEL_WORD: {"blank": 92, "non_blank": 8},
            adapter.LABEL_NON_SEMANTIC: {"blank": 100, "non_blank": 0},
            adapter.LABEL_SILENCE: {"blank": 100, "non_blank": 0},
        }
        report = adapter.separation_report(table)
        assert report["blank_rate_word"] == pytest.approx(0.92)
        # The 07-31 shape: audible voice and words read almost the same.
        assert report["margin_vs_non_semantic_pp"] == pytest.approx(8.0)

    def test_a_label_with_no_frames_reports_none_rather_than_zero(self) -> None:
        report = adapter.separation_report(
            {adapter.LABEL_WORD: {"blank": 0, "non_blank": 0}}
        )
        assert report["blank_rate_word"] is None
        assert "margin_vs_non_semantic_pp" not in report

    def test_unsure_frames_are_counted_in_neither_direction(
        self, tmp_path: Path
    ) -> None:
        """Folding them into either side would let an unresolved labelling
        question look like a result."""
        manifest = _manifest(tmp_path, windows=1, frames=20)
        labels = _labels(
            tmp_path,
            "pause-0000",
            [
                (adapter.LABEL_WORD, 0, 5),
                (adapter.LABEL_UNSURE, 5, 15),
                (adapter.LABEL_SILENCE, 15, 20),
            ],
        )
        readings = _readings(tmp_path, "pause-0000", [(15, 20)])
        result = review_page.build(
            manifest_path=manifest,
            labels_path=labels,
            output_dir=tmp_path,
            readings_path=readings,
            min_blank_s=0.6,
        )
        assert result["unsure_frames"] == 10
        assert result["decisive_frames"] == 10
        assert result["blank_rate_word"] == 0.0
        assert result["blank_rate_silence"] == 1.0

    def test_blank_runs_become_the_frames_they_covered(self) -> None:
        mask = adapter.blank_frames_from_runs(
            [(adapter.frames_to_seconds(2), adapter.frames_to_seconds(5))], 8
        )
        assert mask == [False, False, True, True, True, False, False, False]


class TestReviewPage:
    """Read-only, and honest about which labels it was built from."""

    @pytest.fixture()
    def built(self, tmp_path: Path):
        manifest = _manifest(tmp_path, windows=1, frames=20)
        labels = _labels(
            tmp_path,
            "pause-0000",
            [
                (adapter.LABEL_WORD, 0, 10),
                (adapter.LABEL_NON_SEMANTIC, 10, 20),
            ],
        )
        readings = _readings(tmp_path, "pause-0000", [(10, 20)])
        result = review_page.build(
            manifest_path=manifest,
            labels_path=labels,
            output_dir=tmp_path,
            readings_path=readings,
            min_blank_s=0.6,
        )
        return result, (tmp_path / "review.html").read_text(encoding="utf-8")

    def test_the_review_page_cannot_edit_the_labels(self, built) -> None:
        """A comparison that moves with the reader's opinion is not a
        measurement. The shared core always DEFINES a save path, so the property
        is that this adapter never calls it: no state object, no save handler,
        and the button disabled."""
        _result, text = built
        assert "saveButton.disabled=true" in text
        adapter_js = text.split("const ROWS=", 1)[1]
        assert "createAuditReviewCore" not in adapter_js
        assert "state.save" not in adapter_js
        assert "fetch(" not in adapter_js

    def test_the_result_records_the_labels_it_read(self, built) -> None:
        result, _text = built
        assert len(result["inputs"]["labels"]["sha256"]) == 64
        assert result["inputs"]["labels"]["rows"] == 1

    def test_a_label_for_an_unknown_window_is_refused(self, tmp_path: Path) -> None:
        manifest = _manifest(tmp_path, windows=1, frames=20)
        labels = _labels(tmp_path, "pause-9999", [(adapter.LABEL_WORD, 0, 20)])
        with pytest.raises(SystemExit, match="unknown row_id"):
            review_page.build(
                manifest_path=manifest,
                labels_path=labels,
                output_dir=tmp_path,
                readings_path=_readings(tmp_path, "pause-9999", []),
                min_blank_s=0.6,
            )

    def test_disagreement_is_only_marked_where_the_human_decided(self, built) -> None:
        _result, text = built
        assert "if(label==='unsure')return null;" in text

    def test_the_comparison_is_painted_per_frame(self, built) -> None:
        """Span arithmetic is how the previous attempt at this comparison went
        wrong - a 17 s span scored as fifteen lost seconds."""
        result, text = built
        assert "row.labels.map" in text
        assert result["confusion_frames"][adapter.LABEL_WORD]["non_blank"] == 10
        assert result["confusion_frames"][adapter.LABEL_NON_SEMANTIC]["blank"] == 10
