"""The onset audit has to stay blind, and has to refuse to overclaim.

Two failure modes are worth tests here. The first is leakage: this audit's whole
design rests on the auditor not knowing which clips were deliberately mis-cut,
and there are two channels for that to leak - the manifest carrying the answer,
and clip duration varying by stratum. An earlier audit in this project was
already weakened by the second one.

The second is reading a null result as a pass. If a deliberate 400 ms late cut
is not heard as chopped, the audit had no power at that offset; concluding "the
head is accurate" from that would be exactly backwards, so the evaluator's
verdict logic is pinned against it.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.audits import evaluate_alignment_onset_audit as evaluate  # noqa: E402
from tools.audits import select_alignment_onset_audit as select  # noqa: E402
from tools.audits.binary_clip_audit import ALLOWED_MANIFEST_FIELDS  # noqa: E402

SELECT_SCRIPT = PROJECT_ROOT / "tools" / "audits" / "select_alignment_onset_audit.py"


def _line(index: int, *, start: float = 10.0, duration: float = 3.0) -> dict:
    return {
        "schema": "real_alignment_line_v1",
        "line_id": f"line-{index:03d}",
        "sample_id": f"sample-{index % 7:02d}",
        "source_id": f"video-{index % 7:02d}",
        "source_partition": "train",
        "audio": f"/audio/sample-{index % 7:02d}.wav",
        "window_duration_s": 75.0,
        "region_start_s": start - 0.2,
        "region_end_s": start + duration + 0.2,
        "line_start_s": start,
        "line_end_s": start + duration,
        "line_duration_s": duration,
        "characters": 20,
        "chars_per_s": 6.0,
        "alignment_score": -1.5,
        "boundary_trimmed": False,
        "text": "テスト",
    }


def _run_select(tmp_path: Path, lines: list[dict], **kwargs) -> Path:
    lines_path = tmp_path / "lines.jsonl"
    with lines_path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    output_dir = tmp_path / "audit"
    argv = [
        sys.executable,
        str(SELECT_SCRIPT),
        "--lines",
        str(lines_path),
        "--output-dir",
        str(output_dir),
    ]
    for key, value in kwargs.items():
        argv.extend([f"--{key.replace('_', '-')}", str(value)])
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return output_dir


@pytest.fixture
def selection(tmp_path: Path) -> Path:
    lines = [_line(index, start=10.0 + index * 0.01) for index in range(200)]
    return _run_select(
        tmp_path,
        lines,
        aligned_n=10,
        control_early_n=10,
        probe_late_150ms_n=10,
        probe_late_400ms_n=10,
    )


class TestBlinding:
    def test_the_manifest_carries_no_answer(self, selection: Path) -> None:
        rows = [
            json.loads(line)
            for line in (selection / "manifest.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        assert rows
        for row in rows:
            assert set(row) <= ALLOWED_MANIFEST_FIELDS, set(row) - ALLOWED_MANIFEST_FIELDS
            serialized = json.dumps(row, ensure_ascii=False)
            for stratum in select.STRATUM_OFFSETS:
                assert stratum not in serialized

    def test_every_clip_has_the_same_duration(self, selection: Path) -> None:
        """Duration must not encode the stratum; it did in an earlier audit."""
        rows = [
            json.loads(line)
            for line in (selection / "manifest.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        durations = {round(row["end_s"] - row["start_s"], 6) for row in rows}
        assert len(durations) == 1

    def test_a_line_is_used_by_at_most_one_stratum(self, selection: Path) -> None:
        """Hearing one onset at two offsets would make the mis-cut obvious."""
        answers = [
            json.loads(line)
            for line in (selection / "answers.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        line_ids = [row["line_id"] for row in answers]
        assert len(set(line_ids)) == len(line_ids)


class TestPlaybackContext:
    def test_the_clip_starts_a_fixed_run_up_before_the_cut(
        self, selection: Path
    ) -> None:
        """The run-up must be identical everywhere, or it encodes the stratum."""
        manifest = {
            json.loads(line)["row_id"]: json.loads(line)
            for line in (selection / "manifest.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        }
        for line in (selection / "answers.jsonl").read_text(
            encoding="utf-8"
        ).splitlines():
            if not line.strip():
                continue
            answer = json.loads(line)
            row = manifest[answer["row_id"]]
            cut = answer["line_start_s"] + select.STRATUM_OFFSETS[answer["stratum"]]
            assert row["start_s"] == pytest.approx(
                cut - select.CONTEXT_SECONDS, abs=1e-4
            )

    def test_the_page_enters_the_clip_at_the_cut_not_at_its_start(self) -> None:
        """A mismatch here would judge a moment nobody meant to test."""
        from tools.audits import generate_alignment_onset_audit_html as generate

        spec = generate._spec(generate.DEFAULT_REVIEW_PROMPT)
        assert spec.context_seconds == select.CONTEXT_SECONDS
        page = generate.render_page(
            spec,
            [
                {
                    "row_id": "onset-0000",
                    "audio_src": "media/onset-0000.mp3",
                    "start_s": 8.0,
                    "end_s": 12.0,
                    "clip_duration_s": 4.0,
                }
            ],
        )
        # The button is emitted by the browser from this value, so the config is
        # what decides it; the markup literal sits in the JS source either way.
        assert '"contextSeconds": 2.0' in page
        assert 'class="play-context"' in page
        assert "▶ 从切点" in page

    def test_two_option_audits_keep_a_single_play_button(self) -> None:
        """A plain two-option spec must not sprout a context button."""
        from tools.audits.binary_clip_audit import (
            BinaryClipAuditSpec,
            BinaryOption,
            render_page,
        )

        spec = BinaryClipAuditSpec(
            title="t",
            option_a=BinaryOption(value="words", label="A"),
            option_b=BinaryOption(value="no_words", label="B"),
            prompt="p",
            intro_html="",
            verdict_schema="s",
            storage_key="k",
            status_label="l",
            boundary_contract="c",
        )
        assert spec.context_seconds == 0.0
        page = render_page(
            spec,
            [
                {
                    "row_id": "r0",
                    "audio_src": "media/r0.mp3",
                    "start_s": 0.0,
                    "end_s": 2.0,
                    "clip_duration_s": 2.0,
                }
            ],
        )
        assert '"contextSeconds": 0.0' in page
        assert "重播" in page


class TestCutGeometry:
    def test_offsets_are_applied_with_the_documented_sign(
        self, selection: Path
    ) -> None:
        manifest = {
            json.loads(line)["row_id"]: json.loads(line)
            for line in (selection / "manifest.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        }
        for line in (selection / "answers.jsonl").read_text(
            encoding="utf-8"
        ).splitlines():
            if not line.strip():
                continue
            answer = json.loads(line)
            row = manifest[answer["row_id"]]
            cut = answer["line_start_s"] + select.STRATUM_OFFSETS[answer["stratum"]]
            assert row["start_s"] + select.CONTEXT_SECONDS == pytest.approx(
                cut, abs=1e-4
            )
        # The control must be earlier than the prediction, the probes later.
        assert select.STRATUM_OFFSETS["control_early"] < 0
        assert select.STRATUM_OFFSETS["aligned"] == 0
        assert select.STRATUM_OFFSETS["probe_late_150ms"] > 0
        assert select.STRATUM_OFFSETS["probe_late_400ms"] > (
            select.STRATUM_OFFSETS["probe_late_150ms"]
        )

    def test_lines_without_room_for_every_offset_are_rejected(
        self, tmp_path: Path
    ) -> None:
        """A line at the window edge cannot host all four cuts comparably."""
        lines = [_line(index, start=10.0 + index * 0.01) for index in range(60)]
        lines.append(_line(900, start=0.1))
        lines.append(_line(901, start=74.5))
        output_dir = _run_select(
            tmp_path,
            lines,
            aligned_n=5,
            control_early_n=5,
            probe_late_150ms_n=5,
            probe_late_400ms_n=5,
        )
        summary = json.loads((output_dir / "selection_summary.json").read_text("utf-8"))
        assert summary["rejected"]["no_room_for_every_offset"] == 2

    def test_asking_for_more_clips_than_exist_fails_loudly(
        self, tmp_path: Path
    ) -> None:
        lines = [_line(index, start=10.0 + index * 0.01) for index in range(8)]
        lines_path = tmp_path / "lines.jsonl"
        with lines_path.open("w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(json.dumps(line, ensure_ascii=False) + "\n")
        result = subprocess.run(
            [
                sys.executable,
                str(SELECT_SCRIPT),
                "--lines",
                str(lines_path),
                "--output-dir",
                str(tmp_path / "audit"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "eligible lines exist" in (result.stderr + result.stdout)


class TestThirdOption:
    def test_the_page_offers_a_non_semantic_button(self, tmp_path: Path) -> None:
        """The case that prompted it: an onset that is a sucking sound."""
        from tools.audits import generate_alignment_onset_audit_html as generate

        spec = generate._spec(generate.DEFAULT_REVIEW_PROMPT)
        contract = generate.option_contract(spec)
        assert spec.option_c is not None
        assert set(contract.values()) == {"intact", "clipped", "non_semantic", "unsure"}

    def test_two_option_audits_are_unaffected(self) -> None:
        """Other audits share this spec; none may gain a phantom button."""
        from tools.audits.binary_clip_audit import (
            BinaryClipAuditSpec,
            BinaryOption,
            option_contract,
        )

        spec = BinaryClipAuditSpec(
            title="t",
            option_a=BinaryOption(value="words", label="A"),
            option_b=BinaryOption(value="no_words", label="B"),
            prompt="p",
            intro_html="",
            verdict_schema="s",
            storage_key="k",
            status_label="l",
            boundary_contract="c",
        )
        assert spec.option_c is None
        assert set(option_contract(spec).values()) == {
            "words",
            "no_words",
            "unsure",
        }

    def test_duplicate_or_reserved_option_values_are_refused(self) -> None:
        from tools.audits.binary_clip_audit import (
            BinaryClipAuditSpec,
            BinaryOption,
            option_contract,
        )

        def _spec(c_value: str) -> BinaryClipAuditSpec:
            return BinaryClipAuditSpec(
                title="t",
                option_a=BinaryOption(value="a", label="A"),
                option_b=BinaryOption(value="b", label="B"),
                option_c=BinaryOption(value=c_value, label="C"),
                prompt="p",
                intro_html="",
                verdict_schema="s",
                storage_key="k",
                status_label="l",
                boundary_contract="c",
            )

        for bad in ("a", "unsure"):
            with pytest.raises(ValueError, match="distinct"):
                option_contract(_spec(bad))
        assert option_contract(_spec("c"))


def _verdicts(counts: dict[str, tuple[int, int]]) -> tuple[list[dict], list[dict]]:
    """Build answers/verdicts where each stratum has (clipped, intact) counts."""
    answers: list[dict] = []
    verdicts: list[dict] = []
    index = 0
    for stratum, (clipped, intact) in counts.items():
        for position in range(clipped + intact):
            row_id = f"onset-{index:04d}"
            index += 1
            answers.append(
                {
                    "row_id": row_id,
                    "stratum": stratum,
                    "offset_s": select.STRATUM_OFFSETS[stratum],
                    "line_id": row_id,
                    "source_id": "video",
                    "source_partition": "train",
                    "line_start_s": 10.0,
                    "line_duration_s": 3.0,
                }
            )
            verdicts.append(
                {
                    "row_id": row_id,
                    "verdict": "clipped" if position < clipped else "intact",
                }
            )
    return answers, verdicts


def _evaluate(tmp_path: Path, counts: dict[str, tuple[int, int]]) -> dict:
    answers, verdicts = _verdicts(counts)
    answers_path = tmp_path / "answers.jsonl"
    verdicts_path = tmp_path / "verdicts.jsonl"
    for path, rows in ((answers_path, answers), (verdicts_path, verdicts)):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return evaluate.build(answers_path, verdicts_path)


class TestVerdictLogic:
    def test_a_failed_control_makes_the_result_unusable(self, tmp_path: Path) -> None:
        result = _evaluate(
            tmp_path,
            {
                "control_early": (20, 20),
                "aligned": (2, 38),
                "probe_late_150ms": (10, 30),
                "probe_late_400ms": (35, 5),
            },
        )
        assert "unusable" in result["verdict"]

    def test_an_undetectable_probe_reads_as_no_power_not_as_a_pass(
        self, tmp_path: Path
    ) -> None:
        """The trap this test exists for.

        The floor is clean and `aligned` looks fine, but the deliberate 400 ms
        cut was not heard either - so the audit measured nothing at that offset
        and must not be reported as the head passing.
        """
        result = _evaluate(
            tmp_path,
            {
                "control_early": (2, 38),
                "aligned": (3, 37),
                "probe_late_150ms": (3, 37),
                "probe_late_400ms": (4, 36),
            },
        )
        assert "no power" in result["verdict"]
        assert "not systematically late" not in result["verdict"]

    def test_a_detectable_probe_with_a_clean_prediction_bounds_the_error(
        self, tmp_path: Path
    ) -> None:
        result = _evaluate(
            tmp_path,
            {
                "control_early": (2, 38),
                "aligned": (3, 37),
                "probe_late_150ms": (6, 34),
                "probe_late_400ms": (34, 6),
            },
        )
        assert "not systematically late by 400 ms" in result["verdict"]

    def test_both_probes_detectable_tightens_the_bound_to_150ms(
        self, tmp_path: Path
    ) -> None:
        result = _evaluate(
            tmp_path,
            {
                "control_early": (1, 39),
                "aligned": (2, 38),
                "probe_late_150ms": (20, 20),
                "probe_late_400ms": (36, 4),
            },
        )
        assert "not systematically late by 150 ms" in result["verdict"]

    def test_a_skipped_past_probe_does_not_mask_a_late_head(
        self, tmp_path: Path
    ) -> None:
        """What actually happened on 2026-07-31.

        The +400 ms probe cleared short first words entirely and landed on the
        next clean onset, so it scored LESS clipped than +150 ms. Reading the
        ladder in offset order would then have called the whole audit powerless
        - even though the only comparison that matters, prediction against
        floor, separated cleanly.
        """
        result = _evaluate(
            tmp_path,
            {
                "control_early": (1, 29),
                "aligned": (13, 14),
                "probe_late_150ms": (15, 9),
                "probe_late_400ms": (9, 14),
            },
        )
        assert result["skip_past_suspected"] is True
        assert "systematically late" in result["verdict"]
        assert "no power" not in result["verdict"]

    def test_a_monotonic_ladder_is_not_flagged(self, tmp_path: Path) -> None:
        result = _evaluate(
            tmp_path,
            {
                "control_early": (1, 39),
                "aligned": (2, 38),
                "probe_late_150ms": (20, 20),
                "probe_late_400ms": (36, 4),
            },
        )
        assert result["skip_past_suspected"] is False

    def test_a_late_head_is_reported_as_late(self, tmp_path: Path) -> None:
        result = _evaluate(
            tmp_path,
            {
                "control_early": (2, 38),
                "aligned": (30, 10),
                "probe_late_150ms": (28, 12),
                "probe_late_400ms": (36, 4),
            },
        )
        assert "systematically late" in result["verdict"]
        assert "not systematically late" not in result["verdict"]

    def test_non_semantic_leaves_the_timing_rate_but_is_still_counted(
        self, tmp_path: Path
    ) -> None:
        """C is a fact about the clip, not about the cut, so it cannot be scored."""
        answers, verdicts = _verdicts(
            {"aligned": (4, 16), "control_early": (2, 18)}
        )
        for verdict in verdicts[:6]:
            verdict["verdict"] = "non_semantic"
        answers_path = tmp_path / "answers.jsonl"
        verdicts_path = tmp_path / "verdicts.jsonl"
        for path, rows in ((answers_path, answers), (verdicts_path, verdicts)):
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        result = evaluate.build(answers_path, verdicts_path)
        aligned = result["strata"]["aligned"]
        assert aligned["non_semantic"] == 6
        assert aligned["decisive"] == 14
        assert aligned["clipped"] == 0
        assert aligned["non_semantic_share"] == pytest.approx(6 / 20)

    def test_too_much_non_semantic_reports_low_power_not_a_result(
        self, tmp_path: Path
    ) -> None:
        """The known cost of making C exclusive, surfaced instead of hidden.

        If most openings in this domain are moans and breathing, the timing
        question simply was not answered often enough to compare - and that must
        read as "draw more clips", not as the head passing or failing.
        """
        answers, verdicts = _verdicts(
            {
                "control_early": (1, 29),
                "aligned": (1, 29),
                "probe_late_150ms": (5, 20),
                "probe_late_400ms": (20, 5),
            }
        )
        for verdict in verdicts:
            if verdict["verdict"] == "intact":
                verdict["verdict"] = "non_semantic"
        answers_path = tmp_path / "answers.jsonl"
        verdicts_path = tmp_path / "verdicts.jsonl"
        for path, rows in ((answers_path, answers), (verdicts_path, verdicts)):
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        result = evaluate.build(answers_path, verdicts_path)
        assert "too few decisive answers" in result["verdict"]
        assert "top-up" in result["verdict"]

    def test_unsure_is_an_exit_not_an_answer(self, tmp_path: Path) -> None:
        answers, verdicts = _verdicts({"aligned": (2, 8), "control_early": (1, 9)})
        verdicts[0]["verdict"] = "unsure"
        answers_path = tmp_path / "answers.jsonl"
        verdicts_path = tmp_path / "verdicts.jsonl"
        for path, rows in ((answers_path, answers), (verdicts_path, verdicts)):
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        result = evaluate.build(answers_path, verdicts_path)
        aligned = result["strata"]["aligned"]
        assert aligned["unsure"] == 1
        assert aligned["decisive"] == 9
        assert aligned["clipped"] == 1

    def test_a_verdict_for_an_unknown_row_is_refused(self, tmp_path: Path) -> None:
        answers, verdicts = _verdicts({"aligned": (1, 1)})
        verdicts.append({"row_id": "onset-9999", "verdict": "intact"})
        answers_path = tmp_path / "answers.jsonl"
        verdicts_path = tmp_path / "verdicts.jsonl"
        for path, rows in ((answers_path, answers), (verdicts_path, verdicts)):
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with pytest.raises(ValueError, match="unknown row_id"):
            evaluate.build(answers_path, verdicts_path)


class TestInputProvenance:
    """A result must carry enough about its inputs to be caught when stale.

    This audit's verdicts are typed by hand over days. The first `result.json`
    was written mid-pass, recorded `aligned` at 48.1% and a verdict of "the head
    IS systematically late", and sat on disk for two days after the completed
    110/110 pass reversed both - because nothing in the file said which verdicts
    it had been built from.
    """

    def test_the_result_records_the_verdict_file_it_was_built_from(
        self, tmp_path: Path
    ) -> None:
        result = _evaluate(tmp_path, {"aligned": (2, 8), "control_early": (1, 9)})
        verdicts = result["inputs"]["verdicts"]
        assert verdicts["rows"] == 20
        assert verdicts["verdict_counts"] == {"clipped": 3, "intact": 17}
        assert len(verdicts["sha256"]) == 64

    def test_editing_one_verdict_changes_the_recorded_digest(
        self, tmp_path: Path
    ) -> None:
        """The check that makes staleness mechanical: re-hash the input file and
        compare it with what the result claims."""
        answers, verdicts = _verdicts({"aligned": (2, 8), "control_early": (1, 9)})
        answers_path = tmp_path / "answers.jsonl"
        verdicts_path = tmp_path / "verdicts.jsonl"

        def _write(rows: list[dict]) -> None:
            with verdicts_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        with answers_path.open("w", encoding="utf-8") as handle:
            for row in answers:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        _write(verdicts)
        before = evaluate.build(answers_path, verdicts_path)
        verdicts[0]["verdict"] = "non_semantic"
        _write(verdicts)
        after = evaluate.build(answers_path, verdicts_path)
        assert before["inputs"]["verdicts"]["sha256"] != after["inputs"]["verdicts"]["sha256"]

    def test_the_newest_edit_time_is_carried_through(self, tmp_path: Path) -> None:
        answers, verdicts = _verdicts({"aligned": (2, 8), "control_early": (1, 9)})
        for position, row in enumerate(verdicts):
            row["updated_at"] = f"2026-08-01T16:{position:02d}:00.000Z"
        answers_path = tmp_path / "answers.jsonl"
        verdicts_path = tmp_path / "verdicts.jsonl"
        for path, rows in ((answers_path, answers), (verdicts_path, verdicts)):
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        result = evaluate.build(answers_path, verdicts_path)
        assert result["inputs"]["verdicts"]["latest_updated_at"] == (
            "2026-08-01T16:19:00.000Z"
        )
