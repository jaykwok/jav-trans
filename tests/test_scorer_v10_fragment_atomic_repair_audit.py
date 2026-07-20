from __future__ import annotations

import json
from pathlib import Path

from tools.audits.evaluate_scorer_v10_fragment_atomic_repair_audit import evaluate
from tools.audits.generate_scorer_v10_fragment_atomic_repair_audit_html import (
    FRAGMENT_VERDICT_SCHEMA,
    MANUAL_VERDICT_SCHEMA,
    build_audit,
    infer_atomic_units,
)


def _span(start: int, end: int, label: str) -> dict[str, object]:
    return {
        "label": label,
        "start_frame": start,
        "end_frame": end,
        "start_s": start * 0.02,
        "end_s": end * 0.02,
    }


def _gap(
    *,
    audit_id: str,
    cluster_id: str,
    source_id: str,
    audio: str,
    gap_index: int,
    left: tuple[int, int],
    gap: tuple[int, int],
    right: tuple[int, int],
) -> dict[str, object]:
    return {
        "audit_id": audit_id,
        "cluster_id": cluster_id,
        "source_id": source_id,
        "audio": audio,
        "partition": "val",
        "truth_run_index": 0,
        "gap_index": gap_index,
        "left_span": _span(*left, "model_speech_left"),
        "gap_span": _span(*gap, "truth_speech_model_background"),
        "right_span": _span(*right, "model_speech_right"),
    }


def _verdict(audit_id: str, verdict: str) -> dict[str, str]:
    return {
        "schema": FRAGMENT_VERDICT_SCHEMA,
        "audit_id": audit_id,
        "verdict": verdict,
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "source.wav").write_bytes(b"RIFF-test")
    manifest_rows = [
        _gap(
            audit_id="source-a:truth0:gap0:2-3",
            cluster_id="source-a:truth0",
            source_id="source-a",
            audio="audio/source.wav",
            gap_index=0,
            left=(0, 2),
            gap=(2, 3),
            right=(3, 5),
        ),
        _gap(
            audit_id="source-a:truth0:gap1:5-6",
            cluster_id="source-a:truth0",
            source_id="source-a",
            audio="audio/source.wav",
            gap_index=1,
            left=(3, 5),
            gap=(5, 6),
            right=(6, 8),
        ),
        _gap(
            audit_id="source-b:truth0:gap0:4-5",
            cluster_id="source-b:truth0",
            source_id="source-b",
            audio="audio/source.wav",
            gap_index=0,
            left=(0, 4),
            gap=(4, 5),
            right=(5, 9),
        ),
    ]
    verdict_rows = [
        _verdict("source-a:truth0:gap0:2-3", "separate_drop_nonsemantic"),
        _verdict("source-a:truth0:gap1:5-6", "same_asr_unit_keep_continuous"),
        _verdict("source-b:truth0:gap0:4-5", "separate_drop_nonsemantic"),
    ]
    manifest = tmp_path / "fragment_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in manifest_rows), encoding="utf-8"
    )
    verdicts = tmp_path / "fragment_verdicts.jsonl"
    verdicts.write_text(
        "".join(json.dumps(row) + "\n" for row in verdict_rows), encoding="utf-8"
    )
    canonical = tmp_path / "canonical_sources.jsonl"
    canonical.write_text(
        "".join(
            json.dumps(
                {
                    "source_id": source_id,
                    "sample_rate": 16000,
                    "sample_count": frame_count * 320,
                    "canonical_spans": [
                        {
                            "start_sample": 0,
                            "end_sample": frame_count * 320,
                            "label": "speech",
                            "core_id": f"core-{source_id}",
                        }
                    ],
                }
            )
            + "\n"
            for source_id, frame_count in (("source-a", 8), ("source-b", 9))
        ),
        encoding="utf-8",
    )
    return manifest, verdicts, canonical


def test_fragment_atomic_inference_only_leaves_unresolved_drop_sides(
    tmp_path: Path,
) -> None:
    manifest, verdicts, _canonical = _fixture(tmp_path)
    atomic_rows, relations = infer_atomic_units(
        audit_rows=[json.loads(line) for line in manifest.read_text().splitlines()],
        verdict_rows=[json.loads(line) for line in verdicts.read_text().splitlines()],
    )
    assert len(atomic_rows) == 8
    assert sum(bool(row["review_required"]) for row in atomic_rows) == 2
    assert len(relations) == 2
    source_a = [row for row in atomic_rows if row["source_id"] == "source-a"]
    assert {row["inferred_label"] for row in source_a} == {"speech", "background"}


def test_fragment_atomic_page_contains_only_minimal_manual_targets(
    tmp_path: Path,
) -> None:
    manifest, verdicts, canonical = _fixture(tmp_path)
    index = build_audit(
        canonical_sources=canonical,
        fragmentation_audit_manifest=manifest,
        fragmentation_manual_verdicts=verdicts,
        output_dir=tmp_path / "audit",
    )
    page = index.read_text(encoding="utf-8")
    summary = json.loads((index.parent / "summary.json").read_text(encoding="utf-8"))
    assert summary["atomic_unit_count"] == 8
    assert summary["auto_resolved_count"] == 6
    assert summary["review_item_count"] == 2
    assert summary["inference_conflict_count"] == 0
    assert summary["canonical_sources"] == str(canonical)
    assert "完整 island 串" in page
    assert "只补审无法判断左右归属的 2 个" in page
    assert "speech_scorer_v10_fragment_atomic_manual_verdict_v1" in page
    assert (index.parent / "audio" / "item-000.wav").is_file()


def test_fragment_atomic_evaluator_enforces_separate_drop_relation(
    tmp_path: Path,
) -> None:
    manifest, verdicts, canonical = _fixture(tmp_path)
    audit_dir = tmp_path / "audit"
    build_audit(
        canonical_sources=canonical,
        fragmentation_audit_manifest=manifest,
        fragmentation_manual_verdicts=verdicts,
        output_dir=audit_dir,
    )
    atomic_rows = [
        json.loads(line)
        for line in (audit_dir / "atomic_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    review_rows = [row for row in atomic_rows if row["review_required"]]
    manual = audit_dir / "manual_verdicts.jsonl"
    manual.write_text(
        "".join(
            json.dumps(
                {
                    "schema": MANUAL_VERDICT_SCHEMA,
                    "atomic_id": row["atomic_id"],
                    "verdict": "speech" if index == 0 else "background",
                }
            )
            + "\n"
            for index, row in enumerate(review_rows)
        ),
        encoding="utf-8",
    )
    result = evaluate(
        audit_summary=audit_dir / "summary.json",
        atomic_manifest=audit_dir / "atomic_manifest.jsonl",
        relation_manifest=audit_dir / "relation_manifest.jsonl",
        manual_verdicts=manual,
        output=audit_dir / "manual_gate.json",
    )
    assert result["complete"] is True
    assert result["relation_violation_count"] == 0
    assert result["canonical_recompile_ready"] is True
    decisions = [
        json.loads(line)
        for line in Path(result["decisions"]).read_text(encoding="utf-8").splitlines()
    ]
    assert {row["schema"] for row in decisions} == {
        "speech_scorer_v10_fragment_atomic_repair_decision_v1"
    }
    assert all(int(row["end_sample"]) > int(row["start_sample"]) for row in decisions)

    manual.write_text(
        "".join(
            json.dumps(
                {
                    "schema": MANUAL_VERDICT_SCHEMA,
                    "atomic_id": row["atomic_id"],
                    "verdict": "speech",
                }
            )
            + "\n"
            for row in review_rows
        ),
        encoding="utf-8",
    )
    rejected = evaluate(
        audit_summary=audit_dir / "summary.json",
        atomic_manifest=audit_dir / "atomic_manifest.jsonl",
        relation_manifest=audit_dir / "relation_manifest.jsonl",
        manual_verdicts=manual,
        output=audit_dir / "rejected_gate.json",
    )
    assert rejected["relation_violation_count"] == 1
    assert rejected["canonical_recompile_ready"] is False
