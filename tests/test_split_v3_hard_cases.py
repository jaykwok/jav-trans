from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.boundary.ja import label_semantic_split_hard_cases_with_omni as labeler
from tools.audits import generate_split_v3_hard_case_audit_html as audit
from tools.audits import evaluate_split_v3_hard_case_audit as evaluate_audit


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _args(tmp_path: Path) -> argparse.Namespace:
    reexport = tmp_path / "reexport"
    features = reexport / "features" / "window-a"
    audio = tmp_path / "window.wav"
    audio.write_bytes(b"wav")
    split_path = features / "semantic_split_features.jsonl"
    chunks_path = features / "pre_asr_candidates.jsonl"
    _write_jsonl(
        split_path,
        [
            {
                "index": 1,
                "time_s": 2.0,
                "label": "cut",
                "accepted": True,
                "p_cut": 0.91,
            },
            {
                "index": 2,
                "time_s": 6.0,
                "label": "continue",
                "accepted": False,
                "p_cut": 0.42,
            },
        ],
    )
    _write_jsonl(
        chunks_path,
        [{"start": 0.0, "end": 10.0, "duration_s": 10.0}],
    )
    _write_jsonl(
        reexport / "source_windows.jsonl",
        [
            {
                "window_id": "window-a",
                "video_id": "video-a",
                "audio_wav": str(audio),
                "duration_s": 10.0,
                "source_start_s": 0.0,
                "semantic_split_metadata": str(split_path),
                "pre_asr_candidates": str(chunks_path),
            }
        ],
    )
    legacy_path = tmp_path / "legacy.jsonl"
    _write_jsonl(
        legacy_path,
        [
            {
                "window_id": "window-a",
                "feature_index": 1,
                "label": "continue",
                "confidence": 0.95,
                "flags": ["same_sentence", "breath"],
            },
            {
                "window_id": "window-a",
                "feature_index": 2,
                "label": "cut",
                "confidence": 0.90,
                "flags": ["speaker_change"],
            },
        ],
    )
    return argparse.Namespace(
        reexport_dir=str(reexport),
        legacy_labels=str(legacy_path),
        output_dir=str(tmp_path / "out"),
        max_windows=100,
        max_candidates_per_window=24,
        max_total_candidates=0,
        long_residual_min_s=8.0,
        confidence_floor=0.8,
        model="qwen3.5-omni-plus",
        env_file="",
        audio_content_mode="input_audio",
        timeout_s=10.0,
        max_tokens=4096,
        rpm=0.0,
        max_attempts=2,
        enable_thinking=True,
        thinking_budget=1024,
        clip_radius_s=4.0,
        prepare_only=False,
        resume_selection=False,
        heldout_source_audio="",
        heldout_center_s=23.538217544555664,
    )


def test_split_v3_hard_case_selection_covers_false_cut_and_long_residual(tmp_path: Path) -> None:
    args = _args(tmp_path)

    selected = labeler.prepare_hard_cases(args)

    assert len(selected) == 1
    candidates = {row["feature_index"]: row for row in selected[0]["candidates"]}
    assert "false_cut_risk" in candidates[1]["hard_case_categories"]
    assert "false_continue_risk" in candidates[2]["hard_case_categories"]
    assert "nonspeech_junction" in candidates[1]["hard_case_categories"]
    assert "speaker_change" in candidates[2]["hard_case_categories"]
    assert all("long_residual" in row["hard_case_categories"] for row in candidates.values())


def test_split_v3_heldout_is_known_same_sentence_continue(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path)
    source = tmp_path / "heldout.wav"
    source.write_bytes(b"wav")
    args.heldout_source_audio = str(source)

    def fake_slice(**kwargs):
        output = kwargs["output_path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"wav")
        return output

    monkeypatch.setattr(labeler, "slice_audio_clip", fake_slice)
    heldout = labeler.prepare_hard_cases(args)[0]
    candidate = heldout["candidates"][0]

    assert heldout["window_id"] == "FJIN-059-known-same-sentence-23.538"
    assert heldout["categories"] == ["known_same_sentence_false_cut"]
    assert candidate["expected_gate_label"] == "continue"
    assert candidate["hard_case_categories"] == ["known_same_sentence_false_cut"]


def test_split_v3_prompt_does_not_leak_model_scores_or_labels(tmp_path: Path) -> None:
    item = labeler.prepare_hard_cases(_args(tmp_path))[0]
    candidates = {row["feature_index"]: row for row in item["candidates"]}

    # Candidate f00001 @ 2.0s in a 10s window -> clip [0, 6], offset 2.0.
    prompt_1 = labeler.build_prompt(candidates[1], clip_start=0.0, clip_end=6.0)
    payload_1 = json.loads(prompt_1)
    assert payload_1 == {"duration_s": 6.0, "candidate": {"id": "f00001", "time_s": 2.0}}

    # Candidate f00002 @ 6.0s -> clip [2, 10], offset 4.0.
    prompt_2 = labeler.build_prompt(candidates[2], clip_start=2.0, clip_end=10.0)
    payload_2 = json.loads(prompt_2)
    assert payload_2 == {"duration_s": 8.0, "candidate": {"id": "f00002", "time_s": 4.0}}

    assert "p_cut" not in prompt_1
    assert "current_label" not in prompt_1
    assert "唯一任务" in labeler.SYSTEM_PROMPT
    assert "短暂停顿" in labeler.SYSTEM_PROMPT
    assert "学生｜は静かです" in labeler.SYSTEM_PROMPT
    assert "话题变化" in labeler.SYSTEM_PROMPT


def test_split_v3_realtime_plus_labels_one_window(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path)

    def fake_slice(**kwargs):
        output = kwargs["output_path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"wav")
        return output

    decisions = {
        "f00001": {
            "id": "f00001",
            "label": "continue",
            "confidence": 0.98,
            "left_complete": False,
            "right_complete": False,
            "merged_better": True,
            "flags": ["same_sentence"],
            "reason": "句内停顿",
        },
        "f00002": {
            "id": "f00002",
            "label": "cut",
            "confidence": 0.96,
            "left_complete": True,
            "right_complete": True,
            "merged_better": False,
            "flags": ["speaker_change"],
            "reason": "完整话轮边界",
        },
    }

    def fake_call(**kwargs):
        assert kwargs["model"] == "qwen3.5-omni-plus"
        assert kwargs["system_prompt"] == labeler.SYSTEM_PROMPT
        assert kwargs["enable_thinking"] is True
        assert kwargs["thinking_budget"] == 1024
        candidate_id = json.loads(kwargs["prompt"])["candidate"]["id"]
        return decisions[candidate_id], {"usage": {"total_tokens": 10}}

    monkeypatch.setattr(labeler, "slice_audio_clip", fake_slice)
    monkeypatch.setattr(labeler, "call_omni", fake_call)
    monkeypatch.setattr(labeler, "load_env_file", lambda _path: None)
    monkeypatch.setattr(labeler, "first_env_value", lambda _names: ("", ""))
    summary = labeler.run(args)
    labels = labeler._read_jsonl(Path(args.output_dir) / "omni_split_labels.jsonl")

    assert summary["processed_candidates"] == 2
    assert summary["clip_radius_s"] == 4.0
    assert sorted(row["label"] for row in labels) == ["continue", "cut"]
    assert all(row["trainable"] for row in labels)
    assert all(row["reason"] for row in labels)
    assert all("request_candidate_offset_s" in row for row in labels)
    assert all(row["request_clip_radius_s"] == 4.0 for row in labels)


def test_split_v3_clip_radius_is_cli_adjustable() -> None:
    args = labeler.parse_args(
        [
            "--reexport-dir", "x",
            "--legacy-labels", "y",
            "--output-dir", "z",
            "--clip-radius-s", "3.0",
        ]
    )
    assert args.clip_radius_s == 3.0

    default_args = labeler.parse_args(
        ["--reexport-dir", "x", "--legacy-labels", "y", "--output-dir", "z"]
    )
    assert default_args.clip_radius_s == labeler.CLIP_RADIUS_S == 4.0

    # Radius participates in clip geometry: narrower radius, tighter clip.
    assert labeler._clip_bounds(6.0, 10.0, radius_s=3.0) == (3.0, 9.0)
    assert labeler._clip_bounds(6.0, 10.0, radius_s=4.0) == (2.0, 10.0)


def test_split_v3_total_candidate_limit_round_robins_windows() -> None:
    windows = [
        {"window_id": "a", "candidates": [{"id": "a1"}, {"id": "a2"}]},
        {"window_id": "b", "candidates": [{"id": "b1"}, {"id": "b2"}]},
        {"window_id": "c", "candidates": [{"id": "c1"}, {"id": "c2"}]},
    ]

    limited = labeler._limit_total_candidates(windows, 4)

    assert [row["window_id"] for row in limited] == ["a", "b", "c"]
    assert [[item["id"] for item in row["candidates"]] for row in limited] == [
        ["a1", "a2"],
        ["b1"],
        ["c1"],
    ]


def test_split_v3_audit_prioritizes_disagreements_and_has_manual_controls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args = _args(tmp_path)
    selected = labeler.prepare_hard_cases(args)
    labels = [
        {
            "window_id": "window-a",
            "feature_index": 1,
            "time_s": 2.0,
            "partition": "train",
            "current_label": "cut",
            "legacy_label": "continue",
            "label": "continue",
            "omni_label": "continue",
            "confidence": 0.98,
            "left_complete": False,
            "right_complete": False,
            "merged_better": True,
            "flags": ["same_sentence"],
            "hard_case_categories": ["false_cut_risk"],
            "expected_gate_label": None,
        }
    ]
    selected_path = tmp_path / "selected.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(selected_path, selected)
    _write_jsonl(labels_path, labels)

    def fake_slice(**kwargs):
        kwargs["output"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output"].write_bytes(b"wav")
        return 0.0, 2.0

    monkeypatch.setattr(audit, "_slice_context", fake_slice)
    summary = audit.build_audit(
        selected_windows=selected_path,
        labels=labels_path,
        output_dir=tmp_path / "audit",
        limit=200,
        context_s=5.0,
        update_nav=False,
    )
    page = (tmp_path / "audit" / "index.html").read_text(encoding="utf-8")

    assert summary["item_count"] == 1
    assert summary["current_disagreement_count"] == 1
    assert "manual_verdicts.jsonl" in page
    assert "cut</button>" in page
    assert "continue</button>" in page
    assert ".join('\\n')+'\\n'" in page


def test_split_v3_audit_evaluation_applies_ninety_percent_gates(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    verdicts = tmp_path / "manual_verdicts.jsonl"
    rows = [
        {"candidate_id": "a", "label": "cut", "expected_gate_label": "cut"},
        {"candidate_id": "b", "label": "continue", "expected_gate_label": None},
    ]
    _write_jsonl(manifest, rows)
    _write_jsonl(
        verdicts,
        [
            {"candidate_id": "a", "verdict": "cut"},
            {"candidate_id": "b", "verdict": "continue"},
        ],
    )

    summary = evaluate_audit.evaluate(
        manifest=manifest,
        verdicts=verdicts,
        output=tmp_path / "evaluation.json",
    )

    assert summary["pass"] is True
    assert summary["cut_precision"] == 1.0
    assert summary["cut_recall"] == 1.0
    assert summary["continue_recall"] == 1.0
