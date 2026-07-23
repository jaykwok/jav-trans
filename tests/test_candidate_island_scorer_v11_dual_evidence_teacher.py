from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.audits.compare_candidate_island_preaudits as comparison_tools
from tools.audits.generate_candidate_island_dual_evidence_review import (
    BRIDGE_AUDIT_AXES,
    BRIDGE_COMBINATION_RESULTS,
    BRIDGE_VERDICT_SCHEMA,
    _is_valid_bridge_combination,
    _bridged_background_gaps,
    generate,
    parse_args as parse_review_args,
)
from tools.audits.audit_prompt import resolve_audit_prompt
from tools.audits.review_page_core import (
    AuditOptionAxis,
    AuditReviewPageSpec,
    render_audit_review_page,
    validate_audit_option_contract,
)
from tools.boundary.ja.label_candidate_island_scorer_v11_dual_evidence_with_omni import (
    PROTECT_SYSTEM_PROMPT,
    REMOVE_SYSTEM_PROMPT,
    _normalize_evidence_spans,
    _resolve_verified_audio,
    _selected_rows,
    merge_dual_evidence,
    parse_args as parse_teacher_args,
)


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_dual_evidence_prompts_keep_scorer_responsibilities_separate() -> None:
    assert "Proposal、Split、CueQC 和 Inner 之前" in PROTECT_SYSTEM_PROMPT
    assert "句子切分属于后续 Split" in PROTECT_SYSTEM_PROMPT
    assert "这是高召回保护通道" in PROTECT_SYSTEM_PROMPT
    assert "无法可靠排除语言或交流功能时，宁可保护" in PROTECT_SYSTEM_PROMPT
    assert "不代表 outside" in PROTECT_SYSTEM_PROMPT
    assert "不使用 ASR 文本" in REMOVE_SYSTEM_PROMPT
    assert "不要求逻辑上证明“绝对不可能存在语言”" in REMOVE_SYSTEM_PROMPT
    assert "未标记区域只是 unresolved，不代表 inside" in REMOVE_SYSTEM_PROMPT
    args = parse_teacher_args(["--manifest", "m", "--output-dir", "o"])
    assert args.env_file == "gemini"
    assert args.source_id == []


def test_dual_evidence_merge_uses_only_one_sided_evidence() -> None:
    protected = _normalize_evidence_spans(
        {
            "protected_spans": [
                {"start_s": 0.04, "end_s": 0.14, "reason": "dialogue"}
            ]
        },
        field="protected_spans",
        label="inside_candidate",
        duration_s=0.2,
        frame_count=10,
    )
    removable = _normalize_evidence_spans(
        {
            "safe_outside_spans": [
                {
                    "start_s": 0.0,
                    "end_s": 0.08,
                    "category": "ambience",
                    "reason": "room tone",
                },
                {
                    "start_s": 0.16,
                    "end_s": 0.2,
                    "category": "silence",
                    "reason": "silence",
                },
            ]
        },
        field="safe_outside_spans",
        label="outside_candidate",
        duration_s=0.2,
        frame_count=10,
    )
    merged = merge_dual_evidence(
        protected_spans=protected,
        safe_outside_spans=removable,
        frame_count=10,
    )
    assert [
        (span["start_frame"], span["end_frame"])
        for span in merged["safe_outside_spans"]
    ] == [(0, 2), (8, 10)]
    assert [
        (span["start_frame"], span["end_frame"])
        for span in merged["islands"]
    ] == [(4, 7)]
    assert [
        (span["start_frame"], span["end_frame"])
        for span in merged["conflict_spans"]
    ] == [(2, 4)]
    assert [
        (span["start_frame"], span["end_frame"])
        for span in merged["unresolved_spans"]
    ] == [(7, 8)]
    assert [
        (span["start_frame"], span["end_frame"])
        for span in merged["unsure_spans"]
    ] == [(2, 4), (7, 8)]


def test_dual_evidence_requires_explicit_response_field() -> None:
    try:
        _normalize_evidence_spans(
            {},
            field="protected_spans",
            label="inside_candidate",
            duration_s=1.0,
            frame_count=50,
        )
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("missing dual-evidence field must be rejected")
    assert "must contain protected_spans" in message


def test_dual_evidence_source_selection_preserves_requested_order() -> None:
    rows = [{"source_id": "a"}, {"source_id": "b"}, {"source_id": "c"}]
    assert [
        row["source_id"]
        for row in _selected_rows(rows, source_ids=["c", "a"], limit=0)
    ] == ["c", "a"]


def test_dual_evidence_audio_fallback_requires_matching_identity(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    audio = audio_root / "s.wav"
    audio.write_bytes(b"wav")
    import hashlib

    sha = hashlib.sha256(b"wav").hexdigest()
    resolved = _resolve_verified_audio(
        {
            "source_id": "s",
            "audio": "missing/source.wav",
            "audio_sha256": sha,
        },
        manifest=tmp_path / "manifest.jsonl",
        audio_root=audio_root,
    )
    assert resolved == audio.resolve()


def test_audit_page_core_and_prompt_are_adapter_configurable(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("custom bridge review", encoding="utf-8")
    resolved = resolve_audit_prompt(
        prompt_file=str(prompt_file),
        default_prompt="default",
    )
    assert resolved.text == "custom bridge review"
    assert resolved.source == str(prompt_file.resolve())
    assert len(resolved.sha256) == 64
    with pytest.raises(ValueError, match="only one"):
        resolve_audit_prompt(
            prompt="x",
            prompt_file=str(prompt_file),
            default_prompt="default",
        )
    args = parse_review_args(
        [
            "--manifest",
            "m",
            "--human-verdicts",
            "h",
            "--candidate",
            "c",
            "--output-dir",
            "o",
            "--prompt",
            "review",
        ]
    )
    assert args.prompt == "review"
    page = render_audit_review_page(
        AuditReviewPageSpec(
            title="Adapter",
            intro_html="<section>intro</section>",
            body_html='<div id="list"></div>',
            adapter_css=".x{color:red}",
            adapter_js="reviewCoreMarker=true;",
        )
    )
    assert "createAuditReviewCore" in page
    assert "/__audit_api__/save-labels" in page
    assert "reviewCoreMarker=true" in page
    with pytest.raises(ValueError, match="unreachable"):
        validate_audit_option_contract(
            axes=(AuditOptionAxis(field="verdict", options=("keep", "unsure")),),
            combination_results={("keep",): "keep"},
            is_valid_combination=lambda combination: combination == ("keep",),
        )
    with pytest.raises(ValueError, match="missing valid combinations"):
        validate_audit_option_contract(
            axes=(
                AuditOptionAxis(field="content", options=("speech", "noise")),
                AuditOptionAxis(field="coverage", options=("covered", "missed")),
            ),
            combination_results={
                ("speech", "covered"): "ok",
                ("speech", "missed"): "missed",
                ("noise", "covered"): "noise",
            },
        )


def test_bridge_audit_axes_enumerate_all_reachable_mixed_outcomes() -> None:
    validate_audit_option_contract(
        axes=BRIDGE_AUDIT_AXES,
        combination_results=BRIDGE_COMBINATION_RESULTS,
        is_valid_combination=_is_valid_bridge_combination,
    )
    assert len(BRIDGE_COMBINATION_RESULTS) == 15
    assert set(BRIDGE_COMBINATION_RESULTS.values()) == {
        "acceptable_nonsemantic_bridge",
        "human_background_contains_semantic_dialogue",
        "semantic_missed_and_background_overmerged",
        "semantic_missed_or_clipped",
        "semantic_present_and_background_overmerged",
        "teacher_overmerged_independent_background",
        "unsure",
    }
    for index, axis in enumerate(BRIDGE_AUDIT_AXES):
        assert set(axis.options) == {
            combination[index] for combination in BRIDGE_COMBINATION_RESULTS
        }


def test_scorer_review_reports_bridged_split_level_background_without_auto_failure() -> None:
    human = [
        "inside_candidate",
        "inside_candidate",
        "outside_candidate",
        "outside_candidate",
        "inside_candidate",
        "inside_candidate",
    ]
    protect = [True, True, True, True, True, True]
    gaps = _bridged_background_gaps(human, protect, source_id="source-a")
    assert gaps == [
        {
            "gap_id": "source-a::bridge-gap::000002-000004",
            "label": "bridge",
            "start_s": 0.04,
            "end_s": 0.08,
            "start_frame": 2,
            "end_frame": 4,
            "duration_s": 0.04,
            "protected_frames": 2,
            "protected_ratio": 1.0,
            "fully_bridged": True,
            "protected_overlap_spans": [
                {
                    "label": "protected_overlap",
                    "start_frame": 2,
                    "end_frame": 4,
                    "start_s": 0.04,
                    "end_s": 0.08,
                }
            ],
            "unprotected_overlap_spans": [],
        }
    ]


def test_bridge_gap_exposes_protected_and_unprotected_audio_subspans() -> None:
    human = [
        "inside_candidate",
        "inside_candidate",
        "outside_candidate",
        "outside_candidate",
        "inside_candidate",
        "inside_candidate",
    ]
    gap = _bridged_background_gaps(
        human,
        [True, True, True, False, True, True],
        source_id="source-b",
    )[0]
    assert gap["protected_frames"] == 1
    assert gap["protected_ratio"] == 0.5
    assert gap["fully_bridged"] is False
    assert gap["protected_overlap_spans"] == [
        {
            "label": "protected_overlap",
            "start_frame": 2,
            "end_frame": 3,
            "start_s": 0.04,
            "end_s": 0.06,
        }
    ]
    assert gap["unprotected_overlap_spans"] == [
        {
            "label": "unprotected_overlap",
            "start_frame": 3,
            "end_frame": 4,
            "start_s": 0.06,
            "end_s": 0.08,
        }
    ]


def test_dual_evidence_review_compares_against_full_human_truth(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(comparison_tools, "PROJECT_ROOT", tmp_path)
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"wav")
    manifest = tmp_path / "manifest.jsonl"
    human = tmp_path / "human.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    source = {
        "source_id": "s",
        "partition": "test",
        "frame_count": 10,
        "duration_s": 0.2,
        "audio": str(audio),
        "audio_sha256": "sha",
    }
    _write(manifest, [source])
    _write(
        human,
        [
            {
                "source_id": "s",
                "spans": [
                    {"label": "outside_candidate", "start_frame": 0, "end_frame": 3},
                    {"label": "inside_candidate", "start_frame": 3, "end_frame": 7},
                    {"label": "outside_candidate", "start_frame": 7, "end_frame": 10},
                ],
            }
        ],
    )
    _write(
        candidate,
        [
            {
                **source,
                "boundary_serialization_contract_id": "boundary_acoustic_binary_v12",
                "protected_evidence_spans": [
                    {"start_frame": 2, "end_frame": 7, "start_s": 0.04, "end_s": 0.14}
                ],
                "remove_evidence_spans": [
                    {"start_frame": 0, "end_frame": 4, "start_s": 0.0, "end_s": 0.08},
                    {"start_frame": 8, "end_frame": 10, "start_s": 0.16, "end_s": 0.2},
                ],
                "islands": [
                    {"start_frame": 4, "end_frame": 7, "start_s": 0.08, "end_s": 0.14}
                ],
                "unsure_spans": [
                    {"start_frame": 2, "end_frame": 4, "start_s": 0.04, "end_s": 0.08},
                    {"start_frame": 7, "end_frame": 8, "start_s": 0.14, "end_s": 0.16},
                ],
                "safe_outside_spans": [
                    {"start_frame": 0, "end_frame": 2, "start_s": 0.0, "end_s": 0.04},
                    {"start_frame": 8, "end_frame": 10, "start_s": 0.16, "end_s": 0.2},
                ],
                "conflict_spans": [
                    {"start_frame": 2, "end_frame": 4, "start_s": 0.04, "end_s": 0.08}
                ],
            }
        ],
    )
    summary = generate(
        manifest=manifest,
        human_verdicts=human,
        candidate=candidate,
        output_dir=tmp_path / "out",
        update_nav=False,
    )
    assert summary["source_count"] == 1
    assert summary["unsafe_outside_frames"] == 0
    assert summary["zero_true_speech_outside"] is True
    assert summary["protect_recall"] == 1.0
    assert summary["remove_precision"] == 5 / 6
    assert summary["final_outside_precision"] == 1.0
    assert summary["supervised_ratio"] == 0.7
    assert summary["manual_verdict_schema"] == (
        BRIDGE_VERDICT_SCHEMA
    )
    assert summary["manual_verdicts"].endswith("manual_verdicts.jsonl")
    assert summary["review_prompt_source"] == "builtin-default"
    assert len(summary["review_prompt_sha256"]) == 64
    page = (tmp_path / "out" / "index.html").read_text(encoding="utf-8")
    assert "Protect × Remove 双证据" in page
    assert "真语音被 outside 命中" in page
    assert 'preload="metadata"' in page
    assert "waitForMetadata" in page
    assert "audio.play()" in page
    assert "contains_semantic_dialogue" in page
    assert "semantic_missed_or_clipped" in page
    assert "semantic_missed_and_background_overmerged" in page
    assert "content_verdict" in page
    assert "semantic_coverage_verdict" in page
    assert "envelope_verdict" in page
    assert "acceptable_continuous_envelope" in page
    assert "overmerged_independent_background" in page
    assert BRIDGE_VERDICT_SCHEMA in page
    assert "/__audit_api__/save-labels" in page
    assert "manual_verdicts.jsonl" in page
    assert "localStorage" in page
