#!/usr/bin/env python3
"""Derive Scorer-duty truth for manually selected downstream isolation gaps.

The source review is intentionally preserved.  Selected background gaps become
``unsure`` for Scorer until a source/audio/checkpoint-bound workflow replay
demonstrates that Proposal + Split isolate the gap and CueQC drops the resulting
provisional sub-island.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import quote


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for _root in (PROJECT_ROOT, SRC_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402


SUMMARY_SCHEMA = "candidate_island_downstream_isolation_compile_summary_v1"
SELECTION_SCHEMA = "candidate_island_downstream_isolation_selection_v1"
REQUIREMENT_SCHEMA = "candidate_island_downstream_isolation_requirement_v1"
AUDIT_SCHEMA = "candidate_island_downstream_isolation_audit_v1"
EVIDENCE_SCHEMA = "candidate_island_downstream_isolation_evidence_v1"
RESPONSIBILITY_VERDICT_SCHEMA = (
    "candidate_island_scorer_v11_responsibility_manual_verdict_v1"
)
HELDOUT_VERDICT_SCHEMA = "candidate_island_scorer_v11_heldout_manual_verdict_v1"
BRIDGE_VERDICT_SCHEMA = "candidate_island_scorer_v11_bridge_gap_manual_verdict_v3"
HELDOUT_AUDIT_ITEM_SCHEMA = "candidate_island_scorer_v11_heldout_audit_item_v1"
DUAL_REVIEW_SUMMARY_SCHEMA = "candidate_island_dual_evidence_review_summary_v1"
FRAME_HOP_S = 0.02
LABELS = {"outside_candidate", "inside_candidate", "unsure"}
REQUIRED_EVIDENCE_STAGES = (
    "scorer_candidate_islands",
    "proposal_candidates",
    "split_events",
    "provisional_sub_islands",
    "cueqc_decisions",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL objects required: {path}")
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str | Path, *, owner: Path | None = None) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    if owner is not None:
        owner_relative = (owner.parent / candidate).resolve()
        if owner_relative.exists():
            return owner_relative
    return (PROJECT_ROOT / candidate).resolve()


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _index(rows: Sequence[dict[str, Any]], field: str, *, name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get(field) or "")
        if not key:
            raise ValueError(f"{name} row is missing {field}")
        if key in result:
            raise ValueError(f"duplicate {name} {field}: {key}")
        result[key] = row
    return result


def _require_contract(row: dict[str, Any], *, name: str) -> None:
    if row.get("boundary_serialization_contract_id") != (
        ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    ):
        raise ValueError(f"wrong central boundary contract in {name}")


def _require_sha256(value: Any, *, name: str) -> str:
    normalized = str(value or "").lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return normalized


def _validate_spans(row: dict[str, Any]) -> list[str]:
    source_id = str(row.get("source_id") or "")
    frame_count = int(row.get("frame_count") or 0)
    if frame_count <= 0 or float(row.get("frame_hop_s") or 0.0) != FRAME_HOP_S:
        raise ValueError(f"invalid held-out frame geometry: {source_id}")
    labels: list[str] = []
    cursor = 0
    for span in row.get("spans") or ():
        label = str(span.get("label") or "")
        start = int(span.get("start_frame", -1))
        end = int(span.get("end_frame", -1))
        if label not in LABELS or start != cursor or not start < end <= frame_count:
            raise ValueError(f"held-out spans must cover the full source: {source_id}")
        labels.extend([label] * (end - start))
        cursor = end
    if cursor != frame_count:
        raise ValueError(f"held-out spans do not cover source tail: {source_id}")
    return labels


def _label_runs(labels: Sequence[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index < len(labels) and labels[index] == labels[start]:
            continue
        result.append(
            {
                "label": labels[start],
                "start_frame": start,
                "end_frame": index,
                "start_s": round(start * FRAME_HOP_S, 6),
                "end_s": round(index * FRAME_HOP_S, 6),
            }
        )
        start = index
    return result


def _bound_summary_path(
    summary: dict[str, Any],
    *,
    field: str,
    sha_field: str,
    owner: Path,
) -> Path:
    path = _resolve(str(summary.get(field) or ""), owner=owner)
    if not path.is_file():
        raise FileNotFoundError(path)
    if _sha256(path) != _require_sha256(summary.get(sha_field), name=sha_field):
        raise ValueError(f"dual review {field} SHA mismatch")
    return path


def _audio_url(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"audit audio must be inside project root: {path}") from error
    return "/" + "/".join(quote(part) for part in relative.parts)


def _resolve_audit_audio(
    value: str,
    *,
    audit_manifest: Path,
    audio_root: Path | None,
) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        return relative.resolve()
    bound = (audit_manifest.parent / relative).resolve()
    if bound.is_file() or audio_root is None:
        return bound
    return (audio_root / relative.name).resolve()


def _event_near(events: Sequence[dict[str, Any]], frame: int, *, tolerance: int) -> bool:
    return any(
        str(event.get("argmax_label") or event.get("decision") or "") == "cut"
        and abs(int(event.get("frame", event.get("frame_index", -10**9))) - frame)
        <= tolerance
        for event in events
    )


def _candidate_near(
    candidates: Sequence[dict[str, Any]], frame: int, *, tolerance: int
) -> bool:
    return any(
        abs(int(candidate.get("frame", candidate.get("frame_index", -10**9))) - frame)
        <= tolerance
        for candidate in candidates
    )


def _evaluate_evidence(
    requirement: dict[str, Any],
    evidence: dict[str, Any] | None,
    *,
    expected_checkpoint_shas: dict[str, str] | None,
    tolerance_frames: int,
) -> tuple[str, list[str], dict[str, bool]]:
    checks = {
        "scorer_envelope_kept": False,
        "proposal_at_both_edges": False,
        "split_cut_at_both_edges": False,
        "provisional_background_island": False,
        "cueqc_dropped_island": False,
    }
    if evidence is None:
        return "evidence_missing", list(REQUIRED_EVIDENCE_STAGES), checks
    if evidence.get("schema") != EVIDENCE_SCHEMA:
        raise ValueError(f"wrong downstream evidence schema: {requirement['requirement_id']}")
    _require_contract(evidence, name=f"evidence {requirement['requirement_id']}")
    for field in (
        "requirement_id",
        "source_id",
        "partition",
        "start_frame",
        "end_frame",
        "source_audio_sha256",
    ):
        expected = requirement[field]
        actual = evidence.get(field)
        if str(actual) != str(expected):
            raise ValueError(
                f"downstream evidence identity mismatch for {requirement['requirement_id']}: {field}"
            )
    if expected_checkpoint_shas is None:
        raise ValueError("expected checkpoint SHAs are required with downstream evidence")
    for stage, expected_sha in expected_checkpoint_shas.items():
        actual_sha = _require_sha256(
            evidence.get(f"{stage}_checkpoint_sha256"),
            name=f"{stage}_checkpoint_sha256",
        )
        if actual_sha != expected_sha:
            raise ValueError(
                f"{stage} checkpoint SHA mismatch: {requirement['requirement_id']}"
            )

    missing = [
        stage
        for stage in REQUIRED_EVIDENCE_STAGES
        if not isinstance(evidence.get(stage), list) or not evidence.get(stage)
    ]
    if missing:
        return "evidence_missing", missing, checks

    start = int(requirement["start_frame"])
    end = int(requirement["end_frame"])
    scorer_islands = list(evidence["scorer_candidate_islands"])
    proposals = list(evidence["proposal_candidates"])
    split_events = list(evidence["split_events"])
    sub_islands = list(evidence["provisional_sub_islands"])
    cueqc = list(evidence["cueqc_decisions"])
    checks["scorer_envelope_kept"] = any(
        int(island.get("start_frame", 10**9)) < start
        and int(island.get("end_frame", -1)) > end
        for island in scorer_islands
    )
    checks["proposal_at_both_edges"] = _candidate_near(
        proposals, start, tolerance=tolerance_frames
    ) and _candidate_near(proposals, end, tolerance=tolerance_frames)
    checks["split_cut_at_both_edges"] = _event_near(
        split_events, start, tolerance=tolerance_frames
    ) and _event_near(split_events, end, tolerance=tolerance_frames)
    isolated_ids = {
        str(island.get("island_id") or "")
        for island in sub_islands
        if abs(int(island.get("start_frame", -10**9)) - start) <= tolerance_frames
        and abs(int(island.get("end_frame", -10**9)) - end) <= tolerance_frames
        and str(island.get("island_id") or "")
    }
    checks["provisional_background_island"] = bool(isolated_ids)
    checks["cueqc_dropped_island"] = any(
        str(decision.get("island_id") or "") in isolated_ids
        and str(decision.get("argmax_label") or decision.get("decision") or "")
        == "drop"
        for decision in cueqc
    )
    status = (
        "downstream_isolation_demonstrated"
        if all(checks.values())
        else "isolation_not_demonstrated"
    )
    return status, [], checks


def _render_page(rows: Sequence[dict[str, Any]]) -> str:
    cards: list[str] = []
    for index, row in enumerate(rows):
        source_id = html.escape(str(row["source_id"]))
        status = html.escape(str(row["evidence_status"]))
        missing = ", ".join(row["missing_stages"]) or "none"
        cards.append(
            f"""<article><h2>{index + 1}. {source_id}</h2>
<p><code>{row['start_s']:.2f}-{row['end_s']:.2f}s</code> · <b>{status}</b></p>
<button onclick="playRange({index},{row['start_s']},{row['end_s']})">播放精确区间</button>
<button onclick="stopAudio({index})">停止</button>
<audio id="audio-{index}" preload="metadata" src="{html.escape(row['audio_url'])}"></audio>
<p>缺失阶段：<code>{html.escape(missing)}</code></p>
<pre>{html.escape(json.dumps(row['checks'], ensure_ascii=False, sort_keys=True))}</pre></article>"""
        )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Downstream isolation evidence</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;background:#f2f5f7;color:#17212b}}.contract,article{{background:#fff;border:1px solid #cbd5dd;border-radius:10px;padding:14px;margin:12px 0}}button{{padding:7px 10px;margin-right:6px}}code,pre{{background:#edf1f4;padding:2px 5px;border-radius:4px}}pre{{overflow:auto}}</style></head><body>
<h1>Scorer v11 downstream isolation evidence</h1><section class="contract"><p>本页不要求重新听感裁决。它只核对已经人工确认的独立背景，是否在同一 source/audio/checkpoint 绑定下被 Proposal 提候选、Split 双边切开、形成 provisional sub-island，并由 CueQC 二分类 argmax 删除。</p><p><b>evidence_missing</b> 不等于 Scorer 错误；在证据齐备前对应帧保持 <code>unsure=-100</code>。播放器只播放精确区间，不附加上下文。</p></section>{''.join(cards)}
<script>const timers=new Map();function stopAudio(i){{const a=document.getElementById('audio-'+i);if(timers.has(i)){{clearInterval(timers.get(i));timers.delete(i)}}a.pause()}}function playRange(i,s,e){{stopAudio(i);const a=document.getElementById('audio-'+i);a.currentTime=s;a.play();const t=setInterval(()=>{{if(a.currentTime>=e||a.ended)stopAudio(i)}},20);timers.set(i,t)}}</script></body></html>"""


def compile_downstream_isolation(
    *,
    heldout_verdicts: Path,
    review_summary: Path,
    bridge_verdicts: Path,
    selection: Path,
    output_dir: Path,
    downstream_evidence: Path | None = None,
    expected_checkpoint_shas: dict[str, str] | None = None,
    tolerance_frames: int = 15,
    verify_audio: bool = True,
    audio_root: Path | None = None,
) -> dict[str, Any]:
    heldout_verdicts = heldout_verdicts.resolve()
    review_summary = review_summary.resolve()
    bridge_verdicts = bridge_verdicts.resolve()
    selection = selection.resolve()
    if downstream_evidence is not None:
        downstream_evidence = downstream_evidence.resolve()
    if audio_root is not None:
        audio_root = audio_root.resolve()
        if not audio_root.is_dir():
            raise FileNotFoundError(audio_root)
    for path in (heldout_verdicts, review_summary, bridge_verdicts, selection):
        if not path.is_file():
            raise FileNotFoundError(path)
    if tolerance_frames < 0:
        raise ValueError("tolerance_frames must be non-negative")

    summary = _read_json(review_summary)
    if summary.get("schema") != DUAL_REVIEW_SUMMARY_SCHEMA:
        raise ValueError("wrong dual review summary schema")
    _require_contract(summary, name="dual review summary")
    bound_human = _bound_summary_path(
        summary,
        field="human_verdicts",
        sha_field="human_verdicts_sha256",
        owner=review_summary,
    )
    if bound_human != heldout_verdicts:
        raise ValueError("dual review is bound to a different held-out verdict file")
    audit_manifest = _bound_summary_path(
        summary,
        field="manifest",
        sha_field="manifest_sha256",
        owner=review_summary,
    )
    bound_bridge = _resolve(str(summary.get("manual_verdicts") or ""), owner=review_summary)
    if bound_bridge != bridge_verdicts:
        raise ValueError("dual review is bound to a different bridge verdict file")

    heldout_rows = _index(
        _read_jsonl(heldout_verdicts), "source_id", name="held-out verdict"
    )
    bridge_rows = _index(
        _read_jsonl(bridge_verdicts), "gap_id", name="bridge verdict"
    )
    selection_rows = _index(
        _read_jsonl(selection), "gap_id", name="isolation selection"
    )
    audit_rows = _index(
        _read_jsonl(audit_manifest), "source_id", name="held-out audit manifest"
    )
    if not selection_rows:
        raise ValueError("at least one downstream isolation selection is required")

    evidence_rows: dict[str, dict[str, Any]] = {}
    if downstream_evidence is not None:
        if not downstream_evidence.is_file():
            raise FileNotFoundError(downstream_evidence)
        evidence_rows = _index(
            _read_jsonl(downstream_evidence),
            "requirement_id",
            name="downstream evidence",
        )
        extra_evidence = sorted(set(evidence_rows) - set(selection_rows))
        if extra_evidence:
            raise ValueError(f"downstream evidence has unselected requirements: {extra_evidence}")
        if expected_checkpoint_shas is None:
            raise ValueError("expected checkpoint SHAs are required with downstream evidence")
        expected_checkpoint_shas = {
            stage: _require_sha256(value, name=f"expected {stage} checkpoint SHA")
            for stage, value in expected_checkpoint_shas.items()
        }
        if set(expected_checkpoint_shas) != {"scorer", "proposal", "split", "cueqc"}:
            raise ValueError("expected checkpoint SHAs must bind scorer/proposal/split/cueqc")
    elif expected_checkpoint_shas:
        raise ValueError("checkpoint SHAs cannot be supplied without downstream evidence")

    heldout_sha = _sha256(heldout_verdicts)
    bridge_sha = _sha256(bridge_verdicts)
    selection_sha = _sha256(selection)
    requirements: list[dict[str, Any]] = []
    audit_results: list[dict[str, Any]] = []
    requirements_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    audio_by_source: dict[str, Path] = {}
    for gap_id, selected in selection_rows.items():
        if selected.get("schema") != SELECTION_SCHEMA:
            raise ValueError(f"wrong downstream isolation selection schema: {gap_id}")
        _require_contract(selected, name=f"selection {gap_id}")
        if selected.get("decision") != "independent_background_needs_downstream_isolation":
            raise ValueError(f"wrong downstream isolation decision: {gap_id}")
        bridge = bridge_rows.get(gap_id)
        if bridge is None:
            raise ValueError(f"selected bridge verdict is missing: {gap_id}")
        if bridge.get("schema") != BRIDGE_VERDICT_SCHEMA or not bool(bridge.get("complete")):
            raise ValueError(f"selected bridge verdict is incomplete: {gap_id}")
        _require_contract(bridge, name=f"bridge verdict {gap_id}")
        if bridge.get("content_verdict") != "no_semantic_dialogue" or bridge.get(
            "envelope_verdict"
        ) != "overmerged_independent_background":
            raise ValueError(f"selected gap is not confirmed independent background: {gap_id}")
        for field in ("source_id", "partition", "start_frame", "end_frame"):
            if str(selected.get(field)) != str(bridge.get(field)):
                raise ValueError(f"selection/bridge identity mismatch for {gap_id}: {field}")
        source_id = str(bridge["source_id"])
        heldout = heldout_rows.get(source_id)
        audit_item = audit_rows.get(source_id)
        if heldout is None or audit_item is None:
            raise ValueError(f"selected gap source is absent from held-out truth: {gap_id}")
        if heldout.get("schema") != HELDOUT_VERDICT_SCHEMA or audit_item.get(
            "schema"
        ) != HELDOUT_AUDIT_ITEM_SCHEMA:
            raise ValueError(f"wrong held-out source schema: {source_id}")
        for item_name, item in (("held-out", heldout), ("audit", audit_item)):
            _require_contract(item, name=f"{item_name} {source_id}")
        labels = _validate_spans(heldout)
        start = int(bridge["start_frame"])
        end = int(bridge["end_frame"])
        if not 0 <= start < end <= len(labels) or set(labels[start:end]) != {
            "outside_candidate"
        }:
            raise ValueError(f"selected gap is not source background truth: {gap_id}")
        if str(heldout.get("partition")) != str(bridge.get("partition")) or str(
            audit_item.get("partition")
        ) != str(bridge.get("partition")):
            raise ValueError(f"selected gap partition mismatch: {gap_id}")
        audio = _resolve_audit_audio(
            str(audit_item.get("audio") or ""),
            audit_manifest=audit_manifest,
            audio_root=audio_root,
        )
        if not audio.is_file():
            raise FileNotFoundError(audio)
        audio_sha = _require_sha256(
            audit_item.get("audio_sha256"), name=f"audio SHA {source_id}"
        )
        if verify_audio and _sha256(audio) != audio_sha:
            raise ValueError(f"held-out audit audio SHA mismatch: {source_id}")
        audio_by_source[source_id] = audio
        requirement = {
            "schema": REQUIREMENT_SCHEMA,
            "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
            "requirement_id": gap_id,
            "gap_id": gap_id,
            "source_id": source_id,
            "partition": str(bridge["partition"]),
            "start_frame": start,
            "end_frame": end,
            "start_s": round(start * FRAME_HOP_S, 6),
            "end_s": round(end * FRAME_HOP_S, 6),
            "duration_s": round((end - start) * FRAME_HOP_S, 6),
            "source_audio": _display(audio),
            "source_audio_sha256": audio_sha,
            "duty_label": "independent_background_needs_downstream_isolation",
            "scorer_label_without_bound_evidence": "unsure",
            "scorer_training_label_without_bound_evidence": -100,
            "raw_bridge_verdict": str(bridge.get("verdict") or ""),
            "raw_heldout_verdicts_sha256": heldout_sha,
            "raw_bridge_verdicts_sha256": bridge_sha,
            "selection_sha256": selection_sha,
        }
        status, missing_stages, checks = _evaluate_evidence(
            requirement,
            evidence_rows.get(gap_id),
            expected_checkpoint_shas=expected_checkpoint_shas,
            tolerance_frames=tolerance_frames,
        )
        requirement["evidence_status"] = status
        requirement["missing_stages"] = missing_stages
        requirement["scorer_canonical_label"] = (
            "inside_candidate"
            if status == "downstream_isolation_demonstrated"
            else "unsure"
        )
        requirement["scorer_training_label"] = (
            1 if requirement["scorer_canonical_label"] == "inside_candidate" else -100
        )
        requirements.append(requirement)
        requirements_by_source[source_id].append(requirement)
        audit_results.append(
            {
                "schema": AUDIT_SCHEMA,
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                **{
                    field: requirement[field]
                    for field in (
                        "requirement_id",
                        "source_id",
                        "partition",
                        "start_frame",
                        "end_frame",
                        "start_s",
                        "end_s",
                        "duration_s",
                        "source_audio",
                        "source_audio_sha256",
                        "duty_label",
                        "evidence_status",
                        "missing_stages",
                        "scorer_canonical_label",
                        "scorer_training_label",
                    )
                },
                "checks": checks,
                "audio_url": _audio_url(audio),
                "evidence_file": (
                    _display(downstream_evidence) if downstream_evidence is not None else None
                ),
                "evidence_file_sha256": (
                    _sha256(downstream_evidence)
                    if downstream_evidence is not None
                    else None
                ),
                "expected_checkpoint_shas": expected_checkpoint_shas,
            }
        )

    responsibility_rows: list[dict[str, Any]] = []
    canonical_counts: Counter[str] = Counter()
    for source_id, heldout in heldout_rows.items():
        if heldout.get("schema") != HELDOUT_VERDICT_SCHEMA:
            raise ValueError(f"wrong held-out verdict schema: {source_id}")
        _require_contract(heldout, name=f"held-out verdict {source_id}")
        labels = _validate_spans(heldout)
        source_requirements = sorted(
            requirements_by_source.get(source_id, ()), key=lambda row: row["start_frame"]
        )
        cursor = -1
        for requirement in source_requirements:
            start = int(requirement["start_frame"])
            end = int(requirement["end_frame"])
            if start < cursor:
                raise ValueError(f"overlapping downstream requirements: {source_id}")
            labels[start:end] = [str(requirement["scorer_canonical_label"])] * (end - start)
            cursor = end
        canonical_counts.update(labels)
        responsibility_rows.append(
            {
                "schema": RESPONSIBILITY_VERDICT_SCHEMA,
                "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
                "source_id": source_id,
                "partition": str(heldout.get("partition") or ""),
                "frame_count": int(heldout["frame_count"]),
                "frame_hop_s": FRAME_HOP_S,
                "reviewed_full_source": True,
                "verdict": "complete_with_scorer_responsibility_mapping",
                "spans": _label_runs(labels),
                "review_provenance": "human_full_source_plus_downstream_isolation_v1",
                "human_review_required": False,
                "training_manifest_allowed": True,
                "unsure_training_label": -100,
                "downstream_isolation_unsure_only": bool(source_requirements)
                and all(
                    requirement["scorer_canonical_label"] == "unsure"
                    for requirement in source_requirements
                ),
                "downstream_isolation_requirement_ids": [
                    requirement["requirement_id"] for requirement in source_requirements
                ],
                "raw_heldout_verdicts": _display(heldout_verdicts),
                "raw_heldout_verdicts_sha256": heldout_sha,
                "downstream_isolation_selection": _display(selection),
                "downstream_isolation_selection_sha256": selection_sha,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    requirements_path = output_dir / "downstream_isolation_requirements.jsonl"
    audit_path = output_dir / "downstream_isolation_audit.jsonl"
    responsibility_path = output_dir / "responsibility_verdicts.jsonl"
    _write_jsonl(requirements_path, requirements)
    _write_jsonl(audit_path, audit_results)
    _write_jsonl(responsibility_path, responsibility_rows)
    (output_dir / "index.html").write_text(
        _render_page(audit_results), encoding="utf-8"
    )
    status_counts = Counter(row["evidence_status"] for row in requirements)
    result = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "heldout_verdicts": _display(heldout_verdicts),
        "heldout_verdicts_sha256": heldout_sha,
        "review_summary": _display(review_summary),
        "review_summary_sha256": _sha256(review_summary),
        "bridge_verdicts": _display(bridge_verdicts),
        "bridge_verdicts_sha256": bridge_sha,
        "selection": _display(selection),
        "selection_sha256": selection_sha,
        "downstream_evidence": (
            _display(downstream_evidence) if downstream_evidence is not None else None
        ),
        "downstream_evidence_sha256": (
            _sha256(downstream_evidence) if downstream_evidence is not None else None
        ),
        "expected_checkpoint_shas": expected_checkpoint_shas,
        "audio_root": _display(audio_root) if audio_root is not None else None,
        "tolerance_frames": tolerance_frames,
        "requirement_count": len(requirements),
        "source_count": len(requirements_by_source),
        "evidence_status_counts": dict(sorted(status_counts.items())),
        "all_requirements_evidence_missing": bool(requirements)
        and status_counts["evidence_missing"] == len(requirements),
        "requirements": _display(requirements_path),
        "requirements_sha256": _sha256(requirements_path),
        "audit": _display(audit_path),
        "audit_sha256": _sha256(audit_path),
        "responsibility_verdicts": _display(responsibility_path),
        "responsibility_verdicts_sha256": _sha256(responsibility_path),
        "responsibility_verdict_schema": RESPONSIBILITY_VERDICT_SCHEMA,
        "canonical_frame_counts": dict(sorted(canonical_counts.items())),
        "unsure_training_label": -100,
        "raw_manual_verdicts_modified": False,
        "training_manifest_allowed": True,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout-verdicts", required=True)
    parser.add_argument("--review-summary", required=True)
    parser.add_argument("--bridge-verdicts", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--downstream-evidence")
    parser.add_argument("--expected-scorer-checkpoint-sha256")
    parser.add_argument("--expected-proposal-checkpoint-sha256")
    parser.add_argument("--expected-split-checkpoint-sha256")
    parser.add_argument("--expected-cueqc-checkpoint-sha256")
    parser.add_argument("--tolerance-frames", type=int, default=15)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audio-root")
    parser.add_argument("--skip-audio-content-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    expected_values = {
        "scorer": args.expected_scorer_checkpoint_sha256,
        "proposal": args.expected_proposal_checkpoint_sha256,
        "split": args.expected_split_checkpoint_sha256,
        "cueqc": args.expected_cueqc_checkpoint_sha256,
    }
    present_expected = {key: value for key, value in expected_values.items() if value}
    return compile_downstream_isolation(
        heldout_verdicts=Path(args.heldout_verdicts),
        review_summary=Path(args.review_summary),
        bridge_verdicts=Path(args.bridge_verdicts),
        selection=Path(args.selection),
        output_dir=Path(args.output_dir),
        downstream_evidence=(
            Path(args.downstream_evidence) if args.downstream_evidence else None
        ),
        expected_checkpoint_shas=present_expected or None,
        tolerance_frames=args.tolerance_frames,
        verify_audio=not args.skip_audio_content_check,
        audio_root=Path(args.audio_root) if args.audio_root else None,
    )


if __name__ == "__main__":
    main()
