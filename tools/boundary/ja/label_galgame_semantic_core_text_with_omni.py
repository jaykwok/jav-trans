#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.asr.cueqc.label_pre_asr_with_omni import (  # noqa: E402
    DEFAULT_API_KEY_ENV_CANDIDATES,
    DEFAULT_BASE_URL_ENV_CANDIDATES,
    DEFAULT_ENV_FILE,
    extract_json_object,
    first_env_value,
    load_env_file,
)
from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402
from tools.boundary.ja.build_galgame_synthetic_timeline import (  # noqa: E402
    load_excluded_source_audio_ids,
    load_manifest_rows,
    valid_source_rows,
)


SCHEMA = "galgame_semantic_core_text_teacher_v1"
SUMMARY_SCHEMA = "galgame_semantic_core_text_summary_v1"
PROMPT_VERSION = "galgame_fullclip_semantic_core_text_batch_v1"
DEFAULT_MODEL = "qwen3.5-omni-plus"
LABELS = ("all_semantic", "contains_nonsemantic", "unsure")


SYSTEM_PROMPT = """你是日语 Galgame 语音 clip 的文本语义分类 teacher。输入是一批可信参考文本；音频与文本逐条对应，但本任务只需判断整条 full clip 能否作为纯 semantic speech core。

每项只能返回以下一个标签：
- all_semantic：从头到尾全部是有语言语义、可作为字幕的日语词、助词、应答词或完整句子。省略号、标点、口吃不影响；但整条必须没有非语义片段。
- contains_nonsemantic：任意位置包含喘息、呻吟、亲吻/舔舐声、笑声、无意义叫声、短促非词拟声、纯拉长音或其他无字幕价值发声。只要混入一小段也用此标签。
- unsure：无法可靠判断某段究竟是有意义词语还是非词发声。

不要根据 id 猜测，不要改写文本，不要输出时间轴、理由、Markdown 或额外字段。只输出：
{"items":[{"id":"c000","label":"all_semantic|contains_nonsemantic|unsure"}]}
必须逐项返回，顺序与输入完全一致。"""


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_excluded_candidate_audio_ids(paths: list[str] | None) -> set[str]:
    excluded: set[str] = set()
    for manifest_path in paths or []:
        for row in load_manifest_rows(Path(manifest_path)):
            audio_id = str(
                row.get("audio_id")
                or (Path(str(row["audio"])).stem if row.get("audio") else "")
            )
            if audio_id:
                excluded.add(audio_id)
    return excluded


def select_candidates(
    rows: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    excluded_audio_ids: set[str],
) -> list[dict[str, Any]]:
    available = [
        row
        for row in rows
        if str(row.get("audio_id") or "") not in excluded_audio_ids
        and str(row.get("text") or "").strip()
    ]
    if len(available) < count:
        raise ValueError(f"semantic core candidate inventory {len(available)} < {count}")
    rng = np.random.default_rng(seed)
    indexes = rng.choice(len(available), size=count, replace=False)
    return [available[int(index)] for index in indexes]


def build_prompt(batch: list[dict[str, Any]]) -> tuple[str, dict[str, dict[str, Any]]]:
    mapping: dict[str, dict[str, Any]] = {}
    items = []
    for index, row in enumerate(batch):
        short_id = f"c{index:03d}"
        mapping[short_id] = row
        items.append({"id": short_id, "text": str(row["text"])})
    return json.dumps({"items": items}, ensure_ascii=False, separators=(",", ":")), mapping


def validate_response(
    parsed: dict[str, Any], mapping: dict[str, dict[str, Any]]
) -> list[tuple[dict[str, Any], str]]:
    items = parsed.get("items")
    if not isinstance(items, list):
        raise ValueError("response.items must be a list")
    expected_ids = list(mapping)
    actual_ids = [str(item.get("id") or "") for item in items if isinstance(item, dict)]
    if actual_ids != expected_ids:
        raise ValueError("response ids/order do not exactly match request")
    result: list[tuple[dict[str, Any], str]] = []
    for item in items:
        label = str(item.get("label") or "")
        if label not in LABELS:
            raise ValueError(f"unsupported semantic core label: {label!r}")
        result.append((mapping[str(item["id"])], label))
    return result


def call_text_teacher(
    *,
    model: str,
    api_key: str,
    base_url: str,
    timeout_s: float,
    prompt: str,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from openai import OpenAI

    kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout_s}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    stream = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        modalities=["text"],
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"enable_thinking": False},
    )
    text_parts: list[str] = []
    usage: dict[str, Any] | None = None
    model_names: set[str] = set()
    for chunk in stream:
        payload = chunk.model_dump(mode="json")
        if payload.get("usage"):
            usage = dict(payload["usage"])
        if payload.get("model"):
            model_names.add(str(payload["model"]))
        for choice in getattr(chunk, "choices", None) or []:
            content = getattr(choice.delta, "content", None) or ""
            if content:
                text_parts.append(content)
    content = "".join(text_parts)
    return extract_json_object(content), {
        "content": content,
        "usage": usage,
        "response_models": sorted(model_names),
    }


def is_provider_data_inspection_rejection(error: Exception) -> bool:
    value = str(error).lower()
    return "data_inspection_failed" in value


def teacher_label_row(
    *,
    source: dict[str, Any],
    label: str,
    model: str,
    decision_source: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "audio_id": str(source["audio_id"]),
        "audio": str(source["audio"]),
        "duration_s": float(source["duration_s"]),
        "reference_text": str(source["text"]),
        "source": str(source.get("input") or source.get("source") or ""),
        "label": label,
        "decision_source": decision_source,
    }


def select_audit_rows(rows: list[dict[str, Any]], *, count: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for label in LABELS:
        group = [row for row in rows if row["label"] == label]
        if group:
            row = group[int(rng.integers(0, len(group)))]
            selected.append(row)
            used.add(str(row["audio_id"]))
    remaining = [row for row in rows if str(row["audio_id"]) not in used]
    if remaining and len(selected) < count:
        indexes = rng.choice(
            len(remaining), size=min(count - len(selected), len(remaining)), replace=False
        )
        selected.extend(remaining[int(index)] for index in indexes)
    if len(selected) != count:
        raise ValueError(f"semantic core audit requires {count} rows; got {len(selected)}")
    return selected


def build_audit(*, rows: list[dict[str, Any]], output_dir: Path) -> Path:
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    payload_rows = []
    for row in rows:
        source = Path(str(row["audio"]))
        target = audio_dir / f"{row['audio_id']}{source.suffix.lower()}"
        shutil.copyfile(source, target)
        payload_rows.append(
            {
                **row,
                "audio": target.relative_to(output_dir).as_posix(),
            }
        )
    payload = json.dumps(payload_rows, ensure_ascii=False).replace("</", "<\\/")
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Galgame semantic core text teacher smoke</title><style>
body{{margin:0;background:#0d1117;color:#e6edf3;font-family:system-ui}}header{{position:sticky;top:0;background:#161b22;padding:12px 18px}}main{{max-width:1000px;margin:auto;padding:16px}}article{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;margin:14px 0}}audio{{width:100%}}button{{margin:4px;padding:7px 12px;background:#21262d;color:#e6edf3;border:1px solid #484f58;border-radius:6px}}button.active{{background:#1f6feb}}textarea{{width:98%;min-height:55px;background:#0d1117;color:#e6edf3}}
</style></head><body><header><strong>Galgame full-clip semantic core · text teacher smoke</strong> <button id="save">保存</button> <span id="status"></span></header><main><p>检查 teacher 对整条文本的分类是否正确。<b>all_semantic</b> 只有整条从头到尾都可作为字幕语义时才成立；混有喘息、呻吟、亲吻/舔舐拟声或非词发声必须是 <b>contains_nonsemantic</b>。</p><div id="list"></div></main><script>
const rows={payload};const key='galgame-semantic-core-text-teacher-v1:'+location.pathname;const ann=JSON.parse(localStorage.getItem(key)||'{{}}');function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}function ensure(id){{ann[id]??={{verdict:'',note:''}};return ann[id];}}function persist(){{localStorage.setItem(key,JSON.stringify(ann));render();}}function render(){{const root=document.getElementById('list');root.innerHTML='';for(const r of rows){{const a=ensure(r.audio_id),card=document.createElement('article');card.innerHTML=`<h2>${{esc(r.audio_id)}}</h2><p><b>teacher：</b>${{esc(r.label)}} · ${{Number(r.duration_s).toFixed(3)}}s</p><p>${{esc(r.reference_text)}}</p><audio controls preload="metadata" src="${{esc(r.audio)}}"></audio><div>${{['correct','wrong','unsure'].map(v=>`<button data-v="${{v}}" class="${{a.verdict===v?'active':''}}">${{{{correct:'正确',wrong:'错误',unsure:'不确定'}}[v]}}</button>`).join('')}}</div><textarea placeholder="备注">${{esc(a.note)}}</textarea>`;card.querySelectorAll('[data-v]').forEach(b=>b.onclick=()=>{{a.verdict=b.dataset.v;a.updated_at=new Date().toISOString();persist();}});card.querySelector('textarea').onchange=e=>{{a.note=e.target.value;a.updated_at=new Date().toISOString();persist();}};root.appendChild(card);}}document.getElementById('status').textContent=`完成 ${{rows.filter(r=>ensure(r.audio_id).verdict).length}}/${{rows.length}}`;}}
document.getElementById('save').onclick=async()=>{{const content=rows.map(r=>{{const a=ensure(r.audio_id);return JSON.stringify({{schema:'galgame_semantic_core_text_manual_verdict_v1',audio_id:r.audio_id,teacher_label:r.label,verdict:a.verdict||'unreviewed',note:a.note||'',updated_at:a.updated_at||new Date().toISOString()}});}}).join('\\n')+'\\n';const res=await fetch('/__audit_api__/save-labels',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{href:location.pathname,filename:'manual_verdicts.jsonl',content}})}});const out=await res.json();document.getElementById('status').textContent=out.ok?'已保存到 '+out.path:'保存失败: '+out.error;}};render();
</script></body></html>"""
    page = output_dir / "index.html"
    page.write_text(html, encoding="utf-8")
    update_audit_entrypoints(latest_html=page, title="Galgame semantic core text teacher smoke")
    (output_dir / "audit_items.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in payload_rows),
        encoding="utf-8",
    )
    return page


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_env_file(Path(args.env_file))
    _key_env, api_key = first_env_value(DEFAULT_API_KEY_ENV_CANDIDATES)
    _url_env, base_url = first_env_value(DEFAULT_BASE_URL_ENV_CANDIDATES)
    _model_env, configured_model = first_env_value(("OMNI_MODEL", "QWEN_OMNI_MODEL"))
    model = args.model or configured_model or DEFAULT_MODEL
    if not api_key:
        raise ValueError("Omni API key is not configured")
    source_rows, skipped = valid_source_rows(load_manifest_rows(Path(args.manifest)))
    excluded = load_excluded_source_audio_ids(args.exclude_source_manifest)
    excluded_candidate_ids = load_excluded_candidate_audio_ids(
        args.exclude_candidate_manifest
    )
    excluded.update(excluded_candidate_ids)
    excluded.update(str(value) for value in (args.exclude_source_audio_id or []))
    candidates = select_candidates(
        source_rows,
        count=args.candidate_count,
        seed=args.seed,
        excluded_audio_ids=excluded,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "candidates.jsonl"
    candidates_path.write_text(
        "".join(
            json.dumps(
                {
                    "audio_id": row["audio_id"],
                    "audio": row["audio"],
                    "duration_s": row["duration_s"],
                    "reference_text": row["text"],
                    "source": row.get("input") or row.get("source") or "",
                },
                ensure_ascii=False,
            )
            + "\n"
            for row in candidates
        ),
        encoding="utf-8",
    )
    labels_path = output_dir / "labels.jsonl"
    raw_path = output_dir / "raw_responses.jsonl"
    existing = {str(row["audio_id"]): row for row in _rows(labels_path)}
    pending = [row for row in candidates if str(row["audio_id"]) not in existing]
    for batch_index, offset in enumerate(range(0, len(pending), args.batch_size)):
        batch = pending[offset : offset + args.batch_size]
        prompt, mapping = build_prompt(batch)
        last_error: Exception | None = None
        for attempt in range(1, args.max_attempts + 1):
            try:
                parsed, raw = call_text_teacher(
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    timeout_s=args.timeout_s,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                )
                validated = validate_response(parsed, mapping)
                _append(
                    raw_path,
                    {
                        "batch_index": batch_index,
                        "attempt": attempt,
                        "model": model,
                        **raw,
                    },
                )
                for source, label in validated:
                    row = teacher_label_row(
                        source=source,
                        label=label,
                        model=model,
                        decision_source="teacher_response",
                    )
                    _append(labels_path, row)
                    existing[row["audio_id"]] = row
                print(
                    f"semantic_core_text={len(existing)}/{len(candidates)} "
                    f"batch={batch_index + 1} model={model}",
                    flush=True,
                )
                last_error = None
                break
            except Exception as error:  # noqa: BLE001
                last_error = error
                _append(
                    raw_path,
                    {
                        "batch_index": batch_index,
                        "attempt": attempt,
                        "model": model,
                        "error": repr(error),
                    },
                )
                if is_provider_data_inspection_rejection(error):
                    for source in batch:
                        row = teacher_label_row(
                            source=source,
                            label="unsure",
                            model=model,
                            decision_source="provider_data_inspection_rejected",
                        )
                        _append(labels_path, row)
                        existing[row["audio_id"]] = row
                    print(
                        f"semantic_core_text={len(existing)}/{len(candidates)} "
                        f"batch={batch_index + 1} model={model} "
                        "provider_data_inspection_rejected=true",
                        flush=True,
                    )
                    last_error = None
                    break
                if attempt == args.max_attempts and isinstance(error, ValueError):
                    for source in batch:
                        row = teacher_label_row(
                            source=source,
                            label="unsure",
                            model=model,
                            decision_source="teacher_schema_failed_after_retries",
                        )
                        _append(labels_path, row)
                        existing[row["audio_id"]] = row
                    print(
                        f"semantic_core_text={len(existing)}/{len(candidates)} "
                        f"batch={batch_index + 1} model={model} "
                        "teacher_schema_failed_after_retries=true",
                        flush=True,
                    )
                    last_error = None
                    break
                if attempt < args.max_attempts:
                    time.sleep(min(8.0, float(attempt)))
        if last_error is not None:
            raise RuntimeError(f"semantic core text batch {batch_index} failed") from last_error
        if args.request_interval_s > 0:
            time.sleep(args.request_interval_s)
    labels = _rows(labels_path)
    counts = Counter(str(row["label"]) for row in labels)
    decision_counts = Counter(str(row.get("decision_source") or "legacy") for row in labels)
    audit_rows = select_audit_rows(labels, count=args.audit_count, seed=args.seed)
    page = build_audit(rows=audit_rows, output_dir=Path(args.audit_output_dir))
    summary = {
        "schema": SUMMARY_SCHEMA,
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "candidate_count": len(candidates),
        "labeled_count": len(labels),
        "label_counts": {label: counts[label] for label in LABELS},
        "decision_source_counts": dict(sorted(decision_counts.items())),
        "excluded_source_audio_id_count": len(excluded),
        "excluded_candidate_audio_id_count": len(excluded_candidate_ids),
        "invalid_source_count": len(skipped),
        "labels": str(labels_path),
        "audit_page": str(page),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-label Galgame full-clip semantic core texts.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit-output-dir", required=True)
    parser.add_argument("--candidate-count", type=int, default=100)
    parser.add_argument("--audit-count", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--exclude-source-manifest", action="append")
    parser.add_argument(
        "--exclude-candidate-manifest",
        action="append",
        help=(
            "JSON/JSONL candidate or source-sample manifest whose top-level "
            "audio_id (or audio filename stem) values are excluded. Repeatable."
        ),
    )
    parser.add_argument("--exclude-source-audio-id", action="append")
    parser.add_argument("--model", default="")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--request-interval-s", type=float, default=0.5)
    args = parser.parse_args()
    if args.candidate_count <= 0 or args.audit_count <= 0 or args.batch_size <= 0:
        parser.error("counts and batch size must be positive")
    if args.audit_count > args.candidate_count:
        parser.error("--audit-count cannot exceed --candidate-count")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False))
