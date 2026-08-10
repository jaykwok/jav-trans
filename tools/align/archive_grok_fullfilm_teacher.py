"""Archive a completed Grok full-film STT run as a training-data source.

The paid provider responses and compiled absolute word times are preserved.  The
large source videos and derived audio chunks are not copied; source identities
and a PowerShell rebuild entry point make those chunks reproducible.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable


ARCHIVE_SCHEMA = "jav_grok_stt_frame_teacher_archive_v1"
FILM_SCHEMA = "jav_grok_stt_frame_teacher_film_v1"
CHUNK_SCHEMA = "jav_grok_stt_frame_teacher_chunk_v1"
DEFAULT_PARTITION = "diagnostic"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _required_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_run(source_dir: Path) -> dict[str, Any]:
    """Validate a completed run before anything is copied."""
    manifest = _read_json(_required_file(source_dir, "manifest.json"))
    summary = _read_json(_required_file(source_dir, "summary.json"))
    words = _read_jsonl(_required_file(source_dir, "grok.words.jsonl"))
    cuts = _read_jsonl(_required_file(source_dir, "grok.speaker_cuts.jsonl"))
    errors_path = _required_file(source_dir, "errors.json")
    errors = json.loads(errors_path.read_text(encoding="utf-8"))
    if not isinstance(errors, (dict, list)):
        raise ValueError(f"expected an error object or array: {errors_path}")

    films = manifest.get("films") or []
    chunks = manifest.get("chunks") or []
    if not films or not chunks:
        raise ValueError("full-film manifest must contain films and chunks")
    film_ids = {str(row["film_id"]) for row in films}
    if len(film_ids) != len(films):
        raise ValueError("duplicate film_id in manifest")
    chunk_ids = {str(row["chunk_id"]) for row in chunks}
    if len(chunk_ids) != len(chunks):
        raise ValueError("duplicate chunk_id in manifest")

    response_dir = source_dir / "responses"
    response_paths = sorted(response_dir.glob("*.json"))
    response_ids = {path.stem for path in response_paths}
    if response_ids != chunk_ids:
        missing = sorted(chunk_ids - response_ids)
        extra = sorted(response_ids - chunk_ids)
        raise ValueError(f"response/chunk mismatch: missing={missing} extra={extra}")
    chunks_by_id = {str(row["chunk_id"]): row for row in chunks}
    for path in response_paths:
        response = _read_json(path)
        if response.get("chunk") != chunks_by_id[path.stem]:
            raise ValueError(f"response does not match manifest chunk: {path}")

    durations = {str(row["film_id"]): float(row["duration_s"]) for row in films}
    previous_key: tuple[str, float, float] | None = None
    for row in words:
        film_id = str(row.get("film_id") or "")
        if film_id not in film_ids:
            raise ValueError(f"word references unknown film: {film_id}")
        start_s = float(row["start_s"])
        end_s = float(row["end_s"])
        if start_s < 0.0 or end_s < start_s or end_s > durations[film_id] + 1e-3:
            raise ValueError(f"word outside source duration: {row}")
        key = (film_id, start_s, end_s)
        if previous_key is not None and key < previous_key:
            raise ValueError("compiled words are not sorted by film and time")
        previous_key = key
    for row in cuts:
        film_id = str(row.get("film_id") or "")
        if film_id not in film_ids:
            raise ValueError(f"speaker cut references unknown film: {film_id}")
        cut_s = row.get("cut_s")
        if bool(row.get("accepted")) and (
            cut_s is None or not 0.0 <= float(cut_s) <= durations[film_id]
        ):
            raise ValueError(f"invalid accepted speaker cut: {row}")

    expected_words = int(summary.get("word_count") or 0)
    expected_cuts = int(summary.get("speaker_change_count") or 0)
    if expected_words != len(words) or expected_cuts != len(cuts):
        raise ValueError(
            "summary counts do not match compiled rows: "
            f"words={expected_words}/{len(words)} cuts={expected_cuts}/{len(cuts)}"
        )
    has_errors = (
        bool(errors)
        if isinstance(errors, list)
        else errors.get("errors") not in (None, [])
    )
    if has_errors:
        raise ValueError("source run contains provider errors")
    return {
        "manifest": manifest,
        "summary": summary,
        "words": words,
        "cuts": cuts,
        "response_paths": response_paths,
    }


def _source_records(
    films: list[dict[str, Any]], *, hash_sources: bool
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for film in films:
        source = Path(str(film["source"]))
        exists = source.is_file()
        record = {
            "film_id": str(film["film_id"]),
            "source_path": str(source),
            "source_available_at_archive": exists,
            "source_bytes": source.stat().st_size if exists else None,
            "source_sha256": _sha256(source) if exists and hash_sources else None,
            "duration_s": float(film["duration_s"]),
            "chunk_count": int(film["chunk_count"]),
        }
        records.append(record)
    return records


def _row_ranges(rows: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    ranges: dict[str, tuple[int, int]] = {}
    for index, row in enumerate(rows):
        film_id = str(row["film_id"])
        if film_id not in ranges:
            ranges[film_id] = (index, index + 1)
        else:
            ranges[film_id] = (ranges[film_id][0], index + 1)
    return ranges


def _render_readme(
    *,
    dataset_name: str,
    summary: dict[str, Any],
    source_records: list[dict[str, Any]],
    partition: str,
) -> str:
    film_lines = []
    for record in source_records:
        film = (summary.get("films") or {}).get(record["film_id"], {})
        film_lines.append(
            f"- `{record['film_id']}`：{film.get('word_count', 0):,} 个词时间，"
            f"{film.get('accepted_nonoverlap_speaker_cuts', 0):,} 个非重叠换人切点，"
            f"{record['duration_s'] / 3600.0:.3f} 小时"
        )
    return f"""# JAV Grok STT frame teacher v1

这是两部真实 JAV 全片的 Grok STT 词级时间轴与 speaker diarization 归档，供真实域
CTC/blank 帧监督和字幕边界实验使用。教师模型为 `{summary.get('model')}`。

## 当前状态

- 数据集：`{dataset_name}`
- 影片：{len(source_records)} 部，媒体总长 {sum(float(row['duration_s']) for row in source_records) / 3600.0:.3f} 小时
- 词时间：{int(summary.get('word_count') or 0):,} 条
- 非重叠换人切点：{int(summary.get('accepted_nonoverlap_speaker_cuts') or 0):,} 条
- OpenRouter 实收：${float(summary.get('provider_actual_cost_usd') or 0.0):.6f}
- 分区：`{partition}`；两片均参与过时间轴诊断，不是未观察的 held-out

{chr(10).join(film_lines)}

## 目录

- `teacher/grok.words.jsonl`：按影片和绝对源 PTS 排序的词级时间轴。
- `teacher/grok.speaker_cuts.jsonl`：请求内、相邻说话人不同且不重叠时的候选切点。
- `teacher/responses/`：53 个不可替代的付费 Grok 原始响应。
- `source_run/manifest.json`：原始 300 秒分块、5 秒 overlap 和源时间映射。
- `compiled/films.jsonl`：影片级监督身份、词/切点行范围和源媒体哈希。
- `compiled/chunks.jsonl`：53 个请求块到响应、源区间和可重建音频缓存的映射。
- `rebuild/source_films.json`：源媒体身份与重建几何。
- `rebuild/rebuild_audio.ps1`：校验源影片 SHA-256 后重建 MP3 分块，不调用 Grok。
- `archive_manifest.json`：归档统计及每个文件的 SHA-256。

## 监督边界

当前两片只归档为 `frame_only_candidate`，没有伪造 CTC 文本目标，也没有进入现役头训练。
speaker ID 仅在单个 Grok 请求内有效；speaker change 只能用作非重叠切点，不能当作
跨影片人物身份。重叠区域与 Grok 未返回词的安静区域不能直接视为可靠 blank。

现有训练器尚未实现 frame-only 行的 CTC loss mask，因此 `training_ready=false`。在该合同
落地前，不应把这些行以空文本方式送进 CTC loss。

## 重建音频缓存

在项目根目录执行：

```powershell
datasets\\train\\{dataset_name}\\rebuild\\rebuild_audio.ps1 `
  -OutputDir agents\\temp\\rebuilt-jav-grok-audio
```

源影片没有复制进数据集。脚本默认核对归档时记录的完整 SHA-256，并拒绝覆盖已有输出；
输出是可删除、可重建的派生缓存。
"""


def _render_rebuild_script() -> str:
    return r'''param(
    [Parameter(Mandatory = $true)]
    [string]$OutputDir,
    [switch]$SkipSourceHashCheck
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$datasetRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\..\..")).Path
$sourceManifest = Get-Content -LiteralPath (Join-Path $datasetRoot "rebuild\source_films.json") -Raw | ConvertFrom-Json
if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $resolvedOutput = [System.IO.Path]::GetFullPath($OutputDir)
}
else {
    $resolvedOutput = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputDir))
}
if (Test-Path -LiteralPath $resolvedOutput) {
    throw "Refusing to overwrite existing output: $resolvedOutput"
}

$videoArgs = @()
foreach ($film in $sourceManifest.films) {
    if (-not (Test-Path -LiteralPath $film.source_path -PathType Leaf)) {
        throw "Missing source video: $($film.source_path)"
    }
    if (-not $SkipSourceHashCheck) {
        if (-not $film.source_sha256) {
            throw "No archived source hash for $($film.film_id); pass -SkipSourceHashCheck only after manual verification"
        }
        $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $film.source_path).Hash.ToLowerInvariant()
        if ($actualHash -ne $film.source_sha256.ToLowerInvariant()) {
            throw "Source hash mismatch for $($film.film_id)"
        }
    }
    $videoArgs += @("--video", "$($film.film_id)=$($film.source_path)")
}

Push-Location $repoRoot
try {
    uv run python -m tools.omni.run_grok_stt_fullfilm @videoArgs `
        --output-dir $resolvedOutput `
        --model $sourceManifest.model `
        --chunk-s $sourceManifest.chunk_s `
        --overlap-s $sourceManifest.overlap_s `
        --price-per-hour-usd $sourceManifest.price_per_hour_usd `
        --max-cost-usd 10 `
        --prepare-only
}
finally {
    Pop-Location
}
'''


def archive_run(
    source_dir: Path,
    output_dir: Path,
    *,
    partition: str = DEFAULT_PARTITION,
    hash_sources: bool = False,
) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite archive: {output_dir}")
    validated = validate_run(source_dir)
    manifest = validated["manifest"]
    summary = validated["summary"]
    words = validated["words"]
    cuts = validated["cuts"]
    films = list(manifest["films"])
    chunks = list(manifest["chunks"])
    sources = _source_records(films, hash_sources=hash_sources)
    source_by_film = {row["film_id"]: row for row in sources}
    word_ranges = _row_ranges(words)
    cut_ranges = _row_ranges(cuts)
    word_counts = Counter(str(row["film_id"]) for row in words)
    accepted_cut_counts = Counter(
        str(row["film_id"]) for row in cuts if bool(row.get("accepted"))
    )

    (output_dir / "teacher" / "responses").mkdir(parents=True)
    (output_dir / "source_run").mkdir(parents=True)
    shutil.copy2(source_dir / "manifest.json", output_dir / "source_run" / "manifest.json")
    shutil.copy2(source_dir / "summary.json", output_dir / "teacher" / "summary.json")
    shutil.copy2(source_dir / "errors.json", output_dir / "teacher" / "errors.json")
    shutil.copy2(
        source_dir / "grok.words.jsonl", output_dir / "teacher" / "grok.words.jsonl"
    )
    shutil.copy2(
        source_dir / "grok.speaker_cuts.jsonl",
        output_dir / "teacher" / "grok.speaker_cuts.jsonl",
    )
    for response in validated["response_paths"]:
        shutil.copy2(response, output_dir / "teacher" / "responses" / response.name)

    film_rows: list[dict[str, Any]] = []
    for film in films:
        film_id = str(film["film_id"])
        source = source_by_film[film_id]
        word_start, word_end = word_ranges.get(film_id, (0, 0))
        cut_start, cut_end = cut_ranges.get(film_id, (0, 0))
        film_rows.append(
            {
                "schema": FILM_SCHEMA,
                "film_id": film_id,
                "partition": partition,
                "evaluation_eligible": False,
                "evaluation_exclusion_reason": "used_for_timeline_diagnosis_before_archive",
                "supervision_mode": "frame_only_candidate",
                "training_ready": False,
                "training_blocker": "frame_only_ctc_loss_mask_not_implemented",
                "source_video": source["source_path"],
                "source_available_at_archive": source["source_available_at_archive"],
                "source_bytes": source["source_bytes"],
                "source_sha256": source["source_sha256"],
                "duration_s": source["duration_s"],
                "chunk_count": source["chunk_count"],
                "word_count": word_counts[film_id],
                "accepted_nonoverlap_speaker_cuts": accepted_cut_counts[film_id],
                "word_rows": {
                    "path": "teacher/grok.words.jsonl",
                    "start": word_start,
                    "end_exclusive": word_end,
                },
                "speaker_cut_rows": {
                    "path": "teacher/grok.speaker_cuts.jsonl",
                    "start": cut_start,
                    "end_exclusive": cut_end,
                },
            }
        )
    _write_jsonl(output_dir / "compiled" / "films.jsonl", film_rows)

    chunk_rows = []
    for chunk in chunks:
        film_id = str(chunk["film_id"])
        source = source_by_film[film_id]
        chunk_rows.append(
            {
                "schema": CHUNK_SCHEMA,
                "chunk_id": str(chunk["chunk_id"]),
                "film_id": film_id,
                "partition": partition,
                "supervision_mode": "frame_only_candidate",
                "training_ready": False,
                "source_video": source["source_path"],
                "source_sha256": source["source_sha256"],
                "nominal_start_s": float(chunk["nominal_start_s"]),
                "nominal_end_s": float(chunk["nominal_end_s"]),
                "request_start_s": float(chunk["request_start_s"]),
                "request_end_s": float(chunk["request_end_s"]),
                "response": f"teacher/responses/{chunk['chunk_id']}.json",
                "rebuild_audio": f"cache/audio/{chunk['chunk_id']}.mp3",
                "audio_archived": False,
            }
        )
    _write_jsonl(output_dir / "compiled" / "chunks.jsonl", chunk_rows)

    rebuild_manifest = {
        "schema": "jav_grok_stt_audio_rebuild_v1",
        "model": manifest.get("model"),
        "chunk_s": float(manifest["chunk_s"]),
        "overlap_s": float(manifest["overlap_s"]),
        "price_per_hour_usd": float(manifest["price_per_hour_usd"]),
        "timeline_filter": manifest.get("timeline_filter"),
        "films": sources,
    }
    _write_json(output_dir / "rebuild" / "source_films.json", rebuild_manifest)
    (output_dir / "rebuild" / "rebuild_audio.ps1").write_text(
        _render_rebuild_script(), encoding="utf-8", newline="\n"
    )
    (output_dir / "README.md").write_text(
        _render_readme(
            dataset_name=output_dir.name,
            summary=summary,
            source_records=sources,
            partition=partition,
        ),
        encoding="utf-8",
        newline="\n",
    )

    file_rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "archive_manifest.json":
            continue
        relative = path.relative_to(output_dir).as_posix()
        row: dict[str, Any] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if path.suffix == ".jsonl":
            row["rows"] = sum(
                1 for line in path.read_text(encoding="utf-8").splitlines() if line
            )
        file_rows.append(row)
    archive_manifest = {
        "schema": ARCHIVE_SCHEMA,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": output_dir.name,
        "source_run": str(source_dir),
        "teacher_model": summary.get("model"),
        "partition": partition,
        "partition_policy": "diagnostic_only_not_unseen_heldout",
        "films": len(film_rows),
        "media_hours": round(sum(row["duration_s"] for row in sources) / 3600.0, 9),
        "chunks": len(chunk_rows),
        "word_times": len(words),
        "accepted_nonoverlap_speaker_cuts": sum(accepted_cut_counts.values()),
        "provider_actual_cost_usd": summary.get("provider_actual_cost_usd"),
        "training_ready": False,
        "training_blocker": "frame_only_ctc_loss_mask_not_implemented",
        "source_media_archived": False,
        "derived_audio_archived": False,
        "files": file_rows,
    }
    _write_json(output_dir / "archive_manifest.json", archive_manifest)
    return archive_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partition", default=DEFAULT_PARTITION)
    parser.add_argument(
        "--hash-sources",
        action="store_true",
        help="Read every source video and record its complete SHA-256.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = archive_run(
        args.source_dir,
        args.output_dir,
        partition=args.partition,
        hash_sources=args.hash_sources,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
