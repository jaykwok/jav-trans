from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be a JSON object")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _segment_id(row: dict[str, Any], index: int) -> str:
    for key in ("segment_id", "id", "chunk_id", "index"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return str(index)


def _embedding(row: dict[str, Any]) -> np.ndarray:
    value = row.get("embedding")
    if value is None:
        value = row.get("speaker_embedding")
    if not isinstance(value, list) or not value:
        raise ValueError(f"segment {row.get('segment_id') or row.get('id')}: missing embedding list")
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"segment {row.get('segment_id') or row.get('id')}: embedding must be 1-D")
    norm = float(np.linalg.norm(arr))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError(f"segment {row.get('segment_id') or row.get('id')}: embedding norm is zero")
    return arr / norm


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def build_adjacent_speaker_change_rows(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    normalized = sorted(
        rows,
        key=lambda row: (
            _float(row, "start", _float(row, "start_s", 0.0)),
            _float(row, "end", _float(row, "end_s", 0.0)),
        ),
    )
    out: list[dict[str, Any]] = []
    if len(normalized) < 2:
        return out

    embeddings = [_embedding(row) for row in normalized]
    for index in range(len(normalized) - 1):
        left = normalized[index]
        right = normalized[index + 1]
        cosine = float(np.clip(np.dot(embeddings[index], embeddings[index + 1]), -1.0, 1.0))
        score = 1.0 - cosine
        left_end = _float(left, "end", _float(left, "end_s", 0.0))
        right_start = _float(right, "start", _float(right, "start_s", left_end))
        out.append(
            {
                "left_segment_id": _segment_id(left, index),
                "right_segment_id": _segment_id(right, index + 1),
                "left_start": _float(left, "start", _float(left, "start_s", 0.0)),
                "left_end": left_end,
                "right_start": right_start,
                "right_end": _float(right, "end", _float(right, "end_s", right_start)),
                "gap_s": round(right_start - left_end, 6),
                "cosine": round(cosine, 6),
                "speaker_change_score": round(score, 6),
                "speaker_change": score >= threshold,
                "threshold": threshold,
            }
        )
    return out


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"pair_count": 0, "speaker_change_count": 0}
    scores = sorted(float(row["speaker_change_score"]) for row in rows)
    return {
        "pair_count": len(rows),
        "speaker_change_count": sum(1 for row in rows if row.get("speaker_change")),
        "score_min": round(scores[0], 6),
        "score_p50": round(scores[len(scores) // 2], 6),
        "score_max": round(scores[-1], 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Offline speaker sidecar probe. Input is JSONL with segment timing and "
            "precomputed ERes2NetV2/3D-Speaker/CAM++ embeddings. The tool only "
            "computes adjacent speaker-change scores; it does not replace VAD."
        )
    )
    parser.add_argument("--segments", required=True, help="JSONL with segment embeddings.")
    parser.add_argument("--output", required=True, help="Output adjacent-pair JSONL path.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="speaker_change_score threshold; score = 1 - cosine.",
    )
    args = parser.parse_args()

    rows = _read_jsonl(Path(args.segments))
    pairs = build_adjacent_speaker_change_rows(rows, threshold=float(args.threshold))
    out_path = Path(args.output)
    _write_jsonl(out_path, pairs)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(_summary(pairs), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"pairs={len(pairs)} output={out_path} summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
