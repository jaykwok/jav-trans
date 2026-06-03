#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.loading import load_audio_16k_mono
from tools.subtitles.probe_speaker_sidecar import build_adjacent_speaker_change_rows


DEFAULT_ERES2NETV2_MODEL_ID = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
DEFAULT_MODELSCOPE_CACHE_DIR = PROJECT_ROOT / "models" / "modelscope"
SUPPORTED_BACKENDS = {"energy_mfcc", "modelscope_eres2netv2"}


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def project_rel(value: str | Path | None) -> str:
    if not value:
        return ""
    raw = Path(value)
    try:
        return raw.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return raw.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_blocks(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    blocks = payload.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError(f"missing blocks list in {path}")
    return [dict(block) for block in blocks if isinstance(block, dict)]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _text(block: dict[str, Any]) -> str:
    return str(block.get("ja_text") or block.get("text") or block.get("ja") or "").strip()


def _normalize_embedding(values: np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 0 or not math.isfinite(norm):
        arr = np.zeros_like(arr, dtype=np.float32)
        if arr.size:
            arr[0] = 1.0
    else:
        arr = arr / norm
    return [round(float(v), 8) for v in arr.tolist()]


def _slice_audio(audio: np.ndarray, start: float, end: float, sample_rate: int) -> np.ndarray:
    start_i = max(0, min(len(audio), int(round(start * sample_rate))))
    end_i = max(start_i, min(len(audio), int(round(end * sample_rate))))
    return np.asarray(audio[start_i:end_i], dtype=np.float32)


def energy_mfcc_embedding(clip: np.ndarray, sample_rate: int) -> np.ndarray:
    if clip.size == 0:
        return np.zeros(8, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(clip, dtype=np.float32))))
    abs_clip = np.abs(clip)
    zcr = float(np.mean(np.abs(np.diff(np.signbit(clip).astype(np.float32))))) if clip.size > 1 else 0.0
    peak = float(np.max(abs_clip)) if clip.size else 0.0
    duration = clip.size / max(1, sample_rate)
    try:
        import librosa

        mfcc = librosa.feature.mfcc(y=clip.astype(np.float32), sr=sample_rate, n_mfcc=6)
        mfcc_mean = np.mean(mfcc, axis=1).astype(np.float32)
    except Exception:
        mfcc_mean = np.zeros(6, dtype=np.float32)
    return np.concatenate(
        [
            np.asarray([rms, zcr, peak, duration], dtype=np.float32),
            mfcc_mean[:4],
        ],
        axis=0,
    )


class ModelScopeEres2NetV2Embedder:
    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        model_cache_dir: Path | None,
        pipeline_factory: Callable[[str, str], Any] | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.model_cache_dir = model_cache_dir
        try:
            self.pipeline = (
                pipeline_factory(model_id, device)
                if pipeline_factory is not None
                else self._build_pipeline(model_id, device, model_cache_dir)
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "modelscope_eres2netv2 backend requires optional ModelScope audio "
                "dependencies. Install them in .venv first, for example: "
                "uv pip install addict sortedcontainers simplejson datasets oss2 "
                "modelscope. Missing module: "
                f"{exc.name or exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                "failed to initialize modelscope_eres2netv2 speaker sidecar backend. "
                "Check ModelScope dependencies, model availability, network/proxy, and "
                f"CUDA device settings. Cause: {type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _build_pipeline(model_id: str, device: str, model_cache_dir: Path | None) -> Any:
        # Importing this module registers the ERes2NetV2 speaker verification pipeline
        # in ModelScope. The public audio lazy import table does not expose it directly.
        import modelscope.pipelines.audio.speaker_verification_eres2netv2_pipeline  # noqa: F401
        from modelscope.hub.snapshot_download import snapshot_download
        from modelscope.metainfo import Pipelines
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        model_path = snapshot_download(
            model_id,
            cache_dir=str(model_cache_dir) if model_cache_dir is not None else None,
        )
        return pipeline(
            task=Tasks.speaker_verification,
            model=model_path,
            pipeline_name=Pipelines.speaker_verification_eres2netv2,
            device=device,
        )

    def embed_batch(self, clips: list[np.ndarray]) -> list[np.ndarray]:
        if not clips:
            return []
        try:
            import torch

            with torch.inference_mode():
                payload = self.pipeline(clips, output_emb=True)
        except Exception as exc:
            raise RuntimeError(
                "modelscope_eres2netv2 embedding extraction failed. "
                f"Check clip duration, audio shape, device, and model files. Cause: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(payload, dict) or "embs" not in payload:
            raise RuntimeError("modelscope_eres2netv2 pipeline did not return an 'embs' array")
        embs = np.asarray(payload["embs"], dtype=np.float32)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        if embs.ndim != 2 or embs.shape[0] != len(clips):
            raise RuntimeError(
                "modelscope_eres2netv2 embedding shape mismatch: "
                f"expected {len(clips)} rows, got {tuple(embs.shape)}"
            )
        return [np.asarray(row, dtype=np.float32) for row in embs]


def _embedding_row(
    *,
    block: dict[str, Any],
    index: int,
    start: float,
    end: float,
    duration: float,
    backend: str,
    embedding: np.ndarray,
    model_id: str | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "segment_id": str(block.get("cue_id", index)),
        "cue_id": block.get("cue_id", index),
        "index": index,
        "start": round(start, 6),
        "end": round(end, 6),
        "duration_s": round(duration, 6),
        "text": _text(block)[:160],
        "backend": backend,
        "embedding": _normalize_embedding(embedding),
    }
    if model_id:
        row["model_id"] = model_id
    source_ids = block.get("source_segment_ids")
    if isinstance(source_ids, list):
        row["source_segment_ids"] = source_ids
    return row


def build_embedding_rows(
    blocks: list[dict[str, Any]],
    audio: np.ndarray,
    *,
    sample_rate: int,
    backend: str,
    min_duration_s: float,
    max_segments: int,
    batch_size: int,
    model_id: str,
    device: str,
    model_cache_dir: Path | None,
    pipeline_factory: Callable[[str, str], Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {"short": 0, "empty_audio": 0}
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"unsupported backend: {backend}")
    embedder = (
        ModelScopeEres2NetV2Embedder(
            model_id=model_id,
            device=device,
            model_cache_dir=model_cache_dir,
            pipeline_factory=pipeline_factory,
        )
        if backend == "modelscope_eres2netv2"
        else None
    )
    pending: list[tuple[int, dict[str, Any], float, float, float, np.ndarray]] = []
    effective_batch_size = max(1, int(batch_size))

    def flush_pending() -> None:
        if not pending:
            return
        if embedder is None:
            raise RuntimeError("internal error: pending model clips without embedder")
        clips = [item[5] for item in pending]
        embeddings = embedder.embed_batch(clips)
        for (index, block, start, end, duration, _clip), embedding in zip(pending, embeddings, strict=True):
            rows.append(
                _embedding_row(
                    block=block,
                    index=index,
                    start=start,
                    end=end,
                    duration=duration,
                    backend=backend,
                    embedding=embedding,
                    model_id=model_id,
                )
            )
        pending.clear()

    for index, block in enumerate(blocks):
        if max_segments > 0 and len(rows) + len(pending) >= max_segments:
            break
        start = _float(block.get("start"))
        end = max(start, _float(block.get("end"), start))
        duration = end - start
        if duration < min_duration_s:
            skipped["short"] += 1
            continue
        clip = _slice_audio(audio, start, end, sample_rate)
        if clip.size == 0:
            skipped["empty_audio"] += 1
            continue
        if backend == "energy_mfcc":
            rows.append(
                _embedding_row(
                    block=block,
                    index=index,
                    start=start,
                    end=end,
                    duration=duration,
                    backend=backend,
                    embedding=energy_mfcc_embedding(clip, sample_rate),
                    model_id=None,
                )
            )
            continue
        pending.append((index, block, start, end, duration, clip))
        if len(pending) >= effective_batch_size:
            flush_pending()
    flush_pending()
    return rows, skipped


def build_summary(
    *,
    bilingual_path: Path,
    audio_path: Path,
    output_dir: Path,
    backend: str,
    min_duration_s: float,
    speaker_threshold: float,
    max_segments: int,
    batch_size: int = 16,
    model_id: str = DEFAULT_ERES2NETV2_MODEL_ID,
    device: str = "gpu",
    model_cache_dir: Path | None = DEFAULT_MODELSCOPE_CACHE_DIR,
    pipeline_factory: Callable[[str, str], Any] | None = None,
) -> dict[str, Any]:
    blocks = load_blocks(bilingual_path)
    audio, sample_rate = load_audio_16k_mono(str(audio_path))
    rows, skipped = build_embedding_rows(
        blocks,
        audio,
        sample_rate=sample_rate,
        backend=backend,
        min_duration_s=min_duration_s,
        max_segments=max_segments,
        batch_size=batch_size,
        model_id=model_id,
        device=device,
        model_cache_dir=model_cache_dir,
        pipeline_factory=pipeline_factory,
    )
    pairs = build_adjacent_speaker_change_rows(rows, threshold=speaker_threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "speaker_embeddings.jsonl"
    pairs_path = output_dir / "speaker_pairs.jsonl"
    write_jsonl(embeddings_path, rows)
    write_jsonl(pairs_path, pairs)
    scores = sorted(float(row["speaker_change_score"]) for row in pairs)
    summary = {
        "source_bilingual": project_rel(bilingual_path),
        "source_audio": project_rel(audio_path),
        "backend": backend,
        "model_id": model_id if backend == "modelscope_eres2netv2" else "",
        "device": device if backend == "modelscope_eres2netv2" else "",
        "model_cache_dir": project_rel(model_cache_dir) if backend == "modelscope_eres2netv2" else "",
        "batch_size": batch_size if backend == "modelscope_eres2netv2" else 0,
        "sample_rate": sample_rate,
        "block_count": len(blocks),
        "embedding_count": len(rows),
        "embedding_dim": len(rows[0]["embedding"]) if rows else 0,
        "pair_count": len(pairs),
        "speaker_change_count": sum(1 for row in pairs if row.get("speaker_change")),
        "speaker_threshold": speaker_threshold,
        "score_min": round(scores[0], 6) if scores else 0.0,
        "score_p50": round(scores[len(scores) // 2], 6) if scores else 0.0,
        "score_max": round(scores[-1], 6) if scores else 0.0,
        "skipped": skipped,
        "embeddings_path": project_rel(embeddings_path),
        "pairs_path": project_rel(pairs_path),
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(build_markdown(summary), encoding="utf-8")
    return summary


def build_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Speaker Sidecar Embedding Probe",
            "",
            f"- source: `{summary['source_bilingual']}`",
            f"- audio: `{summary['source_audio']}`",
            f"- backend: `{summary['backend']}`",
            f"- model_id: `{summary['model_id']}`",
            f"- device: `{summary['device']}`",
            f"- model_cache_dir: `{summary['model_cache_dir']}`",
            f"- batch_size: {summary['batch_size']}",
            f"- embedding_count: {summary['embedding_count']} / {summary['block_count']}",
            f"- embedding_dim: {summary['embedding_dim']}",
            f"- pair_count: {summary['pair_count']}",
            f"- speaker_change_count: {summary['speaker_change_count']}",
            f"- score p50/max: {summary['score_p50']} / {summary['score_max']}",
            f"- skipped: `{summary['skipped']}`",
            f"- embeddings: `{summary['embeddings_path']}`",
            f"- pairs: `{summary['pairs_path']}`",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract offline speaker-sidecar embeddings from subtitle cues and "
            "emit adjacent speaker-change pairs. This does not alter runtime VAD."
        )
    )
    parser.add_argument("--bilingual", required=True, help="bilingual.json path")
    parser.add_argument("--audio", required=True, help="16k/any-rate mono/stereo audio path")
    parser.add_argument(
        "--output-dir",
        default="agents/temp/fusionvad-ja/speaker-sidecar-embeddings",
    )
    parser.add_argument(
        "--backend",
        choices=("energy_mfcc", "modelscope_eres2netv2"),
        default="energy_mfcc",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_ERES2NETV2_MODEL_ID,
        help="ModelScope speaker verification model id for --backend modelscope_eres2netv2.",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        help="ModelScope device string for --backend modelscope_eres2netv2, e.g. gpu or cpu.",
    )
    parser.add_argument(
        "--model-cache-dir",
        default=project_rel(DEFAULT_MODELSCOPE_CACHE_DIR),
        help="Project-local ModelScope cache dir for --backend modelscope_eres2netv2.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-duration-s", type=float, default=0.45)
    parser.add_argument("--speaker-threshold", type=float, default=0.35)
    parser.add_argument("--max-segments", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        summary = build_summary(
            bilingual_path=project_path(args.bilingual),
            audio_path=project_path(args.audio),
            output_dir=project_path(args.output_dir),
            backend=args.backend,
            min_duration_s=float(args.min_duration_s),
            speaker_threshold=float(args.speaker_threshold),
            max_segments=int(args.max_segments),
            batch_size=int(args.batch_size),
            model_id=str(args.model_id),
            device=str(args.device),
            model_cache_dir=project_path(args.model_cache_dir) if args.model_cache_dir else None,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        "summary={summary} embeddings={embeddings} pairs={pairs} changes={changes}".format(
            summary=project_rel(project_path(args.output_dir) / "summary.json"),
            embeddings=summary["embedding_count"],
            pairs=summary["pair_count"],
            changes=summary["speaker_change_count"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
