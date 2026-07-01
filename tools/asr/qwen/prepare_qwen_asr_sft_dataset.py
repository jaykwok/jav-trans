#!/usr/bin/env python3

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from boundary.ja import sample_hf_audio_16k_mono, stable_hf_audio_id  # noqa: E402


ASR_DATASET = "litagin/Galgame_Speech_ASR_16kHz"
SER_DATASET = "litagin/Galgame_Speech_SER_16kHz"
METADATA_DATASET = "litagin/VisualNovel_Dataset_Metadata"
TEXT_FIELDS = ("txt", "text", "transcription", "sentence", "normalized_text")
SCALAR_METADATA_TYPES = (str, int, float, bool, type(None))


@dataclass(frozen=True)
class SourcePlan:
    source_key: str
    dataset: str
    manifest: str | None
    enabled: bool
    train_limit: int
    val_limit: int
    test_limit: int


def normalized_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split()).strip()


def source_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_") or "source"


def stable_row_key(*, source: str, audio_id: str, text: str) -> str:
    digest = hashlib.sha256(f"{source}\0{audio_id}\0{text}".encode("utf-8")).hexdigest()
    return digest[:24]


def extract_text(row: Mapping[str, Any]) -> str:
    for field in TEXT_FIELDS:
        if field in row:
            text = normalized_text(row.get(field))
            if text:
                return text
    return ""


def extract_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in row.items():
        if key in {"ogg", "audio", "txt", "text", "transcription", "sentence", "normalized_text"}:
            continue
        if key.startswith("_"):
            continue
        if isinstance(value, SCALAR_METADATA_TYPES):
            metadata[str(key)] = value
    return metadata


def duration_from_row(row: Mapping[str, Any]) -> float:
    try:
        return float(row.get("duration_s") or row.get("duration") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def should_keep(
    *,
    text: str,
    duration_s: float,
    min_duration_s: float,
    max_duration_s: float,
    allow_empty_text: bool = False,
) -> tuple[bool, str]:
    if not text and not allow_empty_text:
        return False, "missing_text"
    if duration_s and duration_s < min_duration_s:
        return False, "duration_too_short"
    if duration_s and max_duration_s > 0 and duration_s > max_duration_s:
        return False, "duration_too_long"
    return True, ""


def split_sequence_for_index(index: int, *, train_limit: int, val_limit: int, test_limit: int) -> str | None:
    if val_limit > 0 and index < val_limit:
        return "val"
    test_start = max(val_limit, 0)
    if test_limit > 0 and index < test_start + test_limit:
        return "test"
    train_index = index - test_start - max(test_limit, 0)
    if train_limit == 0:
        return "train"
    if train_index < train_limit:
        return "train"
    return None


def source_is_complete(counts: Mapping[str, int], *, plan: SourcePlan) -> bool:
    train_complete = plan.train_limit > 0 and counts.get("train", 0) >= plan.train_limit
    return (
        train_complete
        and counts.get("val", 0) >= plan.val_limit
        and counts.get("test", 0) >= plan.test_limit
    )


def load_dataset_stream(*, dataset_name: str, split: str, revision: str | None, shuffle_seed: int, shuffle_buffer_size: int):
    try:
        from datasets import Features, Value, load_dataset
    except ImportError as exc:
        raise SystemExit("datasets is required: uv pip install datasets") from exc

    dataset_kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if revision:
        dataset_kwargs["revision"] = revision
    features_by_dataset = {
        ASR_DATASET: Features(
            {
                "ogg": Value("binary"),
                "txt": Value("string"),
                "__key__": Value("string"),
                "__url__": Value("string"),
            }
        ),
        SER_DATASET: Features(
            {
                "ogg": Value("binary"),
                "txt": Value("string"),
                "cls": Value("string"),
                "__key__": Value("string"),
                "__url__": Value("string"),
            }
        ),
    }
    features = features_by_dataset.get(dataset_name)
    if features is not None:
        dataset_kwargs["features"] = features
    dataset = load_dataset(dataset_name, **dataset_kwargs)
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer_size)
    return dataset


def iter_manifest_rows(path: Path, *, source: str) -> Iterator[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"manifest must be a JSON list: {path}")
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping):
            continue
        copied = dict(row)
        copied.setdefault("source", source)
        copied.setdefault("input", f"{source}:manifest:{index}")
        copied["_manifest"] = str(path)
        copied["_manifest_index"] = index
        yield copied


def write_local_audio(
    *,
    row: Mapping[str, Any],
    target_audio_dir: Path,
    audio_id: str,
    copy_audio: bool,
) -> tuple[str | None, float, int | None, str | None]:
    source_audio = row.get("audio")
    if not source_audio:
        return None, 0.0, None, "missing_audio"
    source_path = Path(str(source_audio))
    if not source_path.exists():
        return None, 0.0, None, "audio_not_found"
    duration_s = duration_from_row(row)
    sample_rate = row.get("sample_rate")
    if not copy_audio:
        return str(source_path), duration_s, int(sample_rate) if sample_rate else None, None
    suffix = source_path.suffix or ".wav"
    target_path = target_audio_dir / f"{source_slug(audio_id)}{suffix}"
    if not target_path.exists():
        shutil.copy2(source_path, target_path)
    return str(target_path), duration_s, int(sample_rate) if sample_rate else None, None


def write_hf_audio(
    *,
    row: Mapping[str, Any],
    target_audio_dir: Path,
    audio_id: str,
    audio_format: str,
) -> tuple[str | None, float, int | None, str | None]:
    if audio_format == "ogg":
        ogg = row.get("ogg")
        if isinstance(ogg, (bytes, bytearray)):
            target_path = target_audio_dir / f"{source_slug(audio_id)}.ogg"
            if not target_path.exists():
                target_path.write_bytes(bytes(ogg))
            try:
                info = sf.info(io.BytesIO(bytes(ogg)))
                duration_s = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
                return str(target_path), duration_s, int(info.samplerate), None
            except Exception:
                return str(target_path), 0.0, None, None
    try:
        audio, sample_rate = sample_hf_audio_16k_mono(row)
    except Exception as exc:
        return None, 0.0, None, f"decode_error: {exc}"
    target_path = target_audio_dir / f"{source_slug(audio_id)}.wav"
    sf.write(str(target_path), audio, sample_rate)
    return str(target_path), len(audio) / sample_rate if sample_rate else 0.0, int(sample_rate), None


def write_jsonl_row(handle, payload: Mapping[str, Any]) -> None:
    handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")


def process_rows(
    *,
    rows: Iterable[Mapping[str, Any]],
    plan: SourcePlan,
    output_root: Path,
    split_handles: Mapping[str, Any],
    manifest_handles: Mapping[str, Any],
    args: argparse.Namespace,
    from_hf: bool,
) -> dict[str, Any]:
    target_audio_dir = output_root / "audio" / plan.source_key
    target_audio_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    skip_counts: Counter[str] = Counter()
    decoded = 0
    considered = 0
    for source_index, row in enumerate(rows):
        if source_is_complete(counts, plan=plan):
            break
        text = extract_text(row)
        raw_audio_id = str(
            row.get("audio_id")
            or row.get("__key__")
            or row.get("id")
            or stable_hf_audio_id(dataset_name=plan.dataset, split=args.hf_split, index=source_index)
        )
        split = split_sequence_for_index(
            considered,
            train_limit=plan.train_limit,
            val_limit=plan.val_limit,
            test_limit=plan.test_limit,
        )
        if split is None:
            break
        audio_id = f"{plan.source_key}-{source_slug(Path(raw_audio_id).stem)}"
        if from_hf:
            audio_path, duration_s, sample_rate, error = write_hf_audio(
                row=row,
                target_audio_dir=target_audio_dir,
                audio_id=audio_id,
                audio_format=args.hf_audio_format,
            )
        else:
            audio_path, duration_s, sample_rate, error = write_local_audio(
                row=row,
                target_audio_dir=target_audio_dir,
                audio_id=audio_id,
                copy_audio=args.copy_audio,
            )
        if error:
            skip_counts[error.split(":", 1)[0]] += 1
            continue
        assert audio_path is not None
        keep, reason = should_keep(
            text=text,
            duration_s=duration_s,
            min_duration_s=args.min_duration_s,
            max_duration_s=args.max_duration_s,
        )
        if not keep:
            skip_counts[reason] += 1
            continue
        row_key = stable_row_key(source=plan.source_key, audio_id=audio_id, text=text)
        sft_payload = {
            "audio": audio_path,
            "text": text,
            "language": args.language,
        }
        manifest_payload = {
            "row_key": row_key,
            "audio_id": audio_id,
            "audio": audio_path,
            "text": text,
            "language": args.language,
            "source": plan.dataset,
            "source_key": plan.source_key,
            "split": split,
            "duration_s": duration_s,
            "sample_rate": sample_rate,
            "metadata": extract_metadata(row),
        }
        write_jsonl_row(split_handles[split], sft_payload)
        write_jsonl_row(manifest_handles[split], manifest_payload)
        counts[split] += 1
        decoded += 1
        considered += 1
    return {
        "source_key": plan.source_key,
        "dataset": plan.dataset,
        "manifest": plan.manifest,
        "enabled": plan.enabled,
        "counts": dict(sorted(counts.items())),
        "decoded": decoded,
        "considered": considered,
        "skip_counts": dict(sorted(skip_counts.items())),
    }


def iter_hard_negative_rows(paths: Iterable[str]) -> Iterator[dict[str, Any]]:
    for path_text in paths:
        path = Path(path_text)
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                continue
            copied = dict(row)
            copied.setdefault("source", "hard_negative")
            copied.setdefault("input", f"{path}:{index}")
            copied["_manifest"] = str(path)
            copied["_manifest_index"] = index
            yield copied


def append_hard_negatives(
    *,
    rows: Iterable[Mapping[str, Any]],
    output_root: Path,
    split_handles: Mapping[str, Any],
    manifest_handles: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    target_audio_dir = output_root / "audio" / "hard-negative"
    target_audio_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    skip_counts: Counter[str] = Counter()
    for row in rows:
        if args.hard_negative_limit is not None and kept >= args.hard_negative_limit:
            break
        audio_id = str(row.get("audio_id") or row.get("id") or f"hard-negative-{kept:06d}")
        audio_path, duration_s, sample_rate, error = write_local_audio(
            row=row,
            target_audio_dir=target_audio_dir,
            audio_id=f"hard-negative-{source_slug(audio_id)}",
            copy_audio=args.copy_audio,
        )
        if error:
            skip_counts[error] += 1
            continue
        assert audio_path is not None
        keep, reason = should_keep(
            text="",
            duration_s=duration_s,
            min_duration_s=args.min_duration_s,
            max_duration_s=args.max_duration_s,
            allow_empty_text=True,
        )
        if not keep:
            skip_counts[reason] += 1
            continue
        sft_payload = {
            "audio": audio_path,
            "text": "",
            "language": args.language,
        }
        manifest_payload = {
            "row_key": stable_row_key(source="hard-negative", audio_id=audio_id, text=""),
            "audio_id": audio_id,
            "audio": audio_path,
            "text": "",
            "language": args.language,
            "source": str(row.get("source") or "hard_negative"),
            "source_key": "hard-negative",
            "split": args.hard_negative_split,
            "duration_s": duration_s,
            "sample_rate": sample_rate,
            "metadata": extract_metadata(row),
        }
        write_jsonl_row(split_handles[args.hard_negative_split], sft_payload)
        write_jsonl_row(manifest_handles[args.hard_negative_split], manifest_payload)
        kept += 1
    return {"kept": kept, "skip_counts": dict(sorted(skip_counts.items()))}


def source_plans(args: argparse.Namespace) -> list[SourcePlan]:
    return [
        SourcePlan(
            source_key="galgame-asr",
            dataset=args.asr_dataset,
            manifest=args.asr_manifest,
            enabled=not args.no_asr,
            train_limit=args.asr_train_limit,
            val_limit=args.asr_val_limit,
            test_limit=args.asr_test_limit,
        ),
        SourcePlan(
            source_key="galgame-ser",
            dataset=args.ser_dataset,
            manifest=args.ser_manifest,
            enabled=not args.no_ser,
            train_limit=args.ser_train_limit,
            val_limit=args.ser_val_limit,
            test_limit=args.ser_test_limit,
        ),
    ]


def apply_cloud_env(args: argparse.Namespace) -> None:
    if args.hf_cache_dir:
        os.environ["HF_HOME"] = args.hf_cache_dir
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    if args.hf_xet_high_performance:
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"


def run(args: argparse.Namespace) -> None:
    apply_cloud_env(args)
    output_root = Path(args.output_root)
    sft_dir = output_root / "qwen-sft"
    manifest_dir = output_root / "manifest"
    sft_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    split_paths = {split: sft_dir / f"{split}.jsonl" for split in ("train", "val", "test")}
    manifest_paths = {split: manifest_dir / f"{split}.manifest.jsonl" for split in ("train", "val", "test")}
    split_handles = {split: path.open("w", encoding="utf-8") for split, path in split_paths.items()}
    manifest_handles = {split: path.open("w", encoding="utf-8") for split, path in manifest_paths.items()}
    source_summaries = []
    hard_negative_summary: dict[str, Any] | None = None
    try:
        for plan in source_plans(args):
            if not plan.enabled:
                source_summaries.append({"source_key": plan.source_key, "dataset": plan.dataset, "enabled": False})
                continue
            if plan.manifest:
                rows = iter_manifest_rows(Path(plan.manifest), source=plan.dataset)
                source_summaries.append(
                    process_rows(
                        rows=rows,
                        plan=plan,
                        output_root=output_root,
                        split_handles=split_handles,
                        manifest_handles=manifest_handles,
                        args=args,
                        from_hf=False,
                    )
                )
            else:
                dataset = load_dataset_stream(
                    dataset_name=plan.dataset,
                    split=args.hf_split,
                    revision=args.revision,
                    shuffle_seed=args.shuffle_seed,
                    shuffle_buffer_size=args.shuffle_buffer_size,
                )
                source_summaries.append(
                    process_rows(
                        rows=dataset,
                        plan=plan,
                        output_root=output_root,
                        split_handles=split_handles,
                        manifest_handles=manifest_handles,
                        args=args,
                        from_hf=True,
                    )
                )
        if args.hard_negative_jsonl:
            hard_negative_summary = append_hard_negatives(
                rows=iter_hard_negative_rows(args.hard_negative_jsonl),
                output_root=output_root,
                split_handles=split_handles,
                manifest_handles=manifest_handles,
                args=args,
            )
    finally:
        for handle in split_handles.values():
            handle.close()
        for handle in manifest_handles.values():
            handle.close()

    split_counts = {}
    for split, path in split_paths.items():
        split_counts[split] = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    summary = {
        "mode": args.mode,
        "language": args.language,
        "output_root": str(output_root),
        "sft_files": {split: str(path) for split, path in split_paths.items()},
        "manifest_files": {split: str(path) for split, path in manifest_paths.items()},
        "split_counts": split_counts,
        "source_summaries": source_summaries,
        "hard_negative_summary": hard_negative_summary,
        "hf_split": args.hf_split,
        "shuffle_seed": args.shuffle_seed,
        "shuffle_buffer_size": args.shuffle_buffer_size,
        "revision": args.revision,
        "hf_cache_dir": args.hf_cache_dir,
        "hf_endpoint": args.hf_endpoint,
        "hf_audio_format": args.hf_audio_format,
        "metadata_dataset": args.metadata_dataset,
        "training_text_format": "raw transcript plus separate language field",
    }
    summary_path = output_root / "qwen_sft_dataset_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"summary={summary_path}")
    print(f"split_counts={split_counts}")


def mode_default(value: int | None, *, smoke: int, full: int, mode: str) -> int:
    if value is not None:
        return value
    return smoke if mode == "smoke" else full


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare reproducible Qwen3-ASR SFT JSONL from Galgame ASR/SER data."
    )
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output-root")
    parser.add_argument("--language", default="Japanese")
    parser.add_argument("--asr-dataset", default=ASR_DATASET)
    parser.add_argument("--ser-dataset", default=SER_DATASET)
    parser.add_argument("--metadata-dataset", default=METADATA_DATASET)
    parser.add_argument("--asr-manifest")
    parser.add_argument("--ser-manifest")
    parser.add_argument("--no-asr", action="store_true")
    parser.add_argument("--no-ser", action="store_true")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--revision")
    parser.add_argument("--shuffle-buffer-size", type=int)
    parser.add_argument("--shuffle-seed", type=int, default=20260526)
    parser.add_argument("--hf-cache-dir")
    parser.add_argument("--hf-endpoint")
    parser.add_argument("--hf-xet-high-performance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hf-audio-format", choices=["wav", "ogg"], default="wav")
    parser.add_argument("--copy-audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-duration-s", type=float, default=0.2)
    parser.add_argument("--max-duration-s", type=float, default=30.0)
    parser.add_argument("--asr-train-limit", type=int)
    parser.add_argument("--asr-val-limit", type=int)
    parser.add_argument("--asr-test-limit", type=int)
    parser.add_argument("--ser-train-limit", type=int)
    parser.add_argument("--ser-val-limit", type=int)
    parser.add_argument("--ser-test-limit", type=int)
    parser.add_argument("--hard-negative-jsonl", action="append")
    parser.add_argument("--hard-negative-limit", type=int)
    parser.add_argument("--hard-negative-split", choices=["train", "val", "test"], default="train")
    args = parser.parse_args(argv)
    if args.output_root is None:
        default_name = "v1-smoke" if args.mode == "smoke" else "v1-full"
        args.output_root = str(PROJECT_ROOT / "datasets" / "train" / "qwen3-asr-ja-galgame" / default_name)
    args.shuffle_buffer_size = mode_default(args.shuffle_buffer_size, smoke=128, full=4096, mode=args.mode)
    args.asr_train_limit = mode_default(args.asr_train_limit, smoke=40, full=0, mode=args.mode)
    args.asr_val_limit = mode_default(args.asr_val_limit, smoke=5, full=1000, mode=args.mode)
    args.asr_test_limit = mode_default(args.asr_test_limit, smoke=5, full=1000, mode=args.mode)
    args.ser_train_limit = mode_default(args.ser_train_limit, smoke=10, full=0, mode=args.mode)
    args.ser_val_limit = mode_default(args.ser_val_limit, smoke=2, full=500, mode=args.mode)
    args.ser_test_limit = mode_default(args.ser_test_limit, smoke=2, full=500, mode=args.mode)
    if args.shuffle_buffer_size < 0:
        parser.error("--shuffle-buffer-size must be non-negative")
    for key in (
        "asr_train_limit",
        "asr_val_limit",
        "asr_test_limit",
        "ser_train_limit",
        "ser_val_limit",
        "ser_test_limit",
    ):
        if getattr(args, key) < 0:
            parser.error(f"--{key.replace('_', '-')} must be non-negative")
    if args.min_duration_s < 0 or args.max_duration_s < 0:
        parser.error("duration limits must be non-negative")
    if args.hard_negative_limit is not None and args.hard_negative_limit < 0:
        parser.error("--hard-negative-limit must be non-negative")
    return args


if __name__ == "__main__":
    run(parse_args())
