from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from safetensors import safe_open
from safetensors.torch import save_file


HF_TEMPLATE_REPO = "Qwen/Qwen3-ASR-1.7B-hf"
TEMPLATE_FILES = (
    "config.json",
    "generation_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.jinja",
)

_DTYPE_SIZES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


@dataclass(frozen=True)
class TensorInfo:
    source_file: Path
    source_key: str
    target_key: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int


def _read_safetensors_header(path: Path) -> dict:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(header_len).decode("utf-8"))


def _tensor_nbytes(entry: dict) -> int:
    dtype = str(entry.get("dtype") or "").upper()
    shape = entry.get("shape") or []
    if dtype not in _DTYPE_SIZES:
        raise ValueError(f"Unsupported safetensors dtype: {dtype}")
    return int(math.prod(int(dim) for dim in shape)) * _DTYPE_SIZES[dtype]


def _iter_source_tensors(source_model_dir: Path) -> list[TensorInfo]:
    safetensor_paths = sorted(source_model_dir.glob("model*.safetensors"))
    if not safetensor_paths:
        raise FileNotFoundError(f"No model*.safetensors found in {source_model_dir}")

    tensors: list[TensorInfo] = []
    seen_targets: set[str] = set()
    for path in safetensor_paths:
        header = _read_safetensors_header(path)
        for source_key, entry in header.items():
            if source_key == "__metadata__":
                continue
            if source_key.startswith("thinker."):
                target_key = "model." + source_key[len("thinker.") :]
            elif source_key.startswith("model."):
                target_key = source_key
            else:
                raise ValueError(
                    f"Unexpected Qwen3-ASR tensor key {source_key!r}; expected thinker.* or model.*"
                )
            if target_key in seen_targets:
                raise ValueError(f"Duplicate target tensor key after conversion: {target_key}")
            seen_targets.add(target_key)
            tensors.append(
                TensorInfo(
                    source_file=path,
                    source_key=source_key,
                    target_key=target_key,
                    shape=tuple(int(dim) for dim in entry.get("shape") or ()),
                    dtype=str(entry.get("dtype") or ""),
                    nbytes=_tensor_nbytes(entry),
                )
            )
    return sorted(tensors, key=lambda item: item.target_key)


def _download_template_files(template_repo: str, template_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    template_dir.mkdir(parents=True, exist_ok=True)
    for filename in TEMPLATE_FILES:
        hf_hub_download(
            template_repo,
            filename,
            local_dir=str(template_dir),
        )


def _copy_template_files(template_dir: Path, output_dir: Path) -> list[str]:
    copied: list[str] = []
    for filename in TEMPLATE_FILES:
        source = template_dir / filename
        if not source.exists():
            raise FileNotFoundError(f"Template file missing: {source}")
        target = output_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(filename)
    return copied


def _plan_shards(
    tensors: Iterable[TensorInfo],
    *,
    max_shard_size_bytes: int,
) -> list[list[TensorInfo]]:
    shards: list[list[TensorInfo]] = []
    current: list[TensorInfo] = []
    current_size = 0
    for tensor in tensors:
        if current and current_size + tensor.nbytes > max_shard_size_bytes:
            shards.append(current)
            current = []
            current_size = 0
        current.append(tensor)
        current_size += tensor.nbytes
    if current:
        shards.append(current)
    return shards


def _write_shards(
    shards: list[list[TensorInfo]],
    output_dir: Path,
    *,
    dry_run: bool,
) -> dict[str, str]:
    weight_map: dict[str, str] = {}
    shard_count = len(shards)
    for shard_index, shard in enumerate(shards, start=1):
        if shard_count == 1:
            filename = "model.safetensors"
        else:
            filename = f"model-{shard_index:05d}-of-{shard_count:05d}.safetensors"
        for tensor in shard:
            weight_map[tensor.target_key] = filename
        if dry_run:
            continue

        with ExitStack() as stack:
            tensors = {}
            open_files: dict[Path, object] = {}
            for tensor in shard:
                reader = open_files.get(tensor.source_file)
                if reader is None:
                    reader = stack.enter_context(
                        safe_open(tensor.source_file, framework="pt", device="cpu")
                    )
                    open_files[tensor.source_file] = reader
                tensors[tensor.target_key] = reader.get_tensor(tensor.source_key)
            save_file(tensors, output_dir / filename, metadata={"format": "pt"})
    return weight_map


def convert_qwen3_asr_checkpoint_to_hf(
    *,
    source_model_dir: Path,
    output_dir: Path,
    template_dir: Path | None = None,
    template_repo: str = HF_TEMPLATE_REPO,
    max_shard_size_bytes: int = 2 * 1024 * 1024 * 1024,
    dry_run: bool = False,
) -> dict:
    source_model_dir = source_model_dir.resolve()
    output_dir = output_dir.resolve()
    template_dir = (template_dir or output_dir / "_hf_template").resolve()

    if not source_model_dir.exists():
        raise FileNotFoundError(f"Source model dir does not exist: {source_model_dir}")
    if max_shard_size_bytes <= 0:
        raise ValueError("max_shard_size_bytes must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    if not all((template_dir / filename).exists() for filename in TEMPLATE_FILES):
        _download_template_files(template_repo, template_dir)
    copied_template_files = [] if dry_run else _copy_template_files(template_dir, output_dir)

    tensors = _iter_source_tensors(source_model_dir)
    shards = _plan_shards(tensors, max_shard_size_bytes=max_shard_size_bytes)
    weight_map = _write_shards(shards, output_dir, dry_run=dry_run)

    total_size = sum(tensor.nbytes for tensor in tensors)
    if not dry_run and len(shards) > 1:
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": dict(sorted(weight_map.items())),
        }
        (output_dir / "model.safetensors.index.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    report = {
        "schema": "qwen3_asr_non_hf_to_hf_conversion_v1",
        "source_model_dir": str(source_model_dir),
        "output_dir": str(output_dir),
        "template_repo": template_repo,
        "template_dir": str(template_dir),
        "dry_run": dry_run,
        "tensor_count": len(tensors),
        "total_tensor_bytes": total_size,
        "shard_count": len(shards),
        "shards": [
            {
                "index": index,
                "tensor_count": len(shard),
                "tensor_bytes": sum(tensor.nbytes for tensor in shard),
            }
            for index, shard in enumerate(shards, start=1)
        ],
        "copied_template_files": copied_template_files,
        "key_rewrite": "thinker.* -> model.*",
    }
    if not dry_run:
        (output_dir / "conversion_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def _parse_size_bytes(value: str) -> int:
    raw = value.strip().lower()
    units = {"b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3}
    for suffix, multiplier in sorted(units.items(), key=lambda item: -len(item[0])):
        if raw.endswith(suffix):
            return int(float(raw[: -len(suffix)].strip()) * multiplier)
    return int(float(raw))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert qwen-asr Qwen3-ASR checkpoints to Transformers-native -hf layout."
    )
    parser.add_argument("--source-model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--template-dir", type=Path, default=None)
    parser.add_argument("--template-repo", default=HF_TEMPLATE_REPO)
    parser.add_argument("--max-shard-size", default="2GB")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = convert_qwen3_asr_checkpoint_to_hf(
        source_model_dir=args.source_model_dir,
        output_dir=args.output_dir,
        template_dir=args.template_dir,
        template_repo=args.template_repo,
        max_shard_size_bytes=_parse_size_bytes(args.max_shard_size),
        dry_run=args.dry_run,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
