#!/usr/bin/env python3
from __future__ import annotations

from tools.asr.cueqc.compile_pre_asr_v11_features import (
    candidate_for_chunk,
    compile_features,
    expand_chunk_paths,
    label_for_chunk,
    main,
    normalize_label,
    read_chunk_document,
    read_labels,
)

__all__ = [
    "candidate_for_chunk",
    "compile_features",
    "expand_chunk_paths",
    "label_for_chunk",
    "main",
    "normalize_label",
    "read_chunk_document",
    "read_labels",
]


if __name__ == "__main__":
    raise SystemExit(main())
