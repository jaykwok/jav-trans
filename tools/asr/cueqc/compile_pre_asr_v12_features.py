#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.asr.cueqc.pre_asr_feature_compiler import (
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
