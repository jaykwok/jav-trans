#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.asr.cueqc.train_pre_asr_v11_binary import (
    _boost_anchor_positions,
    _matching_candidate_positions,
    _split_label_masks,
    _window_batch_from_anchors,
    load_feature_bundle,
    main,
    train,
)

__all__ = [
    "_boost_anchor_positions",
    "_matching_candidate_positions",
    "_split_label_masks",
    "_window_batch_from_anchors",
    "load_feature_bundle",
    "main",
    "train",
]


if __name__ == "__main__":
    raise SystemExit(main())
