#!/usr/bin/env python3
from __future__ import annotations

from tools.asr.cueqc.train_pre_asr_v11_binary import (
    _split_label_masks,
    _window_batch_from_anchors,
    load_feature_bundle,
    main,
    train,
)

__all__ = [
    "_split_label_masks",
    "_window_batch_from_anchors",
    "load_feature_bundle",
    "main",
    "train",
]


if __name__ == "__main__":
    raise SystemExit(main())
