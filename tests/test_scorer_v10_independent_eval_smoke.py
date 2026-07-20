from __future__ import annotations

import json

import numpy as np
import soundfile as sf

from tools.audits.build_scorer_v10_independent_eval_smoke import build


def test_independent_eval_smoke_uses_each_core_once(tmp_path) -> None:
    cores = tmp_path / "cores.jsonl"
    rows = []
    for index in range(2):
        path = tmp_path / f"core-{index}.wav"
        sf.write(path, np.zeros(1600, dtype=np.float32), 16000)
        rows.append({"audio_id": f"core-{index}", "audio": str(path), "duration_s": 0.1})
    cores.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    negatives = tmp_path / "negatives.jsonl"
    negatives.write_text("".join(json.dumps({"audio": str(tmp_path / "core-0.wav"), "audio_id": kind, "eval_type": kind}) + "\n" for kind in ("breathing", "music", "noise", "non_speech")), encoding="utf-8")
    summary = build(cores=cores, negatives=negatives, output_dir=tmp_path / "out", base_count=1)
    assert summary["unique_core_count"] == 2
    assert summary["max_core_use_count"] == 1
    assert summary["missing_formal_types"] == ["kissing", "moaning"]
