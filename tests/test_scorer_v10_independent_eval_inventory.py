from __future__ import annotations

import json

from tools.audits.audit_scorer_v10_independent_eval_inventory import audit


def test_independent_eval_inventory_excludes_used_identities(tmp_path) -> None:
    canonical = tmp_path / "canonical.jsonl"
    canonical.write_text(json.dumps({"core_ids": ["used-core"], "background_source_ids": ["used-neg"], "background_id": ""}) + "\n", encoding="utf-8")
    cores = tmp_path / "cores.jsonl"
    cores.write_text("\n".join([json.dumps({"audio_id": "used-core"}), json.dumps({"audio_id": "free-core"})]) + "\n", encoding="utf-8")
    labels = tmp_path / "labels"
    labels.mkdir()
    audio = tmp_path / "free.wav"
    audio.write_bytes(b"x")
    (labels / "one.json").write_text(json.dumps({"pre_asr_labels": [
        {"label": "definite_drop", "candidate_id": "used-neg", "audio": str(audio), "omni_flags": ["breathing"]},
        {"label": "definite_drop", "candidate_id": "free-neg", "audio": str(audio), "omni_flags": ["music"]},
    ]}), encoding="utf-8")
    summary = audit(canonical=canonical, cores=cores, joint_labels=labels, output_dir=tmp_path / "out")
    assert summary["unused_core_count"] == 1
    assert summary["unused_negative_count"] == 1
    assert summary["unused_negative_type_counts"] == {"music": 1}
    assert summary["complete_stratified_eval_ready"] is False
