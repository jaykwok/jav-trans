from __future__ import annotations

from tools.asr.cueqc.gate_pre_asr_v12_candidate import _legacy_decision, _metrics


def test_pre_asr_v12_gate_metrics_keep_and_drop_recall():
    rows = [
        {"truth": "drop", "prediction": "drop"},
        {"truth": "drop", "prediction": "keep"},
        {"truth": "keep", "prediction": "keep"},
        {"truth": "keep", "prediction": "drop"},
    ]

    metrics = _metrics(rows, "prediction")

    assert metrics["drop_precision"] == 0.5
    assert metrics["drop_recall"] == 0.5
    assert metrics["semantic_keep_recall"] == 0.5
    assert metrics["false_drop_count"] == 1
    assert metrics["false_keep_count"] == 1


def test_pre_asr_v12_gate_legacy_rules_are_explicit_diagnostic_only():
    features = {
        "duration_s": 1.0,
        "refined_duration_s": 1.0,
        "scorer_speech_p90": 0.0,
        "scorer_speech_active_ratio_05": 0.0,
    }
    config = {
        "drop_threshold": 0.95,
        "hard_keep_veto": False,
        "hard_drop_rule": False,
        "keep_veto": False,
    }

    drop_without_rules, reason_without_rules, _, _ = _legacy_decision(
        p_drop=0.99,
        threshold=0.95,
        features=features,
        config=config,
        force_rules=False,
    )
    drop_with_rules, reason_with_rules, veto, _ = _legacy_decision(
        p_drop=0.99,
        threshold=0.95,
        features=features,
        config=config,
        force_rules=True,
    )

    assert drop_without_rules is True
    assert reason_without_rules == "model_drop_threshold"
    assert drop_with_rules is False
    assert reason_with_rules == "hard_keep_veto"
    assert veto == "duration_at_or_above_hard_keep_min"
