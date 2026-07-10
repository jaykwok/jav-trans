from tools.asr.cueqc.evaluate_pre_asr_manual_holdout import evaluate


def test_manual_holdout_gate_rejects_semantic_keep_deletion():
    labels = [
        {"candidate_id": "keep", "label": "definite_keep"},
        {"candidate_id": "drop", "label": "definite_drop"},
    ]
    paired = [
        {"id": "keep", "v12_prediction": "drop", "v12_prob_drop": 0.8},
        {"id": "drop", "v12_prediction": "drop", "v12_prob_drop": 0.9},
    ]

    summary = evaluate(paired, labels, ["keep", "drop"])

    assert summary["keep_false_drop_count"] == 1
    assert summary["drop_false_keep_count"] == 0
    assert summary["gate_pass"] is False
