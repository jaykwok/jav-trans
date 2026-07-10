from tools.asr.cueqc.analyze_pre_asr_v12_representation import classify_error_cause


def test_representation_diagnostic_distinguishes_failure_layers():
    assert classify_error_cause(
        padded_prob_drop=0.7,
        trimmed_prob_drop=0.2,
        local_prob_drop=0.1,
        threshold=0.5,
    ) == "padding_sensitive_flip"
    assert classify_error_cause(
        padded_prob_drop=0.7,
        trimmed_prob_drop=0.6,
        local_prob_drop=0.2,
        threshold=0.5,
    ) == "temporal_context_flip"
    assert classify_error_cause(
        padded_prob_drop=0.7,
        trimmed_prob_drop=0.6,
        local_prob_drop=0.8,
        threshold=0.5,
    ) == "local_representation_error"
