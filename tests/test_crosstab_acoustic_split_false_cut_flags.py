from __future__ import annotations

from tools.audits.crosstab_acoustic_split_false_cut_flags import analyze


def _label(**overrides):
    row = {
        "trainable": True,
        "current_label": "cut",
        "label": "continue",
        "flags": [],
    }
    row.update(overrides)
    return row


def test_crosstab_counts_nonspeech_share_of_false_cuts() -> None:
    labels = [
        # suspected false-cut at a breath junction
        _label(flags=["short_pause", "breath"]),
        # suspected false-cut, purely linguistic
        _label(flags=["same_sentence"]),
        # a real cut (runtime and omni agree) -> not a false-cut
        _label(current_label="cut", label="cut", flags=["moan"]),
        # a suspected missed-cut (opposite direction)
        _label(current_label="continue", label="cut", flags=["speaker_change"]),
        # non-trainable rows are ignored
        _label(trainable=False, flags=["laughter"]),
    ]

    summary = analyze(labels)

    assert summary["suspected_false_cut_count"] == 2
    assert summary["suspected_missed_cut_count"] == 1
    assert summary["false_cut_buckets"] == {
        "nonspeech_junction": 1,
        "linguistic_or_other": 1,
    }
    assert summary["false_cut_nonspeech_share"] == 0.5
    assert summary["false_cut_flag_histogram"]["breath"] == 1


def test_crosstab_share_is_none_without_false_cuts() -> None:
    labels = [_label(current_label="cut", label="cut", flags=["breath"])]

    summary = analyze(labels)

    assert summary["suspected_false_cut_count"] == 0
    assert summary["false_cut_nonspeech_share"] is None
