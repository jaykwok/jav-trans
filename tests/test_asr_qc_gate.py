import builtins

from whisper import qc as asr_qc
def _segments(total: int, empty: int = 0) -> list[dict]:
    return [
        {"text": "" if index < empty else f"text-{index}"}
        for index in range(total)
    ]


def test_asr_qc_gate_low_empty_rate_no_input(monkeypatch):
    monkeypatch.delenv("QC_IGNORE_EMPTY", raising=False)
    monkeypatch.setenv("QC_EMPTY_THRESHOLD", "0.05")

    def fail_input(*_args, **_kwargs):
        raise AssertionError("input should not be called")

    monkeypatch.setattr(builtins, "input", fail_input)

    assert asr_qc.asr_qc_gate(_segments(20), headless=False) is True


def test_asr_qc_gate_ignore_empty_bypasses_without_input(monkeypatch):
    monkeypatch.setenv("QC_IGNORE_EMPTY", "1")
    monkeypatch.setenv("QC_EMPTY_THRESHOLD", "0.05")

    def fail_input(*_args, **_kwargs):
        raise AssertionError("input should not be called")

    monkeypatch.setattr(builtins, "input", fail_input)

    assert asr_qc.asr_qc_gate(_segments(20, empty=2), headless=True) is True

