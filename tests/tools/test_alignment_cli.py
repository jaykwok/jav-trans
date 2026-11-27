"""CUDA is mandatory in the retained alignment CLIs."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_CUDA_CLI_CASES = [
    ("train_ctc_aligner", ("--cache-dir",), "--output-dir"),
    ("build_alignment_features", ("--manifest",), "--output-dir"),
]


@pytest.fixture(params=_CUDA_CLI_CASES, ids=[case[0] for case in _CUDA_CLI_CASES])
def cuda_cli(request, tmp_path, monkeypatch):
    name, input_flags, output_flag = request.param
    module = importlib.import_module(f"tools.align.{name}")
    output = tmp_path / "result"
    argv = [name, output_flag, str(output)]
    for flag in input_flags:
        argv.extend([flag, str(tmp_path / "missing-input")])

    def premature_work(*_args, **_kwargs):
        pytest.fail("model/cache work started before the CUDA requirement was checked")

    for attribute in (
        "load_checkpoint",
        "build_head",
        "FeatureCache",
        "Qwen3AsrEncoder",
        "apply_vram_safety_cap",
    ):
        if hasattr(module, attribute):
            monkeypatch.setattr(module, attribute, premature_work)
    return module, argv, output


def test_training_and_extraction_fail_before_reading_inputs_without_cuda(cuda_cli, monkeypatch):
    import torch

    module, argv, output = cuda_cli
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="CUDA"):
        module.main()
    assert not output.exists()


@pytest.mark.parametrize("flags", [["--device", "cpu"], ["--allow-cpu"]])
def test_training_and_extraction_reject_removed_cpu_options(cuda_cli, monkeypatch, capsys, flags):
    import torch

    module, argv, output = cuda_cli
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(sys, "argv", [*argv, *flags])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert flags[0] in capsys.readouterr().err
    assert not output.exists()
