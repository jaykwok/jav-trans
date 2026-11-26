from __future__ import annotations

import pytest

from utils.gpu_safety import resolve_inference_device


@pytest.mark.parametrize("requested", ["auto", "cuda", "cuda:0"])
def test_boundary_inference_never_falls_back_to_cpu(monkeypatch, requested) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU fallback is disabled"):
        resolve_inference_device(requested, stage="test stage")


@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("requested", ["cpu", "mps"])
def test_model_workloads_reject_non_cuda_devices(monkeypatch, available, requested) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        resolve_inference_device(requested, stage="test stage")


@pytest.mark.parametrize("requested, expected", [("auto", "cuda"), ("cuda", "cuda"), ("cuda:1", "cuda:1")])
def test_model_device_preserves_cuda_selection(monkeypatch, requested, expected) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert str(resolve_inference_device(requested, stage="test stage")) == expected
