from __future__ import annotations

import pytest

from utils.gpu_safety import resolve_inference_device


def test_boundary_inference_auto_never_falls_back_to_cpu(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU fallback is disabled"):
        resolve_inference_device("auto", stage="test stage")
    with pytest.raises(RuntimeError, match="CPU fallback is disabled"):
        resolve_inference_device("cuda", stage="test stage")
    assert str(resolve_inference_device("cpu", stage="test stage")) == "cpu"
