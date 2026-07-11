from types import SimpleNamespace

from pipeline import memory_safety


def test_physical_ram_snapshot_uses_total_minus_available(monkeypatch):
    fake_psutil = SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(
            total=16 * 1024 * 1024,
            available=4 * 1024 * 1024,
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "psutil", fake_psutil)
    snapshot = memory_safety.physical_ram_snapshot(0.95)
    assert snapshot["physical_ram_total_mb"] == 16.0
    assert snapshot["physical_ram_used_mb"] == 12.0
    assert snapshot["physical_ram_budget_mb"] == 15.2


def test_non_windows_shared_vram_is_not_applicable(monkeypatch):
    monkeypatch.setattr(memory_safety.os, "name", "posix")
    assert memory_safety.shared_vram_snapshot(required=True) == {
        "shared_vram_mb": 0.0,
        "shared_vram_monitor": "not_applicable",
    }


def test_runtime_shared_vram_reports_growth_over_process_baseline(monkeypatch):
    values = iter((74.0, 79.5))
    monkeypatch.setattr(
        memory_safety,
        "shared_vram_snapshot",
        lambda **_kwargs: {
            "shared_vram_mb": next(values),
            "shared_vram_monitor": "fake",
        },
    )
    monkeypatch.setattr(
        memory_safety,
        "physical_ram_snapshot",
        lambda _ratio: {},
    )
    memory_safety._SHARED_VRAM_BASELINE_BY_PID.clear()
    assert memory_safety.runtime_memory_snapshot()["shared_vram_mb"] == 0.0
    second = memory_safety.runtime_memory_snapshot()
    assert second["shared_vram_baseline_mb"] == 74.0
    assert second["shared_vram_raw_mb"] == 79.5
    assert second["shared_vram_mb"] == 5.5
