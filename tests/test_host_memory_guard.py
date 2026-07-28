"""The host RAM cap must fail loudly rather than let Windows swap.

Same rationale as the existing VRAM cap: on this box a trainer that overruns RAM
does not crash, it pages to disk and keeps reporting steps at a fraction of the
normal rate. That reads as "training is slow", so the batch size never gets
fixed. A guard is only useful if it actually trips, so these tests drive it past
both budgets rather than merely constructing it.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.gpu_safety import (  # noqa: E402
    DEFAULT_HOST_MEMORY_RATIO,
    HostMemoryExceeded,
    HostMemoryGuard,
    apply_host_memory_cap,
)


def test_default_ratio_is_95_percent() -> None:
    guard = HostMemoryGuard()
    assert guard.ratio == DEFAULT_HOST_MEMORY_RATIO == 0.95


def test_ratio_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOST_MEMORY_SAFETY_RATIO", "0.5")
    assert HostMemoryGuard().ratio == 0.5


def test_unparseable_ratio_falls_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOST_MEMORY_SAFETY_RATIO", "not-a-number")
    assert HostMemoryGuard().ratio == DEFAULT_HOST_MEMORY_RATIO


def test_ratio_is_clamped_to_a_sane_band() -> None:
    assert HostMemoryGuard(5.0).ratio == 1.0
    assert HostMemoryGuard(0.0).ratio == 0.1


def test_a_healthy_process_passes() -> None:
    guard = HostMemoryGuard()
    guard.check()  # must not raise
    assert guard.peak_bytes > 0


def test_process_budget_trips() -> None:
    """Budget driven below current RSS: the process trip must fire."""
    guard = HostMemoryGuard()
    guard.budget_bytes = 1  # 1 byte; this process certainly holds more

    with pytest.raises(HostMemoryExceeded, match="host RAM budget exceeded"):
        guard.check()


def test_system_floor_trips_independently_of_the_process() -> None:
    """A box starved by some OTHER process must also stop this job."""
    guard = HostMemoryGuard()
    guard.system_floor_bytes = guard.total_bytes * 10  # unreachable floor

    with pytest.raises(HostMemoryExceeded, match="machine is out of RAM"):
        guard.check()


def test_the_error_is_a_memory_error() -> None:
    """Callers that already handle MemoryError should catch this unchanged."""
    assert issubclass(HostMemoryExceeded, MemoryError)


def test_metric_excludes_clean_file_backed_pages(tmp_path) -> None:
    """Reading a memory-mapped file must not consume the budget.

    Regression: the guard first measured RSS, which on Windows grows to include
    touched pages of a mapping. A 4-arm training run streaming the 8 GB feature
    map was aborted at "6.80 GiB used" without ever allocating it. Those pages
    are clean and reclaimable, so they must not count.
    """
    payload = tmp_path / "block.bin"
    block = np.ones(48 * 2**20, dtype=np.uint8)  # 48 MiB on disk
    payload.write_bytes(block.tobytes())
    del block

    guard = HostMemoryGuard()
    before = guard.process_bytes()

    mapped = np.memmap(payload, dtype=np.uint8, mode="r")
    total = int(mapped.sum())  # touch every page
    after = guard.process_bytes()
    del mapped

    assert total > 0
    # Allow ordinary allocator noise, but nothing like the 48 MiB mapping.
    assert after - before < 24 * 2**20, (
        f"metric {guard.metric} grew {(after - before) / 2**20:.1f} MiB "
        "while reading a clean file mapping"
    )


def test_metric_does_count_real_allocations() -> None:
    """The counterpart: the guard must still see genuine heap growth."""
    guard = HostMemoryGuard()
    before = guard.process_bytes()
    ballast = bytearray(128 * 2**20)  # 128 MiB, committed
    ballast[::4096] = b"\x01" * len(ballast[::4096])
    after = guard.process_bytes()
    del ballast

    assert after - before > 64 * 2**20


def test_peak_is_tracked_across_checks() -> None:
    guard = HostMemoryGuard()
    guard.check()
    first = guard.peak_bytes
    ballast = bytearray(64 * 2**20)  # 64 MiB
    guard.check()
    assert guard.peak_bytes >= first
    del ballast


def test_watchdog_records_a_trip_without_raising_in_its_own_thread() -> None:
    """A long single call has no step boundary, so the thread must catch it."""
    guard = HostMemoryGuard(poll_seconds=0.1)
    guard.budget_bytes = 1
    guard.start()
    try:
        deadline = 30
        while guard.tripped is None and deadline:
            import time

            time.sleep(0.1)
            deadline -= 1
    finally:
        guard.stop()

    assert isinstance(guard.tripped, HostMemoryExceeded)


def test_context_manager_starts_and_stops_the_watchdog() -> None:
    with HostMemoryGuard(poll_seconds=0.1) as guard:
        assert guard._thread is not None
    assert guard._thread is None


def test_factory_returns_a_started_guard() -> None:
    guard = apply_host_memory_cap(watchdog=True)
    try:
        assert guard._thread is not None
        assert guard.budget_bytes > 0
    finally:
        guard.stop()


def test_factory_can_skip_the_watchdog() -> None:
    guard = apply_host_memory_cap(watchdog=False)
    assert guard._thread is None
