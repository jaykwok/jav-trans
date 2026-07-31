"""Memory safety caps for training and dataset-build tools.

Windows degrades instead of failing, on both memories, and in both cases the job
appears to still be working:

* VRAM - the WDDM driver silently spills allocations beyond physical VRAM into
  shared system memory, so jobs slow to a crawl long before they fail. Capping
  the PyTorch caching allocator makes overflow surface immediately as a regular
  CUDA OOM so batch sizes get fixed instead of silently degrading.
* Host RAM - the pager swaps to disk. A trainer streaming an 8 GB feature memmap
  on a 17 GB box can thrash for hours at a few percent of its normal step rate,
  which reads as "training is slow" rather than "the batch does not fit".

Both caps therefore convert a silent slowdown into a loud, immediate error.

The runtime ASR worker has its own budget (``ASR_STAGE_WORKER_VRAM_RATIO``);
this module covers everything outside that worker.
"""
from __future__ import annotations

import os
import threading

DEFAULT_VRAM_SAFETY_RATIO = 0.95
DEFAULT_HOST_MEMORY_RATIO = 0.95


def resolve_inference_device(requested: str | None, *, stage: str):
    """Resolve a model device without silently falling back from CUDA to CPU."""
    import torch

    value = str(requested or "auto").strip().lower()
    if value == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{stage} requires CUDA for runtime inference; CPU fallback is disabled"
            )
        value = "cuda"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"{stage} requested CUDA but CUDA is unavailable; CPU fallback is disabled"
        )
    return torch.device(value)


def _resolve_ratio(ratio: float | None, env_key: str, default: float) -> float:
    """Explicit value, else env override, else default - clamped to a sane band."""
    if ratio is None:
        raw = os.getenv(env_key, "").strip()
        try:
            ratio = float(raw) if raw else default
        except ValueError:
            ratio = default
    return min(1.0, max(0.1, float(ratio)))


def apply_vram_safety_cap(ratio: float | None = None) -> float | None:
    """Cap the CUDA caching allocator at ratio x physical VRAM on all devices.

    ``ratio`` defaults to env ``VRAM_SAFETY_RATIO``, then 0.95. Returns the
    applied ratio, or None when CUDA is unavailable.
    """
    import torch

    if not torch.cuda.is_available():
        return None
    ratio = _resolve_ratio(ratio, "VRAM_SAFETY_RATIO", DEFAULT_VRAM_SAFETY_RATIO)
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.set_per_process_memory_fraction(ratio, device_index)
    return ratio


class HostMemoryExceeded(MemoryError):
    """Raised when a job crosses its host RAM budget instead of swapping."""


class HostMemoryGuard:
    """Soft OOM for host RAM: fail loudly rather than let Windows swap.

    Two independent trips, because a job can starve either itself or the box:

    * ``process`` - this process's private memory against ``ratio`` x (its
      private memory plus what was actually free when the guard was created).
      That is the most it could ever have obtained without evicting someone else.
    * ``system``  - machine-wide available memory against a ``1 - ratio`` floor
      of total RAM. Trips when something outside this process ate the headroom,
      which would otherwise show up as our own unexplained slowdown.

    PRIVATE memory, not RSS. A trainer streaming a memory-mapped feature file
    accumulates the touched pages into its working set, so RSS climbs towards
    the size of the file and trips a budget the job never actually spent: those
    pages are clean, file-backed, and reclaimable without ever touching swap.
    Committed private bytes are what genuinely compete for RAM. Measuring the
    wrong one aborted a healthy 4-arm run at 6.80 GiB "used", nearly all of it
    the 8 GB feature map.

    ``check()`` is a couple of counter reads, cheap enough to call every step.
    The watchdog exists for long single calls - a dataset pack, one huge
    forward - where no step boundary comes around to check.
    """

    def __init__(
        self,
        ratio: float | None = None,
        *,
        poll_seconds: float = 2.0,
    ) -> None:
        import psutil

        self.ratio = _resolve_ratio(
            ratio, "HOST_MEMORY_SAFETY_RATIO", DEFAULT_HOST_MEMORY_RATIO
        )
        self._process = psutil.Process()
        self._psutil = psutil
        self._poll_seconds = max(0.1, float(poll_seconds))
        self.metric = self._select_metric()

        baseline = self.process_bytes()
        virtual = psutil.virtual_memory()
        self.total_bytes = int(virtual.total)
        self.budget_bytes = int(self.ratio * (baseline + int(virtual.available)))
        self.system_floor_bytes = int((1.0 - self.ratio) * self.total_bytes)
        self.peak_bytes = baseline

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tripped: BaseException | None = None

    def _select_metric(self) -> str:
        """Cheapest available counter that excludes clean file-backed pages."""
        info = self._process.memory_info()
        if hasattr(info, "private"):  # Windows commit charge
            return "private"
        try:
            self._process.memory_full_info()
        except Exception:
            return "rss"  # last resort; over-counts shared mappings
        return "uss"

    def process_bytes(self) -> int:
        if self.metric == "private":
            return int(self._process.memory_info().private)
        if self.metric == "uss":
            return int(self._process.memory_full_info().uss)
        return int(self._process.memory_info().rss)

    def check(self) -> None:
        """Raise if either budget is exceeded; record the peak otherwise."""
        used = self.process_bytes()
        self.peak_bytes = max(self.peak_bytes, used)
        if used > self.budget_bytes:
            raise HostMemoryExceeded(
                f"host RAM budget exceeded: process {self.metric} "
                f"{used / 2**30:.2f} GiB > {self.budget_bytes / 2**30:.2f} GiB "
                f"({self.ratio:.0%} of what was available at start). "
                "Reduce batch size, window length, or worker count."
            )
        available = int(self._psutil.virtual_memory().available)
        if available < self.system_floor_bytes:
            raise HostMemoryExceeded(
                f"machine is out of RAM: {available / 2**30:.2f} GiB available < "
                f"{self.system_floor_bytes / 2**30:.2f} GiB floor "
                f"(process holds {used / 2**30:.2f} GiB). "
                "Free memory on the host or reduce this job's footprint."
            )

    @property
    def tripped(self) -> BaseException | None:
        """The watchdog's exception, if it fired between explicit checks."""
        return self._tripped

    def _watch(self) -> None:
        while not self._stop.wait(self._poll_seconds):
            try:
                self.check()
            except HostMemoryExceeded as error:  # surfaced at the next check()
                self._tripped = error
                return

    def start(self) -> "HostMemoryGuard":
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._watch, name="host-memory-guard", daemon=True
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._poll_seconds * 2)
            self._thread = None

    def __enter__(self) -> "HostMemoryGuard":
        return self.start()

    def __exit__(self, *_exc_info: object) -> None:
        self.stop()


def apply_host_memory_cap(
    ratio: float | None = None, *, watchdog: bool = True
) -> HostMemoryGuard:
    """Create (and by default start) a host RAM guard for this process.

    ``ratio`` defaults to env ``HOST_MEMORY_SAFETY_RATIO``, then 0.95.
    """
    guard = HostMemoryGuard(ratio)
    return guard.start() if watchdog else guard
