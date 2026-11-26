"""Everything a translation worker thread has to be told before it can work.

Four separate things used to be carried across the thread-pool boundary by hand,
each with its own binder and its own failure mode when it was forgotten:

* **the run's identity** (job, run, video) - without it a model download started
  from a worker emits `model_download` events with an empty `job_id`, which the
  page drops on the floor instead of showing a progress bar;
* **the backend lease** - a fresh pool thread holds none, and the accessor falls
  back to resolving the backend *by name*, which is precisely the lookup the
  lease exists to replace: the task translates through whatever instance the
  settings currently name rather than the one it claimed;
* **the endpoint snapshot** - naming an instance settles which adapter answers,
  not which model or endpoint it is pointed at, and the settings page rewrites
  those while a job runs. A worker with only the lease builds its request from
  whatever the panel says by the time it happens to run;
* **the prompt profile** - the cache key is derived from it, and the repair pass
  has to write corrected text back under the key the first pass used.

Getting three of the four right is not a partial success: each of these has
produced a distinct bug on its own. So they travel together, captured on the
thread that has them and adopted on the thread that needs them, and adding a
fifth means changing one place.

`RunContext` is a value: capturing it twice on one thread gives two equal
objects, and nothing about the run can change through it.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from llm import backends as backends_module
from llm import request_config
from utils import hf_progress


@dataclass(frozen=True)
class RunContext:
    """The thread-scoped state of one translation run."""

    identity: tuple[str, str, str]
    lease: tuple[str, Any] | None
    config: Any | None
    profile: Any | None

    @property
    def job_id(self) -> str:
        return self.identity[0]

    @property
    def run_id(self) -> str:
        return self.identity[1]

    @property
    def video(self) -> str:
        return self.identity[2]

    @classmethod
    def capture(cls) -> "RunContext":
        """What this thread currently holds. Safe on a thread holding nothing."""
        return cls(
            identity=hf_progress.current_identity(),
            lease=backends_module.current_lease(),
            config=request_config.current(),
            profile=request_config.current_profile(),
        )

    def adopt(self) -> None:
        """Make this thread the one described by this context."""
        hf_progress.adopt_identity(self.identity)
        backends_module.bind_lease(self.lease)
        request_config.bind(self.config)
        request_config.bind_profile(self.profile)

    @contextmanager
    def bound(self) -> Iterator["RunContext"]:
        """Adopt for the duration of a block, then restore what was there.

        For a thread that already has a context of its own - a nested lease, a
        test - rather than a pool worker, which starts with nothing and is
        discarded afterwards.
        """
        previous = RunContext.capture()
        self.adopt()
        try:
            yield self
        finally:
            previous.adopt()


__all__ = ["RunContext"]
