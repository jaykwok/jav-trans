"""Process-wide revision clock, restart epoch persistence, and deletion tombstones."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from core.atomic_io import write_json_atomic
from web.models import JobState

log = logging.getLogger(__name__)

EPOCH_FILE_NAME = "state-epoch.json"
TOMBSTONE_LIMIT = 512


def entomb_in(
    tombstones: dict[str, int],
    job_id: str,
    revision: int,
    *,
    limit: int = TOMBSTONE_LIMIT,
) -> None:
    """Record a deletion in ``tombstones``, evicting the oldest entries past ``limit``."""
    tombstones.pop(job_id, None)
    tombstones[job_id] = revision
    while len(tombstones) > limit:
        tombstones.pop(next(iter(tombstones)))


def visible_revision(
    jobs: Iterable[JobState],
    tombstones: Mapping[str, int],
) -> int:
    """Return the highest revision currently visible in ``jobs`` or ``tombstones``."""
    revisions = [0, *(job.revision for job in jobs), *tombstones.values()]
    return max(revisions)


def load_and_advance_epoch(
    epoch_path: Path,
    *,
    time_fn: Callable[[], float] = time.time,
) -> int:
    """Establish a strictly monotonic service instance epoch and persist it atomically."""
    previous = 0
    try:
        previous = int(json.loads(epoch_path.read_text(encoding="utf-8"))["epoch"])
    except (OSError, ValueError, KeyError, TypeError):
        previous = 0
    current = max(int(time_fn() * 1000), previous + 1)
    try:
        write_json_atomic(
            epoch_path,
            {"epoch": current},
            indent=None,
            ensure_ascii=False,
            trailing_newline=False,
        )
    except OSError:
        # Unwritable means the next start may reuse this number. Serving the
        # epoch anyway is still an improvement on serving none.
        log.exception("Failed to persist the state epoch to %s", epoch_path)
    return current


class StateClock:
    """Monotonic revision counter, restart epoch, and bounded deletion tombstones."""

    def __init__(self, *, tombstone_limit: int = TOMBSTONE_LIMIT) -> None:
        self._epoch_lock = threading.Lock()
        self.epoch = 0
        self.revision = 0
        self.tombstone_limit = tombstone_limit
        self.tombstones: dict[str, int] = {}

    def resolve_epoch(
        self,
        epoch_path: Path,
        *,
        current_epoch: int | None = None,
        time_fn: Callable[[], float] = time.time,
    ) -> int:
        with self._epoch_lock:
            existing = self.epoch if current_epoch is None else current_epoch
            if existing:
                self.epoch = existing
                return existing
            self.epoch = load_and_advance_epoch(epoch_path, time_fn=time_fn)
            return self.epoch

    def next_revision(self) -> int:
        self.revision += 1
        return self.revision

    def publish(self, job: JobState) -> JobState:
        job.revision = self.next_revision()
        return job

    def seed_revision(self, jobs: Iterable[JobState]) -> int:
        self.revision = max([self.revision, 0, *(job.revision for job in jobs)])
        return self.revision

    def entomb(self, job_id: str, revision: int) -> None:
        entomb_in(self.tombstones, job_id, revision, limit=self.tombstone_limit)

    def deleted_revision(self, job_id: str) -> int | None:
        return self.tombstones.get(job_id)

    def visible_revision(self, jobs: Iterable[JobState]) -> int:
        return visible_revision(jobs, self.tombstones)
