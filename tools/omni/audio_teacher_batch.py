#!/usr/bin/env python3
"""Bounded, provider-neutral batch execution for audio Teacher calls."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time
from typing import Callable, Generic, Iterator, Sequence, TypeVar


ItemT = TypeVar("ItemT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class CompletedAudioTeacherItem(Generic[ItemT, ResultT]):
    index: int
    item: ItemT
    result: ResultT


def resolve_worker_count(
    *, requested: int, provider_limit: int, item_count: int
) -> int:
    if requested < 0:
        raise ValueError("audio Teacher worker count cannot be negative")
    if provider_limit <= 0:
        raise ValueError("audio Teacher provider concurrency must be positive")
    if requested > provider_limit:
        raise ValueError(
            f"requested {requested} workers exceeds provider-safe limit "
            f"{provider_limit}"
        )
    chosen = requested or provider_limit
    return max(1, min(chosen, max(1, item_count)))


def iter_completed_audio_teacher_items(
    *,
    items: Sequence[ItemT],
    worker: Callable[[ItemT], ResultT],
    max_workers: int,
    sequential_interval_s: float = 0.0,
) -> Iterator[CompletedAudioTeacherItem[ItemT, ResultT]]:
    """Yield completed work while keeping all persistence in the caller.

    Provider adapters remain responsible for their own RPM/TPM/RPD pacing.
    A sequential interval is retained for one-worker compatible providers;
    native multi-key execution does not apply a second global throttle.
    """

    if max_workers <= 0:
        raise ValueError("audio Teacher max_workers must be positive")
    if sequential_interval_s < 0:
        raise ValueError("audio Teacher interval cannot be negative")
    if max_workers == 1:
        for index, item in enumerate(items):
            yield CompletedAudioTeacherItem(
                index=index,
                item=item,
                result=worker(item),
            )
            if index + 1 < len(items) and sequential_interval_s > 0:
                time.sleep(sequential_interval_s)
        return

    executor = ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="audio-teacher",
    )
    futures: dict[Future[ResultT], tuple[int, ItemT]] = {}
    try:
        for index, item in enumerate(items):
            futures[executor.submit(worker, item)] = (index, item)
        for future in as_completed(futures):
            index, item = futures[future]
            yield CompletedAudioTeacherItem(
                index=index,
                item=item,
                result=future.result(),
            )
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
