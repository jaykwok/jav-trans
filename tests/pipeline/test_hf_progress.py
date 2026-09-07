from __future__ import annotations

import threading

import pytest

from core import events
from utils import hf_progress


def _events_sink(monkeypatch):
    captured: list[dict] = []
    monkeypatch.setattr(events, "emit", lambda event: captured.append(event))
    monkeypatch.setattr(hf_progress, "_current_job_id", lambda: "job-1")
    return captured


def test_snapshot_transfer_bar_is_suppressed(monkeypatch):
    """snapshot_download drives a `.transfer` bar whose total is a rolling
    guess huggingface_hub keeps re-inflating (see its own `_update_transfer_bar`),
    never a real percentage. Displaying it alongside the reconstruct bar is what
    made the UI progress bar swing between two unrelated numbers."""
    captured = _events_sink(monkeypatch)

    bar = hf_progress.HfDownloadProgressTqdm(
        total=0,
        initial=0,
        unit="B",
        unit_scale=True,
        desc="Downloading bytes",
        name="huggingface_hub.snapshot_download.transfer",
        disable=True,
    )
    bar.n = 500
    bar.update(0)
    bar.close()

    assert captured == []


def test_xet_get_reconstruction_bar_is_suppressed_in_favour_of_transfer(monkeypatch):
    """A standalone Xet-accelerated hf_hub_download (e.g. the llama.cpp GGUF
    path, if that repo uses Xet storage) drives a reconstruction bar
    (`huggingface_hub.xet_get`) and a `.transfer` twin from
    XetDownloadProgressReporter, both seeded with the same real, known-upfront
    total (`expected_size` from the HTTP HEAD) -- unlike snapshot_download's
    aggregate pair. huggingface_hub's own docstring says reconstruction lags
    behind transfer, and it only advances in batches as chunks flush to disk,
    so showing it (and suppressing the continuously-updating transfer bar, as
    the old suffix-only rule did) left the UI stuck at 0% for most of a
    multi-GB download while the network was actually near-saturated."""
    captured = _events_sink(monkeypatch)

    bar = hf_progress.HfDownloadProgressTqdm(
        total=4_600_000_000,
        initial=0,
        unit="B",
        unit_scale=True,
        desc="model.gguf: reconstructing file",
        name="huggingface_hub.xet_get",
        disable=True,
    )
    bar.n = 500
    bar.update(0)
    bar.close()

    assert captured == []


def test_xet_get_transfer_bar_reports_the_live_download(monkeypatch):
    """The twin above's `.transfer` counterpart is the reliable, continuously-
    updating signal for a standalone Xet download -- it must emit, with a
    clean filename label (huggingface_hub bakes ": downloading bytes" into
    its `desc`, which must not leak into the UI)."""
    captured = _events_sink(monkeypatch)

    bar = hf_progress.HfDownloadProgressTqdm(
        total=4_600_000_000,
        initial=0,
        unit="B",
        unit_scale=True,
        desc="model.gguf: downloading bytes",
        name="huggingface_hub.xet_get.transfer",
        disable=True,
    )
    bar.n = 2_300_000_000
    bar.update(0)
    bar.close()

    assert captured[0]["extra"]["file"] == "model.gguf"
    progress = next(event for event in captured if event["phase"] == "progress")
    assert progress["extra"]["pct"] == 50
    assert captured[-1]["phase"] == "done"


def test_snapshot_reconstruct_bar_emits_with_a_friendly_label(monkeypatch):
    captured = _events_sink(monkeypatch)

    bar = hf_progress.HfDownloadProgressTqdm(
        total=0,
        initial=0,
        unit="B",
        unit_scale=True,
        desc="Reconstructing (incomplete total...)",
        name="huggingface_hub.snapshot_download",
        disable=True,
    )
    # snapshot_download's _AggregatedTqdm grows `.total` directly as each
    # file registers, bypassing our __init__.
    bar.total = 1000
    bar.n = 250
    bar.update(0)
    bar.close()

    assert [event["phase"] for event in captured] == ["start", "progress", "done"]
    for event in captured:
        assert event["extra"]["file"] == "模型文件"
    progress_extra = captured[1]["extra"]
    assert progress_extra["pct"] == 25
    assert progress_extra["size_mb"] == round(1000 / (1024 * 1024), 2)


def test_standalone_file_download_keeps_its_real_filename(monkeypatch):
    """hf_hub_download called directly (not via snapshot_download, e.g. the
    llama.cpp GGUF path) never goes through `_AggregatedTqdm` -- one bar, real
    file size from the start, real filename."""
    captured = _events_sink(monkeypatch)

    bar = hf_progress.HfDownloadProgressTqdm(
        total=4_600_000_000,
        initial=0,
        unit="B",
        unit_scale=True,
        desc="Hy-MT2-7B-Q4_K_M.gguf",
        name="huggingface_hub.http_get",
        disable=True,
    )
    bar.n = 4_600_000_000
    bar.update(0)
    bar.close()

    assert captured[0]["extra"]["file"] == "Hy-MT2-7B-Q4_K_M.gguf"
    assert captured[-1]["phase"] == "done"


def test_a_cancelled_download_is_stopped_through_its_progress_bar(monkeypatch):
    """huggingface_hub has no cancellation token, but it does call the bar.

    A 4.6GB GGUF is minutes of transfer; without this, cancelling the task that
    asked for it leaves the bytes coming. The partial file is resumable, so
    stopping costs nothing but the time already spent.
    """
    _events_sink(monkeypatch)
    cancel = threading.Event()
    bar = hf_progress.tqdm_class(cancel)(
        total=1000,
        initial=0,
        unit="B",
        unit_scale=True,
        desc="sample-model.gguf",
        name="huggingface_hub.xet_get.transfer",
        disable=True,
    )

    bar.update(10)  # not cancelled: an ordinary chunk
    cancel.set()

    with pytest.raises(hf_progress.DownloadCancelled):
        bar.update(10)


def test_a_bar_reports_under_the_run_that_built_it(monkeypatch):
    """huggingface_hub drives `update` from its own transfer threads.

    Reading the identity when a chunk lands means reading it on a thread that
    has none, and falling through to a module-level value the next job to start
    a download has since overwritten. A multi-GB transfer outlives whatever was
    current when it began, so the bar carries its own run.
    """
    monkeypatch.setattr(events, "_thread_local", threading.local())
    captured: list[dict] = []
    monkeypatch.setattr(events, "emit", lambda event: captured.append(event))
    events.set_current_run("sample-a", "run-a")
    events._thread_local.video = "sample-a.mp4"

    bar = hf_progress.HfDownloadProgressTqdm(
        total=100,
        initial=0,
        unit="B",
        unit_scale=True,
        desc="sample-model.gguf",
        name="huggingface_hub.http_get",
        disable=True,
    )
    # The bytes are still arriving when the next job takes over the thread.
    events.set_current_run("sample-b", "run-b")
    events._thread_local.video = "sample-b.mp4"
    bar.update(100)
    bar.close()

    assert [event["phase"] for event in captured] == ["start", "progress", "done"]
    assert {
        (event["job_id"], event["run_id"], event["video"]) for event in captured
    } == {("sample-a", "run-a", "sample-a.mp4")}


def test_without_an_event_the_bar_is_the_plain_one():
    # The ASR download path passes no event, and must keep behaving exactly as
    # it did - including being the same class the other tests here drive.
    assert hf_progress.tqdm_class() is hf_progress.HfDownloadProgressTqdm


def test_propagate_job_id_to_current_thread_sets_the_events_thread_local():
    events.set_current_job_id("")
    try:
        hf_progress.propagate_job_id_to_current_thread("job-42")
        assert events._current_job_id() == "job-42"
    finally:
        events.set_current_job_id("")


def test_current_job_id_reads_through_to_events_thread_local():
    events.set_current_job_id("job-99")
    try:
        assert hf_progress.current_job_id() == "job-99"
    finally:
        events.set_current_job_id("")
