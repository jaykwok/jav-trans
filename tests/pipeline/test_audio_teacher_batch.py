from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import pytest

from tools.omni.audio_teacher_batch import (
    iter_completed_audio_teacher_items,
    resolve_worker_count,
)
from tools.omni.audio_teacher_transport import AudioTeacherResult
from tools.omni.run_audio_teacher import parse_args as parse_generic_args
from tools.omni.run_audio_teacher import run as run_generic_teacher
from tools.omni.inspect_gemini_quota import summarize_quota


def test_worker_count_uses_provider_limit_and_rejects_oversubscription() -> None:
    assert resolve_worker_count(requested=0, provider_limit=10, item_count=25) == 10
    assert resolve_worker_count(requested=0, provider_limit=10, item_count=3) == 3
    assert resolve_worker_count(requested=1, provider_limit=10, item_count=25) == 1
    with pytest.raises(ValueError, match="exceeds provider-safe limit"):
        resolve_worker_count(requested=11, provider_limit=10, item_count=25)


def test_completed_batch_runs_concurrently_and_preserves_item_identity() -> None:
    barrier = threading.Barrier(2)

    def worker(value: int) -> int:
        if value < 2:
            barrier.wait(timeout=2.0)
        time.sleep(0.01 * (2 - min(value, 2)))
        return value * 10

    completed = list(
        iter_completed_audio_teacher_items(
            items=[0, 1, 2],
            worker=worker,
            max_workers=2,
        )
    )
    assert sorted(
        (item.index, item.item, item.result) for item in completed
    ) == [(0, 0, 0), (1, 1, 10), (2, 2, 20)]


def test_completed_batch_cancels_pending_work_after_failure() -> None:
    started: list[int] = []

    def worker(value: int) -> int:
        started.append(value)
        if value == 0:
            raise RuntimeError("failed closed")
        time.sleep(0.05)
        return value

    with pytest.raises(RuntimeError, match="failed closed"):
        list(
            iter_completed_audio_teacher_items(
                items=list(range(20)),
                worker=worker,
                max_workers=2,
            )
        )
    assert len(started) < 20


class _ConcurrentFakeTransport:
    profile = "gemini"
    model = "gemini-3.6-flash"
    transport_name = "google_ai_interactions_inline_audio"
    audio_content_mode = "native_inline_audio"
    execution_contract = "google_ai_interactions_inline_audio_medium_json_v1"

    def __init__(self, workers: int) -> None:
        self.api_key_count = workers
        self.max_concurrency = workers
        self._barrier = threading.Barrier(workers)

    def call_json(self, **kwargs) -> AudioTeacherResult:
        self._barrier.wait(timeout=2.0)
        return AudioTeacherResult(parsed={"ok": True}, raw={"ok": True})


def test_generic_teacher_uses_provider_key_count_for_parallel_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_paths = [tmp_path / "a.wav", tmp_path / "b.wav"]
    for path in audio_paths:
        path.write_bytes(b"RIFF-test")
    transport = _ConcurrentFakeTransport(2)
    monkeypatch.setattr(
        "tools.omni.run_audio_teacher.create_audio_teacher_transport",
        lambda **_kwargs: transport,
    )
    output = tmp_path / "generic-output"
    args = parse_generic_args(
        [
            "--env-file",
            "gemini",
            "--file",
            str(audio_paths[0]),
            "--file",
            str(audio_paths[1]),
            "--prompt",
            "test",
            "--output-dir",
            str(output),
        ]
    )
    summary = run_generic_teacher(args)
    assert summary["worker_count"] == 2
    assert summary["result_count"] == 2
    assert len((output / "results.jsonl").read_text().splitlines()) == 2
    assert len((output / "raw_responses.jsonl").read_text().splitlines()) == 2


def test_generic_teacher_persists_successes_when_another_parallel_item_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_paths = [tmp_path / "good.wav", tmp_path / "bad.wav"]
    for path in audio_paths:
        path.write_bytes(b"RIFF-test")

    class Transport(_ConcurrentFakeTransport):
        def __init__(self) -> None:
            self.api_key_count = 2
            self.max_concurrency = 2

        def call_json(self, **kwargs) -> AudioTeacherResult:
            if Path(kwargs["audio_path"]).stem == "bad":
                raise RuntimeError("provider failure")
            time.sleep(0.02)
            return AudioTeacherResult(parsed={"ok": True}, raw={"ok": True})

    monkeypatch.setattr(
        "tools.omni.run_audio_teacher.create_audio_teacher_transport",
        lambda **_kwargs: Transport(),
    )
    output = tmp_path / "failure-output"
    args = parse_generic_args(
        [
            "--env-file",
            "gemini",
            "--file",
            str(audio_paths[0]),
            "--file",
            str(audio_paths[1]),
            "--prompt",
            "test",
            "--max-attempts",
            "1",
            "--output-dir",
            str(output),
        ]
    )
    with pytest.raises(RuntimeError, match="1 item"):
        run_generic_teacher(args)
    result_rows = (output / "results.jsonl").read_text().splitlines()
    assert len(result_rows) == 1
    assert json.loads(result_rows[0])["item_id"] == "good"
    progress = json.loads((output / "progress.json").read_text())
    assert progress["status"] == "failed"
    assert progress["completed"] == 1
    assert progress["failed"] == 1


def test_quota_summary_is_secret_free_and_reports_aggregate_capacity(
    tmp_path: Path,
) -> None:
    status = {
        "schema": "gemini_native_quota_state_v3",
        "pacific_date": "2026-07-25",
        "rpd_reset_at_utc": "2026-07-26T07:00:00.000Z",
        "rpm_limit": 5,
        "tpm_limit": 250_000,
        "daily_request_limit": 20,
        "keys": {
            "a" * 64: {
                "rpm_requests_in_window": 1,
                "rpm_remaining": 4,
                "tpm_tokens_in_window": 100,
                "tpm_remaining": 249_900,
                "requests_started": 2,
                "rpd_remaining": 18,
                "first_request_at_utc": "x",
                "last_request_at_utc": "y",
                "rpm_ready_at_utc": "z",
                "blocked_until_utc": None,
                "exhausted_by_429": False,
            },
            "b" * 64: {
                "rpm_remaining": 5,
                "tpm_remaining": 250_000,
                "rpd_remaining": 20,
            },
        },
    }
    summary = summarize_quota(status, state_path=tmp_path / "state.json")
    assert summary["key_count"] == 2
    assert summary["rpd_remaining_total"] == 38
    assert summary["keys"][0]["fingerprint_prefix"] == "a" * 12
    assert len(json.dumps(summary["keys"][0])) < 1000
