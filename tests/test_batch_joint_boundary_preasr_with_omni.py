from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.datasets import batch_joint_boundary_preasr_with_omni as batch_omni


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _dataset(tmp_path: Path, candidate_count: int = 3) -> Path:
    dataset = tmp_path / "dataset"
    audio = tmp_path / "window.wav"
    audio.write_bytes(b"source")
    candidates = dataset / "candidates.jsonl"
    _write_jsonl(
        candidates,
        [
            {
                "sample_id": f"sample-{index}",
                "candidate_id": f"candidate-{index}",
                "audio_id": "audio-1",
                "video_id": "video-1",
                "chunk_index": index,
                "start": float(index),
                "end": float(index + 1),
                "duration_s": 1.0,
            }
            for index in range(candidate_count)
        ],
    )
    _write_jsonl(
        dataset / "source_windows.jsonl",
        [
            {
                "window_id": "window-1",
                "audio_wav": str(audio),
                "omni_mp3_32k": str(audio),
                "pre_asr_candidates": str(candidates),
            }
        ],
    )
    return dataset


def test_batch_rejects_realtime_flash_model() -> None:
    with pytest.raises(ValueError, match="not supported"):
        batch_omni._resolve_batch_model("qwen3.5-omni-flash")


def test_prepare_shards_requests_and_disables_thinking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(tmp_path)
    output = tmp_path / "labels"
    batch_dir = tmp_path / "batch"
    manifest_path = batch_dir / "manifest.json"

    def fake_request_audio(item):
        path = Path(item["request_audio"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"wav")
        return path

    monkeypatch.setattr(batch_omni, "_ensure_request_audio", fake_request_audio)
    args = argparse.Namespace(
        env_file=str(tmp_path / "missing.env"),
        dataset_dir=str(dataset),
        output_dir=str(output),
        batch_dir=str(batch_dir),
        manifest=str(manifest_path),
        model="qwen3.5-omni-plus",
        max_runtime_chunks=0,
        audio_content_mode="input_audio",
        max_tokens=256,
        max_requests_per_file=2,
        max_file_bytes=batch_omni.MAX_BATCH_FILE_BYTES,
        max_line_bytes=batch_omni.MAX_BATCH_LINE_BYTES,
        only_custom_ids_from=[],
        include_existing_raw=False,
        include_completed_windows=False,
    )

    batch_omni.prepare_batch(args)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model"] == "qwen3.5-omni-plus"
    assert manifest["enable_thinking"] is False
    assert len(manifest["prepared_sha256"]) == 64
    assert manifest["exported_count"] == 3
    assert [shard["request_count"] for shard in manifest["shards"]] == [2, 1]
    lines = (batch_dir / "requests-00000.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    assert first["url"] == "/v1/chat/completions"
    assert first["body"]["enable_thinking"] is False
    assert first["body"]["model"] == "qwen3.5-omni-plus"


def test_submit_status_and_download_are_manifest_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "requests-00000.jsonl"
    input_path.write_text('{"custom_id":"c1"}\n', encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    batch_omni._save_manifest(
        manifest_path,
        {
            "schema": batch_omni.BATCH_MANIFEST_SCHEMA,
            "batch_dir": str(tmp_path),
            "prompt_version": "prompt-v1",
            "shards": [
                {
                    "index": 0,
                    "input_path": str(input_path),
                    "sha256": batch_omni._sha256(input_path),
                    "status": "prepared",
                    "input_file_id": "",
                    "batch_id": "",
                    "output_file_id": "",
                    "error_file_id": "",
                    "output_path": "",
                    "error_path": "",
                }
            ],
        },
    )

    class FakeFiles:
        def create(self, **kwargs):
            assert kwargs["purpose"] == "batch"
            return SimpleNamespace(id="file-input")

        def content(self, file_id):
            payload = b'{"custom_id":"c1","response":{"status_code":200}}\n'
            return SimpleNamespace(write_to_file=lambda path: Path(path).write_bytes(payload))

    class FakeBatches:
        def create(self, **kwargs):
            assert kwargs["endpoint"] == "/v1/chat/completions"
            return {"id": "batch-1", "status": "validating"}

        def retrieve(self, batch_id):
            assert batch_id == "batch-1"
            return {
                "id": batch_id,
                "status": "completed",
                "output_file_id": "file-output",
                "error_file_id": None,
                "request_counts": {"total": 1, "completed": 1, "failed": 0},
            }

    client = SimpleNamespace(files=FakeFiles(), batches=FakeBatches())
    monkeypatch.setattr(batch_omni, "_provider_client", lambda args: client)
    args = argparse.Namespace(
        manifest=str(manifest_path),
        env_file="",
        base_url="",
        timeout_s=1.0,
        task_name="smoke",
        completion_window="24h",
        refresh=True,
    )

    batch_omni.submit_batch(args)
    batch_omni.download_batch(args)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard = manifest["shards"][0]
    assert shard["batch_id"] == "batch-1"
    assert shard["status"] == "completed"
    assert Path(shard["output_path"]).read_bytes().startswith(b'{"custom_id":"c1"')


def test_submit_recovers_uncertain_create_without_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "requests-00000.jsonl"
    input_path.write_text('{"custom_id":"c1"}\n', encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "schema": batch_omni.BATCH_MANIFEST_SCHEMA,
        "batch_dir": str(tmp_path),
        "prompt_version": "prompt-v1",
        "model": "qwen3.5-omni-plus",
        "shards": [
            {
                "index": 0,
                "input_path": str(input_path),
                "sha256": batch_omni._sha256(input_path),
                "request_count": 1,
                "status": "submitting",
                "input_file_id": "file-input",
                "batch_id": "",
                "create_attempted": True,
            }
        ],
    }
    manifest["prepared_sha256"] = batch_omni._prepared_manifest_sha256(manifest)
    submission_key = batch_omni._submission_key(manifest, manifest["shards"][0])
    manifest["shards"][0]["submission_key"] = submission_key
    batch_omni._save_manifest(manifest_path, manifest)

    class FakeFiles:
        def create(self, **_kwargs):
            raise AssertionError("resume must not upload again")

    class FakeBatches:
        def list(self, **_kwargs):
            return {
                "data": [
                    {
                        "id": "batch-existing",
                        "status": "validating",
                        "input_file_id": "file-input",
                        "metadata": {"submission_key": submission_key},
                    }
                ]
            }

        def create(self, **_kwargs):
            raise AssertionError("resume must not create a duplicate batch")

    monkeypatch.setattr(
        batch_omni,
        "_provider_client",
        lambda _args: SimpleNamespace(files=FakeFiles(), batches=FakeBatches()),
    )
    args = argparse.Namespace(
        manifest=str(manifest_path),
        env_file="",
        base_url="",
        timeout_s=1.0,
        task_name="smoke",
        completion_window="24h",
    )

    batch_omni.submit_batch(args)

    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert stored["shards"][0]["batch_id"] == "batch-existing"


def test_submit_persists_upload_and_create_intent_before_remote_create(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "requests-00000.jsonl"
    input_path.write_text('{"custom_id":"c1"}\n', encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    batch_omni._save_manifest(
        manifest_path,
        {
            "schema": batch_omni.BATCH_MANIFEST_SCHEMA,
            "prompt_version": "prompt-v1",
            "model": "qwen3.5-omni-plus",
            "shards": [
                {
                    "index": 0,
                    "input_path": str(input_path),
                    "sha256": batch_omni._sha256(input_path),
                    "request_count": 1,
                    "input_file_id": "",
                    "batch_id": "",
                }
            ],
        },
    )

    class FakeFiles:
        def create(self, **_kwargs):
            return SimpleNamespace(id="file-persisted")

    class FakeBatches:
        def create(self, **_kwargs):
            raise TimeoutError("provider outcome unknown")

    monkeypatch.setattr(
        batch_omni,
        "_provider_client",
        lambda _args: SimpleNamespace(files=FakeFiles(), batches=FakeBatches()),
    )
    args = argparse.Namespace(
        manifest=str(manifest_path),
        env_file="",
        base_url="",
        timeout_s=1.0,
        task_name="smoke",
        completion_window="24h",
    )

    with pytest.raises(TimeoutError, match="outcome unknown"):
        batch_omni.submit_batch(args)

    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard = stored["shards"][0]
    assert shard["input_file_id"] == "file-persisted"
    assert shard["create_attempted"] is True
    assert len(shard["submission_key"]) == 64


def test_submit_stops_when_uncertain_create_cannot_be_reconciled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "requests-00000.jsonl"
    input_path.write_text('{"custom_id":"c1"}\n', encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "schema": batch_omni.BATCH_MANIFEST_SCHEMA,
        "prompt_version": "prompt-v1",
        "model": "qwen3.5-omni-plus",
        "shards": [
            {
                "index": 0,
                "input_path": str(input_path),
                "sha256": batch_omni._sha256(input_path),
                "request_count": 1,
                "input_file_id": "file-input",
                "batch_id": "",
                "create_attempted": True,
            }
        ],
    }
    batch_omni._save_manifest(manifest_path, manifest)

    class FakeBatches:
        def list(self, **_kwargs):
            return {"data": []}

        def create(self, **_kwargs):
            raise AssertionError("uncertain resume must not create again")

    monkeypatch.setattr(
        batch_omni,
        "_provider_client",
        lambda _args: SimpleNamespace(files=SimpleNamespace(), batches=FakeBatches()),
    )
    args = argparse.Namespace(
        manifest=str(manifest_path),
        env_file="",
        base_url="",
        timeout_s=1.0,
        task_name="smoke",
        completion_window="24h",
    )

    with pytest.raises(RuntimeError, match="refusing duplicate submission"):
        batch_omni.submit_batch(args)
