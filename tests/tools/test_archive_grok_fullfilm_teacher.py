import json

import pytest

from tools.align import archive_grok_fullfilm_teacher as archive


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _make_run(tmp_path):
    source_video = tmp_path / "sample.mp4"
    source_video.write_bytes(b"source-video")
    run = tmp_path / "run"
    chunk = {
        "chunk_id": "sample-0000",
        "film_id": "sample",
        "chunk_index": 0,
        "source": str(source_video),
        "audio": str(run / "audio" / "sample-0000.mp3"),
        "nominal_start_s": 0.0,
        "nominal_end_s": 10.0,
        "request_start_s": 0.0,
        "request_end_s": 10.0,
        "request_duration_s": 10.0,
        "estimated_cost_usd": 0.001,
    }
    manifest = {
        "schema": "grok_fullfilm_stt_v1",
        "model": "x-ai/grok-stt-1.0",
        "chunk_s": 300.0,
        "overlap_s": 5.0,
        "price_per_hour_usd": 0.1,
        "timeline_filter": "aresample=16000:async=1000:first_pts=0",
        "films": [
            {
                "film_id": "sample",
                "source": str(source_video),
                "duration_s": 10.0,
                "chunk_count": 1,
            }
        ],
        "chunks": [chunk],
    }
    response = {
        "schema": "grok_fullfilm_stt_v1",
        "chunk": chunk,
        "model": "x-ai/grok-stt-1.0",
        "parsed": {"words": []},
        "response": {},
    }
    words = [
        {
            "film_id": "sample",
            "chunk_id": "sample-0000",
            "text": "日",
            "start_s": 1.0,
            "end_s": 1.2,
            "speaker": 0,
        }
    ]
    cuts = [
        {
            "film_id": "sample",
            "chunk_id": "sample-0000",
            "accepted": True,
            "cut_s": 5.0,
        }
    ]
    summary = {
        "model": "x-ai/grok-stt-1.0",
        "word_count": 1,
        "speaker_change_count": 1,
        "accepted_nonoverlap_speaker_cuts": 1,
        "provider_actual_cost_usd": 0.001,
        "films": {
            "sample": {"word_count": 1, "accepted_nonoverlap_speaker_cuts": 1}
        },
    }
    _write_json(run / "manifest.json", manifest)
    _write_json(run / "summary.json", summary)
    _write_json(run / "errors.json", [])
    _write_json(run / "responses" / "sample-0000.json", response)
    _write_jsonl(run / "grok.words.jsonl", words)
    _write_jsonl(run / "grok.speaker_cuts.jsonl", cuts)
    return run, source_video


def test_archive_preserves_paid_teacher_and_marks_diagnostic(tmp_path):
    run, source_video = _make_run(tmp_path)
    output = tmp_path / "archive"

    result = archive.archive_run(run, output, hash_sources=True)

    assert result["schema"] == archive.ARCHIVE_SCHEMA
    assert result["word_times"] == 1
    assert result["training_ready"] is False
    film = json.loads(
        (output / "compiled" / "films.jsonl").read_text(encoding="utf-8")
    )
    assert film["partition"] == "diagnostic"
    assert film["evaluation_eligible"] is False
    assert film["supervision_mode"] == "frame_only_candidate"
    assert film["source_sha256"] == archive._sha256(source_video)
    assert (output / "teacher" / "responses" / "sample-0000.json").is_file()
    assert "--prepare-only" in (
        output / "rebuild" / "rebuild_audio.ps1"
    ).read_text(encoding="utf-8")


def test_archive_rejects_response_manifest_mismatch(tmp_path):
    run, _source_video = _make_run(tmp_path)
    response = json.loads(
        (run / "responses" / "sample-0000.json").read_text(encoding="utf-8")
    )
    response["chunk"]["nominal_end_s"] = 9.0
    _write_json(run / "responses" / "sample-0000.json", response)

    with pytest.raises(ValueError, match="does not match manifest chunk"):
        archive.archive_run(run, tmp_path / "archive")


def test_archive_refuses_to_overwrite(tmp_path):
    run, _source_video = _make_run(tmp_path)
    output = tmp_path / "archive"
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        archive.archive_run(run, output)
