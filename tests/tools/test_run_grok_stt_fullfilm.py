import json

import pytest

from tools.omni import run_grok_stt_fullfilm as fullfilm


def test_parse_video_spec_supports_named_and_plain_paths():
    named = fullfilm.parse_video_spec("sample-a=D:/media/example.mp4")
    plain = fullfilm.parse_video_spec("D:/media/another.mp4")

    assert named == {"film_id": "sample-a", "source": "D:\\media\\example.mp4"}
    assert plain == {"film_id": "another", "source": "D:\\media\\another.mp4"}


def test_build_manifest_applies_overlap_and_cost_gate(monkeypatch, tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"media")
    monkeypatch.setattr(fullfilm, "probe_duration_s", lambda _path: 610.0)

    manifest = fullfilm.build_manifest(
        [{"film_id": "sample", "source": str(video)}],
        output_dir=tmp_path / "out",
        model=fullfilm.DEFAULT_MODEL,
        chunk_s=300.0,
        overlap_s=5.0,
        price_per_hour_usd=0.10,
        max_cost_usd=1.0,
    )

    assert len(manifest["chunks"]) == 3
    assert manifest["chunks"][1]["request_start_s"] == 295.0
    assert manifest["chunks"][1]["request_end_s"] == 605.0
    assert manifest["timeline_filter"] == fullfilm.TIMELINE_FILTER
    assert manifest["estimated_cost_usd"] == pytest.approx(630.0 / 3600.0 * 0.1)

    with pytest.raises(RuntimeError, match="preflight refused"):
        fullfilm.build_manifest(
            [{"film_id": "sample", "source": str(video)}],
            output_dir=tmp_path / "other",
            model=fullfilm.DEFAULT_MODEL,
            chunk_s=300.0,
            overlap_s=5.0,
            price_per_hour_usd=0.10,
            max_cost_usd=0.001,
        )


def test_chunk_command_preserves_pts_gaps(tmp_path):
    command = fullfilm.build_chunk_command(
        {
            "request_start_s": 295.0,
            "request_duration_s": 310.0,
            "source": str(tmp_path / "source.mp4"),
        },
        tmp_path / "chunk.mp3",
    )

    assert command[command.index("-af") + 1] == (
        "aresample=16000:async=1000:first_pts=0"
    )


def test_normalized_words_deduplicates_overlap_by_nominal_window():
    record = {
        "chunk": {
            "film_id": "sample",
            "chunk_id": "sample-0001",
            "chunk_index": 1,
            "request_start_s": 295.0,
            "nominal_start_s": 300.0,
            "nominal_end_s": 600.0,
        },
        "parsed": {
            "words": [
                {"text": "前", "start_s": 1.0, "end_s": 2.0, "speaker": 0},
                {"text": "中", "start_s": 10.0, "end_s": 11.0, "speaker": 0},
                {"text": "後", "start_s": 306.0, "end_s": 307.0, "speaker": 1},
            ]
        },
    }

    words = fullfilm.normalized_words(record)

    assert [word["text"] for word in words] == ["中"]
    assert words[0]["start_s"] == 305.0


def test_speaker_change_rows_only_accepts_non_overlapping_changes():
    words = [
        {
            "film_id": "sample",
            "chunk_id": "sample-0000",
            "speaker": 0,
            "text": "a",
            "start_s": 0.0,
            "end_s": 1.0,
        },
        {
            "film_id": "sample",
            "chunk_id": "sample-0000",
            "speaker": 1,
            "text": "b",
            "start_s": 1.4,
            "end_s": 2.0,
        },
        {
            "film_id": "sample",
            "chunk_id": "sample-0000",
            "speaker": 0,
            "text": "c",
            "start_s": 1.8,
            "end_s": 2.4,
        },
    ]

    rows = fullfilm.speaker_change_rows(words)

    assert rows[0]["accepted"] is True
    assert rows[0]["cut_s"] == pytest.approx(1.2)
    assert rows[1]["accepted"] is False
    assert rows[1]["cut_s"] is None


def test_manifest_resume_rejects_changed_geometry(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = {"schema": fullfilm.SCHEMA, "chunks": [{"chunk_id": "a"}]}
    fullfilm.write_or_validate_manifest(path, manifest)
    fullfilm.write_or_validate_manifest(path, manifest)

    assert json.loads(path.read_text(encoding="utf-8")) == manifest
    with pytest.raises(RuntimeError, match="existing manifest differs"):
        fullfilm.write_or_validate_manifest(
            path,
            {"schema": fullfilm.SCHEMA, "chunks": [{"chunk_id": "b"}]},
        )
