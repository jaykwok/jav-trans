from __future__ import annotations

import json
from pathlib import Path

from tools.asr.cueqc import label_pre_asr_with_omni as omni_label


def test_default_omni_env_file_is_provider_neutral() -> None:
    assert omni_label.DEFAULT_ENV_FILE == "~/.config/omni/.env"
    assert "qwen" not in omni_label.DEFAULT_ENV_FILE.lower()


def test_training_label_from_omni_maps_to_existing_v10_labels():
    assert (
        omni_label.training_label_from_omni(
            label="drop",
            confidence=0.91,
            keep_confidence=0.80,
            drop_confidence=0.90,
        )
        == "definite_drop"
    )
    assert (
        omni_label.training_label_from_omni(
            label="keep",
            confidence=0.80,
            keep_confidence=0.80,
            drop_confidence=0.90,
        )
        == "definite_keep"
    )
    assert (
        omni_label.training_label_from_omni(
            label="drop",
            confidence=0.89,
            keep_confidence=0.80,
            drop_confidence=0.90,
        )
        == "ambiguous_ignore"
    )
    assert (
        omni_label.training_label_from_omni(
            label="unsure",
            confidence=1.0,
            keep_confidence=0.80,
            drop_confidence=0.90,
        )
        == "ambiguous_ignore"
    )


def test_label_row_from_response_exports_compile_ready_fields(tmp_path: Path):
    audio = tmp_path / "chunk.mp3"
    audio.write_bytes(b"mp3")
    row = omni_label.label_row_from_response(
        candidate={
            "sample_id": "preasr-AAA-chunk00001",
            "candidate_id": "preasr-AAA-chunk00001",
            "video_id": "AAA",
            "audio_id": "AAA",
            "chunk_index": 1,
            "start": 1.0,
            "end": 1.4,
            "duration_s": 0.4,
            "feature_schema": "pre_asr_cueqc_features_v6",
            "runtime_adapter": "pre_asr_planned_island_sequence_v2",
        },
        audio_path=audio,
        response={
            "label": "drop",
            "confidence": 0.95,
            "semantic_speech_detected": False,
            "flags": ["moan"],
            "reason": "没有语义语音",
        },
        model="qwen3.5-omni-flash",
        fmt="mp3",
        keep_confidence=0.80,
        drop_confidence=0.90,
    )

    assert row["label"] == "definite_drop"
    assert row["display_decision"] == "drop"
    assert row["training_label_included"] is True
    assert row["label_source"] == "omni:qwen3.5-omni-flash"
    assert row["omni_label"] == "drop"
    assert row["omni_flags"] == ["moan"]


def test_training_label_counts_reports_drop_keep_ratio(tmp_path: Path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "\n".join(
            json.dumps({"label": label})
            for label in [
                "definite_drop",
                "definite_drop",
                "definite_keep",
                "ambiguous_ignore",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = omni_label.training_label_counts(labels)

    assert summary["definite_drop_count"] == 2
    assert summary["definite_keep_count"] == 1
    assert summary["ambiguous_ignore_count"] == 1
    assert summary["drop_keep_ratio"] == 2.0


def test_load_env_file_sets_missing_env_only(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OMNI_BASE_URL=https://example.invalid/v1",
                "OMNI_MODEL='qwen3.5-omni-flash'",
                "OMNI_API_KEY=file-key",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OMNI_API_KEY", "process-key")
    monkeypatch.delenv("OMNI_BASE_URL", raising=False)
    monkeypatch.delenv("OMNI_MODEL", raising=False)

    loaded = omni_label.load_env_file(env_file)
    key_name, key_value = omni_label.first_env_value("OMNI_API_KEY")

    assert loaded["OMNI_BASE_URL"] == "https://example.invalid/v1"
    assert loaded["OMNI_MODEL"] == "qwen3.5-omni-flash"
    assert key_name == "OMNI_API_KEY"
    assert key_value == "process-key"
    assert omni_label.first_env_value("OMNI_BASE_URL")[1] == "https://example.invalid/v1"


def test_ogg_audio_format_is_cli_and_payload_supported(tmp_path: Path):
    args = omni_label.parse_args(
        ["--workflow-root", "agents/temp/run", "--audio-format", "ogg", "--skip-items", "10"]
    )
    audio = tmp_path / "chunk.ogg"
    audio.write_bytes(b"ogg")

    uri = omni_label.data_uri_for_audio(audio, "ogg")
    part = omni_label.audio_content_part(audio, fmt="ogg", mode="input_audio")
    raw_part = omni_label.audio_content_part(audio, fmt="ogg", mode="input_audio_raw")
    video_part = omni_label.audio_content_part(audio, fmt="ogg", mode="video_url")

    assert args.audio_format == "ogg"
    assert args.skip_items == 10
    assert uri.startswith("data:audio/ogg;base64,")
    assert part["input_audio"]["format"] == "ogg"
    assert part["input_audio"]["data"].startswith("data:;base64,")
    assert not raw_part["input_audio"]["data"].startswith("data:")
    assert video_part["type"] == "video_url"
    assert video_part["video_url"]["url"].startswith("data:audio/ogg;base64,")


def test_empty_audio_api_error_detection():
    assert omni_label.is_empty_audio_api_error(RuntimeError("The audio is empty"))
    assert not omni_label.is_empty_audio_api_error(RuntimeError("connection reset"))
