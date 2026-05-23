from __future__ import annotations

import numpy as np

from vad.fusionvad_lite import (
    TeacherSegment,
    audit_audio,
    build_supervised_record,
    build_teacher_record,
    read_jsonl,
    segments_to_frame_labels,
    write_jsonl,
)


def test_pseudo_label_discovers_supported_inputs(tmp_path):
    import importlib.util

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_lite"
        / "pseudo_label.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_lite_pseudo_label", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    (tmp_path / "a.wav").write_bytes(b"")
    (tmp_path / "b.mp4").write_bytes(b"")
    (tmp_path / "ignore.txt").write_text("", encoding="utf-8")

    paths = module.discover_inputs([str(tmp_path)])

    assert [path.name for path in paths] == ["a.wav", "b.mp4"]


def test_pseudo_label_sample_audio_reads_hf_ogg_shape():
    import importlib.util

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_lite"
        / "pseudo_label.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_lite_pseudo_label_sample", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    audio, sample_rate, text, audio_id = module.sample_audio(
        {
            "__key__": "abc",
            "ogg": {"array": np.array([0.0, 0.1], dtype=np.float32), "sampling_rate": 16000},
            "txt": "テスト",
        }
    )

    assert np.allclose(audio, [0.0, 0.1])
    assert sample_rate == 16000
    assert text == "テスト"
    assert audio_id == "abc"


def test_pseudo_label_requires_input_or_hf_dataset():
    import importlib.util

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_lite"
        / "pseudo_label.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_lite_pseudo_label_args", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        module.parse_args([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("parse_args should require --input or --hf-dataset")

    args = module.parse_args(["--hf-dataset", "litagin/Galgame_Speech_ASR_16kHz", "--hf-limit", "1"])
    assert args.hf_dataset == "litagin/Galgame_Speech_ASR_16kHz"
    assert args.input is None


def test_segments_to_frame_labels_clamps_and_marks_overlap():
    labels = segments_to_frame_labels(
        [
            TeacherSegment(-1.0, 0.1),
            {"start": 0.2, "end": 0.6},
            (0.9, 2.0),
            (0.7, 0.7),
        ],
        duration_s=1.0,
        frame_hop_s=0.25,
    )

    assert labels == [1, 1, 1, 1]


def test_build_teacher_record_classifies_agreement():
    record = build_teacher_record(
        audio_id="clip",
        source="unit",
        duration_s=1.0,
        teacher_segments={
            "a": [(0.0, 0.5)],
            "b": [(0.25, 0.75)],
            "c": [],
        },
        frame_hop_s=0.25,
        min_speech_teachers=2,
    )

    assert record.label_quality == "teacher_agree"
    assert record.speech_frames == [0, 1, 0, 0]
    assert record.teacher_segments["a"] == [TeacherSegment(0.0, 0.5)]


def test_build_teacher_record_classifies_negative_and_conflict():
    negative = build_teacher_record(
        audio_id="silent",
        source="unit",
        duration_s=0.5,
        teacher_segments={"a": [], "b": []},
        frame_hop_s=0.25,
    )
    conflict = build_teacher_record(
        audio_id="single",
        source="unit",
        duration_s=0.5,
        teacher_segments={"a": [(0.0, 0.5)], "b": []},
        frame_hop_s=0.25,
    )

    assert negative.label_quality == "negative"
    assert negative.speech_frames == [0, 0]
    assert conflict.label_quality == "teacher_conflict"
    assert conflict.speech_frames == [0, 0]


def test_build_supervised_record_uses_supervised_quality():
    record = build_supervised_record(
        audio_id="gold",
        source="ava",
        duration_s=0.5,
        speech_segments=[(0.0, 0.25)],
        frame_hop_s=0.25,
    )

    assert record.label_quality == "supervised"
    assert record.speech_frames == [1, 0]
    assert list(record.teacher_segments) == ["supervised"]


def test_label_jsonl_round_trip(tmp_path):
    path = tmp_path / "labels.jsonl"
    record = build_teacher_record(
        audio_id="clip",
        source="unit",
        duration_s=0.5,
        text="こんにちは",
        teacher_segments={"a": [(0.0, 0.5)], "b": [(0.0, 0.5)]},
        frame_hop_s=0.25,
    )

    write_jsonl(path, [record])
    rows = read_jsonl(path)

    assert rows == [record]


def test_audit_audio_reports_energy_and_text_stats():
    audio = np.ones(16000, dtype=np.float32) * 0.5

    audit = audit_audio(
        audio_id="tone",
        source="unit",
        audio=audio,
        sample_rate=16000,
        text="abc",
    )

    assert audit.duration_s == 1.0
    assert audit.sample_rate == 16000
    assert audit.text_chars == 3
    assert -7.0 < audit.rms_dbfs < -5.0
