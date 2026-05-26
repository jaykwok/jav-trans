from __future__ import annotations

import numpy as np

from vad.base import SpeechSegment
from vad.fusionvad_ja import (
    AdditionFusionBiLSTM,
    DEFAULT_TRAINABLE_LABEL_QUALITIES,
    FeatureConfig,
    FeatureTrainConfig,
    TeacherSegment,
    TimestampSpanVadBackend,
    TrainConfig,
    align_feature_frames,
    audit_audio,
    build_negative_record,
    build_supervised_record,
    build_teacher_record,
    build_weighted_teacher_record,
    build_weak_positive_record,
    build_training_examples,
    build_training_windows,
    count_trainable_parameters,
    default_trainable_records,
    dry_run_batches,
    effective_frame_weights,
    evaluate_addition_fusion_classifier,
    frame_classification_counts,
    frame_count,
    get_research_vad_backend,
    is_default_trainable,
    is_low_frame_rate_ptm,
    is_qwen3_asr_ptm,
    load_manifest_audio_map,
    metrics_from_frame_counts,
    normalize_audio_16k_mono,
    qwen3_asr_audio_output_lengths,
    qwen3_asr_repo_id,
    read_jsonl,
    resize_feature_frames,
    sample_hf_audio_16k_mono,
    segments_to_frame_labels,
    shuffled_window_order,
    stable_hf_audio_id,
    train_addition_fusion_classifier,
    with_frame_weights,
    write_jsonl,
    train_tiny_frame_classifier,
    write_training_manifest,
)


def test_pseudo_label_discovers_supported_inputs(tmp_path):
    import importlib.util

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "pseudo_label.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_pseudo_label", script_path)
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
        / "fusionvad_ja"
        / "pseudo_label.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_pseudo_label_sample", script_path)
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


def test_hf_audio_sample_normalizes_to_16k_mono():
    stereo_8k = np.column_stack(
        [
            np.ones(8000, dtype=np.float32) * 0.2,
            np.ones(8000, dtype=np.float32) * 0.4,
        ]
    )

    audio, sample_rate = sample_hf_audio_16k_mono(
        {
            "ogg": {"array": stereo_8k, "sampling_rate": 8000},
        }
    )

    assert sample_rate == 16000
    assert audio.ndim == 1
    assert len(audio) == 16000
    assert np.isclose(float(np.mean(audio[100:-100])), 0.3, atol=1e-3)


def test_hf_audio_sample_decodes_audio_bytes_mapping():
    import io

    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(8000, dtype=np.float32), 8000, format="WAV")

    audio, sample_rate = sample_hf_audio_16k_mono(
        {
            "audio": {"bytes": buffer.getvalue(), "path": "sample.wav"},
        }
    )

    assert sample_rate == 16000
    assert audio.shape == (16000,)


def test_hf_audio_sample_decodes_torchcodec_like_audio():
    class Samples:
        data = np.vstack(
            [
                np.ones(8000, dtype=np.float32) * 0.2,
                np.ones(8000, dtype=np.float32) * 0.4,
            ]
        )
        sample_rate = 8000

    class Decoder:
        def get_all_samples(self):
            return Samples()

    audio, sample_rate = sample_hf_audio_16k_mono({"audio": Decoder()})

    assert sample_rate == 16000
    assert audio.shape == (16000,)
    assert np.isclose(float(np.mean(audio[100:-100])), 0.3, atol=1e-3)


def test_stable_hf_audio_id_sanitizes_dataset_and_split():
    assert stable_hf_audio_id(dataset_name="diarizers-community/voxconverse", split="dev", index=7) == (
        "diarizers-community_voxconverse-dev-000007"
    )


def test_pseudo_label_requires_input_or_hf_dataset():
    import importlib.util

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "pseudo_label.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_pseudo_label_args", script_path)
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


def test_seed_label_cli_builds_supervised_records_from_timestamp_datasets():
    import importlib.util

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "build_seed_labels.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_seed_labels", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ava = module.record_from_ava_example(
        {
            "onset": [0.0, 1.0, 2.0],
            "offset": [0.5, 1.5, 2.5],
            "cluster": ["human_SPEECH", "music", "human_SPEECH"],
        },
        index=3,
        dataset_name="ava-dataset",
        split="train",
        source="ava",
        frame_hop_s=0.5,
    )
    vox = module.record_from_voxconverse_example(
        {
            "audio": {"path": "clip.wav"},
            "timestamps_start": [0.25, 0.75],
            "timestamps_end": [0.5, 1.25],
            "speakers": ["a", "b"],
        },
        index=4,
        dataset_name="vox-dataset",
        split="dev",
        source="vox",
        frame_hop_s=0.25,
    )

    assert ava is not None
    assert ava.label_quality == "supervised"
    assert list(ava.teacher_segments) == ["supervised"]
    assert ava.speech_frames == [1, 0, 0, 0, 1]
    assert vox is not None
    assert vox.audio_id == "vox-dataset-dev-000004"
    assert vox.speech_frames == [0, 1, 0, 1, 1]


def test_research_vad_backend_exposes_extra_teachers_without_public_registration():
    import pytest
    import vad

    with pytest.raises(ValueError, match="silero"):
        vad.get_vad_backend("silero")

    assert get_research_vad_backend("fusion_lite").name == "fusion_lite_v1"
    assert get_research_vad_backend("whisperseg-adaptive").name == "whisperseg_v1"
    assert get_research_vad_backend("silero").name == "silero_vad"
    ten = get_research_vad_backend("ten_vad")
    assert isinstance(ten, TimestampSpanVadBackend)
    assert ten.signature()["backend"] == "ten_vad"


def test_seed_label_cli_requires_at_least_one_source():
    import importlib.util

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "build_seed_labels.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_seed_labels_args", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        module.parse_args([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("parse_args should require at least one source")

    args = module.parse_args(["--ava-limit", "1"])
    assert args.ava_limit == 1
    assert args.ava_start_index == 0
    assert args.voxconverse_start_index == 0

    args = module.parse_args(["--voxconverse-start-index", "3", "--voxconverse-limit", "1"])
    assert args.voxconverse_start_index == 3

    args = module.parse_args(
        [
            "--negative-input",
            "noise",
            "--negative-start-index",
            "4",
            "--negative-limit",
            "5",
        ]
    )
    assert args.negative_start_index == 4
    assert args.negative_limit == 5


def test_slice_labeled_audio_cli_adjusts_segments():
    import importlib.util

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "slice_labeled_audio.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_slice_labeled_audio", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    record = build_supervised_record(
        audio_id="clip",
        source="unit",
        duration_s=4.0,
        speech_segments=[(0.5, 1.5), (2.0, 3.0)],
        frame_hop_s=0.5,
    )

    adjusted = module.adjusted_segments(record, start_s=1.0, duration_s=1.5)

    assert adjusted["supervised"] == [
        TeacherSegment(start=0.0, end=0.5, score=None),
        TeacherSegment(start=1.0, end=1.5, score=None),
    ]

    args = module.parse_args(["--labels", "labels.jsonl", "--clip-s", "2", "--stride-s", "1"])
    assert args.clip_s == 2
    assert args.stride_s == 1


def test_combine_manifest_cli_loads_manifest_rows(tmp_path):
    import importlib.util
    import json

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "combine_manifests.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_combine_manifests", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps([{"audio_id": "a"}, "ignored"]), encoding="utf-8")

    rows = module.load_manifest(manifest_path)
    args = module.parse_args(["--labels", "a.jsonl", "--manifest", str(manifest_path)])

    assert rows == [{"audio_id": "a"}]
    assert args.labels == ["a.jsonl"]


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


def test_frame_count_tolerates_float_boundary_noise():
    assert frame_count(18.76, 0.02) == 938
    assert frame_count(18.760000000000002, 0.02) == 938
    assert frame_count(18.761, 0.02) == 939


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


def test_build_teacher_record_accepts_project_speech_segments():
    record = build_teacher_record(
        audio_id="clip",
        source="unit",
        duration_s=0.5,
        teacher_segments={
            "a": [SpeechSegment(0.0, 0.25, score=0.7)],
            "b": [SpeechSegment(0.0, 0.25)],
        },
        frame_hop_s=0.25,
        min_speech_teachers=2,
    )

    assert record.label_quality == "teacher_agree"
    assert record.speech_frames == [1, 0]
    assert record.teacher_segments["a"] == [TeacherSegment(0.0, 0.25, score=0.7)]


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


def test_build_negative_record_and_default_trainable_filter():
    negative = build_negative_record(
        audio_id="noise",
        source="musan",
        duration_s=0.5,
        frame_hop_s=0.25,
    )
    conflict = build_teacher_record(
        audio_id="conflict",
        source="unit",
        duration_s=0.5,
        teacher_segments={"a": [(0.0, 0.5)], "b": []},
        frame_hop_s=0.25,
    )

    assert DEFAULT_TRAINABLE_LABEL_QUALITIES == frozenset({"supervised", "teacher_agree", "negative"})
    assert negative.label_quality == "negative"
    assert negative.teacher_segments == {"negative": []}
    assert negative.speech_frames == [0, 0]
    assert is_default_trainable(negative) is True
    assert is_default_trainable(conflict) is False
    assert default_trainable_records([negative, conflict]) == [negative]


def test_build_weak_positive_record_marks_trimmed_clip_as_teacher_agree():
    record = build_weak_positive_record(
        audio_id="galgame",
        source="litagin/Galgame_Speech_ASR_16kHz",
        duration_s=1.0,
        text="こんにちは",
        frame_hop_s=0.25,
        trim_head_s=0.25,
        trim_tail_s=0.25,
    )

    assert record.label_quality == "teacher_agree"
    assert record.speech_frames == [0, 1, 1, 0]
    assert record.teacher_segments["weak_positive"] == [TeacherSegment(0.25, 0.75, score=1.0)]
    assert is_default_trainable(record) is True


def test_build_weighted_teacher_record_ignores_conflicts_and_boundaries():
    record = build_weighted_teacher_record(
        audio_id="clip",
        source="unit",
        duration_s=2.0,
        teacher_segments={
            "whisperseg-adaptive": [(0.2, 0.8), (1.4, 1.8)],
            "fusion_lite": [(0.3, 0.9)],
        },
        frame_hop_s=0.1,
        min_speech_teachers=2,
        min_negative_gap_s=0.3,
        boundary_pad_s=0.1,
        positive_weight=0.5,
        negative_weight=0.25,
    )

    weights = effective_frame_weights(record)

    assert record.label_quality == "teacher_agree"
    assert len(record.speech_frames) == 20
    assert len(weights) == 20
    assert any(value == 1 for value, weight in zip(record.speech_frames, weights) if weight > 0)
    assert any(value == 0 for value, weight in zip(record.speech_frames, weights) if weight > 0)
    assert any(weight == 0 for weight in weights)
    assert weights[14] == 0.0


def test_build_weighted_teacher_record_keeps_text_clip_without_teacher_agreement_non_trainable():
    record = build_weighted_teacher_record(
        audio_id="quiet",
        source="unit",
        duration_s=0.5,
        text="こんにちは",
        teacher_segments={"whisperseg-adaptive": [], "fusion_lite": []},
        frame_hop_s=0.25,
    )

    assert record.label_quality == "teacher_conflict"
    assert record.speech_frames == [0, 0]
    assert effective_frame_weights(record) == [0.0, 0.0]
    assert is_default_trainable(record) is False


def test_training_manifest_resolves_audio_and_dry_runs_batches(tmp_path):
    import json

    import soundfile as sf

    audio_path = tmp_path / "clip.wav"
    sf.write(str(audio_path), np.ones(1600, dtype=np.float32) * 0.1, 16000)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "audio_id": "clip",
                    "audio": str(audio_path),
                }
            ]
        ),
        encoding="utf-8",
    )
    record = build_teacher_record(
        audio_id="clip",
        source="unit",
        duration_s=0.1,
        teacher_segments={"a": [(0.0, 0.1)], "b": [(0.0, 0.1)]},
        frame_hop_s=0.02,
    )

    audio_map = load_manifest_audio_map(manifest_path)
    examples, skipped = build_training_examples(
        [record],
        manifest_audio_map=audio_map,
    )
    batches = dry_run_batches([record], examples, window_s=0.2, max_batches=1)

    assert skipped == []
    assert len(examples) == 1
    assert examples[0].audio_path == str(audio_path)
    assert examples[0].frame_count == 5
    assert examples[0].speech_frame_count == 5
    assert len(batches) == 1
    assert batches[0].audio_shape == (3200,)
    assert batches[0].label_shape == (10,)
    assert batches[0].sample_rate == 16000
    assert batches[0].speech_ratio == 0.5


def test_training_manifest_label_index_stays_original_after_filter(tmp_path):
    import soundfile as sf

    audio_path = tmp_path / "agree.wav"
    sf.write(str(audio_path), np.ones(1600, dtype=np.float32) * 0.1, 16000)
    conflict = build_teacher_record(
        audio_id="conflict",
        source="unit",
        duration_s=0.1,
        teacher_segments={"a": [(0.0, 0.1)], "b": []},
        frame_hop_s=0.02,
    )
    agree = build_teacher_record(
        audio_id="agree",
        source="unit",
        duration_s=0.1,
        teacher_segments={"a": [(0.0, 0.1)], "b": [(0.0, 0.1)]},
        frame_hop_s=0.02,
    )

    examples, skipped = build_training_examples(
        [conflict, agree],
        manifest_audio_map={"agree": str(audio_path)},
    )

    assert skipped == []
    assert len(examples) == 1
    assert examples[0].audio_id == "agree"
    assert examples[0].label_index == 1


def test_tiny_training_writes_checkpoint_and_metrics(tmp_path):
    import soundfile as sf

    audio_path = tmp_path / "clip.wav"
    sf.write(str(audio_path), np.ones(3200, dtype=np.float32) * 0.1, 16000)
    record = build_teacher_record(
        audio_id="clip",
        source="unit",
        duration_s=0.2,
        teacher_segments={"a": [(0.0, 0.2)], "b": [(0.0, 0.2)]},
        frame_hop_s=0.02,
    )
    examples, skipped = build_training_examples(
        [record],
        manifest_audio_map={"clip": str(audio_path)},
    )

    windows = build_training_windows(records=[record], examples=examples, window_s=0.2)
    metrics = train_tiny_frame_classifier(
        records=[record],
        examples=examples,
        output_dir=tmp_path / "train",
        config=TrainConfig(window_s=0.2, max_steps=2),
    )

    assert skipped == []
    assert len(windows) == 1
    assert windows[0][0].shape == (3200,)
    assert windows[0][1].shape == (10,)
    assert metrics.steps == 2
    assert __import__("pathlib").Path(metrics.checkpoint).exists()
    assert __import__("pathlib").Path(metrics.metrics_path).exists()


def test_addition_fusion_model_stays_under_parameter_budget():
    model = AdditionFusionBiLSTM(
        whisper_dim=1280,
        mfcc_dim=40,
        fusion_dim=256,
        hidden_dim=192,
        layers=2,
    )

    assert count_trainable_parameters(model) < 2_000_000


def test_align_feature_frames_crops_to_shortest_length():
    whisper = np.ones((5, 8), dtype=np.float32)
    mfcc = np.ones((3, 4), dtype=np.float32)

    aligned_whisper, aligned_mfcc = align_feature_frames(whisper, mfcc)

    assert aligned_whisper.shape == (3, 8)
    assert aligned_mfcc.shape == (3, 4)


def test_qwen3_asr_ptm_helpers_and_low_rate_resize():
    assert is_qwen3_asr_ptm("qwen3-asr-0.6b")
    assert is_qwen3_asr_ptm("Qwen/Qwen3-ASR-1.7B")
    assert is_low_frame_rate_ptm("qwen3-asr-0.6b")
    assert qwen3_asr_repo_id("qwen3-asr-0.6b") == "Qwen/Qwen3-ASR-0.6B"
    assert qwen3_asr_audio_output_lengths(100) == 13

    ptm = np.asarray([[0.0], [10.0]], dtype=np.float32)
    resized = resize_feature_frames(ptm, 5)
    assert resized.shape == (5, 1)
    assert np.allclose(resized[:, 0], [0.0, 2.5, 5.0, 7.5, 10.0])

    aligned_ptm, aligned_mfcc = align_feature_frames(
        ptm,
        np.ones((5, 4), dtype=np.float32),
        resize_ptm=True,
    )
    assert aligned_ptm.shape == (5, 1)
    assert aligned_mfcc.shape == (5, 4)


def test_qwen3_asr_feature_head_can_stay_under_budget():
    model = AdditionFusionBiLSTM(
        whisper_dim=3584,
        mfcc_dim=40,
        fusion_dim=160,
        hidden_dim=160,
        layers=2,
    )

    assert count_trainable_parameters(model) < 2_000_000


def test_addition_fusion_training_uses_cached_features(tmp_path):
    feature_path = tmp_path / "feature.npz"
    np.savez_compressed(
        feature_path,
        whisper=np.ones((6, 8), dtype=np.float32),
        mfcc=np.ones((6, 4), dtype=np.float32),
    )
    record = build_teacher_record(
        audio_id="clip",
        source="unit",
        duration_s=0.12,
        teacher_segments={"a": [(0.0, 0.06)], "b": [(0.0, 0.06)]},
        frame_hop_s=0.02,
    )

    metrics = train_addition_fusion_classifier(
        records=[record],
        feature_manifest_rows=[
            {
                "audio_id": "clip",
                "feature_path": str(feature_path),
                "label_index": 0,
            }
        ],
        output_dir=tmp_path / "train",
        config=FeatureTrainConfig(
            max_steps=2,
            fusion_dim=8,
            hidden_dim=4,
            layers=1,
            max_trainable_parameters=10_000,
        ),
    )

    assert metrics.steps == 2
    assert __import__("pathlib").Path(metrics.checkpoint).exists()
    assert __import__("pathlib").Path(metrics.metrics_path).exists()


def test_addition_fusion_training_initializes_from_checkpoint(tmp_path):
    feature_path = tmp_path / "feature.npz"
    np.savez_compressed(
        feature_path,
        whisper=np.ones((6, 8), dtype=np.float32),
        mfcc=np.ones((6, 4), dtype=np.float32),
    )
    record = build_teacher_record(
        audio_id="clip",
        source="unit",
        duration_s=0.12,
        teacher_segments={"a": [(0.0, 0.06)], "b": [(0.0, 0.06)]},
        frame_hop_s=0.02,
    )
    feature_rows = [{"audio_id": "clip", "feature_path": str(feature_path), "label_index": 0}]
    base_metrics = train_addition_fusion_classifier(
        records=[record],
        feature_manifest_rows=feature_rows,
        output_dir=tmp_path / "base",
        config=FeatureTrainConfig(
            max_steps=1,
            fusion_dim=8,
            hidden_dim=4,
            layers=1,
            max_trainable_parameters=10_000,
        ),
    )

    fine_tune_metrics = train_addition_fusion_classifier(
        records=[record],
        feature_manifest_rows=feature_rows,
        output_dir=tmp_path / "fine-tune",
        config=FeatureTrainConfig(
            max_steps=1,
            fusion_dim=8,
            hidden_dim=4,
            layers=1,
            max_trainable_parameters=10_000,
            init_checkpoint=base_metrics.checkpoint,
        ),
    )

    checkpoint = __import__("torch").load(fine_tune_metrics.checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint["init_checkpoint"] == base_metrics.checkpoint


def test_addition_fusion_eval_reports_frame_metrics(tmp_path):
    feature_path = tmp_path / "feature.npz"
    np.savez_compressed(
        feature_path,
        whisper=np.ones((6, 8), dtype=np.float32),
        mfcc=np.ones((6, 4), dtype=np.float32),
    )
    record = build_teacher_record(
        audio_id="clip",
        source="unit",
        duration_s=0.12,
        teacher_segments={"a": [(0.0, 0.06)], "b": [(0.0, 0.06)]},
        frame_hop_s=0.02,
    )
    feature_rows = [
        {
            "audio_id": "clip",
            "feature_path": str(feature_path),
            "label_index": 0,
        }
    ]
    train_metrics = train_addition_fusion_classifier(
        records=[record],
        feature_manifest_rows=feature_rows,
        output_dir=tmp_path / "train",
        config=FeatureTrainConfig(
            max_steps=1,
            fusion_dim=8,
            hidden_dim=4,
            layers=1,
            max_trainable_parameters=10_000,
        ),
    )

    eval_metrics = evaluate_addition_fusion_classifier(
        records=[record],
        feature_manifest_rows=feature_rows,
        checkpoint_path=__import__("pathlib").Path(train_metrics.checkpoint),
        output_dir=tmp_path / "eval",
    )

    assert eval_metrics.windows == 1
    assert eval_metrics.frames == 6
    assert eval_metrics.threshold == 0.5
    assert 0.0 <= eval_metrics.frame_accuracy <= 1.0
    assert 0.0 <= eval_metrics.f1 <= 1.0
    assert __import__("pathlib").Path(eval_metrics.metrics_path).exists()

    strict_metrics = evaluate_addition_fusion_classifier(
        records=[record],
        feature_manifest_rows=feature_rows,
        checkpoint_path=__import__("pathlib").Path(train_metrics.checkpoint),
        output_dir=tmp_path / "eval-strict",
        threshold=0.9,
    )
    assert strict_metrics.threshold == 0.9


def test_train_addition_bilstm_cli_offsets_repeated_feature_manifests(tmp_path):
    import importlib.util
    import json

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "train_addition_bilstm.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_train_addition_bilstm", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    first_labels = tmp_path / "first.jsonl"
    second_labels = tmp_path / "second.jsonl"
    write_jsonl(
        first_labels,
        [
            build_teacher_record(
                audio_id="a",
                source="unit",
                duration_s=0.02,
                teacher_segments={"t": [(0.0, 0.02)]},
                frame_hop_s=0.02,
            )
        ],
    )
    write_jsonl(
        second_labels,
        [
            build_teacher_record(
                audio_id="b",
                source="unit",
                duration_s=0.02,
                teacher_segments={"t": [(0.0, 0.02)]},
                frame_hop_s=0.02,
            )
        ],
    )
    first_manifest = tmp_path / "first_features.json"
    second_manifest = tmp_path / "second_features.json"
    first_manifest.write_text(json.dumps([{"audio_id": "a", "feature_path": "a.npz", "label_index": 0}]))
    second_manifest.write_text(json.dumps([{"audio_id": "b", "feature_path": "b.npz", "label_index": 0}]))

    records, rows = module.load_training_inputs(
        labels_paths=[str(first_labels), str(second_labels)],
        feature_manifest_paths=[str(first_manifest), str(second_manifest)],
    )

    assert [record.audio_id for record in records] == ["a", "b"]
    assert [row["label_index"] for row in rows] == [0, 1]


def test_addition_fusion_eval_ignores_zero_weight_frames(tmp_path):
    feature_path = tmp_path / "feature.npz"
    np.savez_compressed(
        feature_path,
        whisper=np.ones((4, 8), dtype=np.float32),
        mfcc=np.ones((4, 4), dtype=np.float32),
    )
    record = with_frame_weights(
        build_teacher_record(
            audio_id="clip",
            source="unit",
            duration_s=0.08,
            teacher_segments={"a": [(0.0, 0.08)], "b": [(0.0, 0.08)]},
            frame_hop_s=0.02,
        ),
        [1.0, 0.0, 0.0, 1.0],
    )
    feature_rows = [{"audio_id": "clip", "feature_path": str(feature_path), "label_index": 0}]
    train_metrics = train_addition_fusion_classifier(
        records=[record],
        feature_manifest_rows=feature_rows,
        output_dir=tmp_path / "train-weighted",
        config=FeatureTrainConfig(
            max_steps=1,
            fusion_dim=8,
            hidden_dim=4,
            layers=1,
            max_trainable_parameters=10_000,
        ),
    )

    eval_metrics = evaluate_addition_fusion_classifier(
        records=[record],
        feature_manifest_rows=feature_rows,
        checkpoint_path=__import__("pathlib").Path(train_metrics.checkpoint),
        output_dir=tmp_path / "eval-weighted",
    )

    assert eval_metrics.frames == 2


def test_frame_metrics_helpers_report_counts():
    counts = frame_classification_counts(
        labels=[1, 1, 0, 0],
        predictions=[1, 0, 1, 0],
    )
    metrics = metrics_from_frame_counts(counts=counts, windows=1)

    assert counts == {
        "frames": 4,
        "correct": 2,
        "positives": 2,
        "predicted_positives": 2,
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
    }
    assert metrics.frame_accuracy == 0.5
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5


def test_frame_metrics_helpers_ignore_zero_weight_frames():
    counts = frame_classification_counts(
        labels=[1, 1, 0, 0],
        predictions=[1, 0, 1, 0],
        weights=[1.0, 0.0, 0.0, 1.0],
    )

    assert counts == {
        "frames": 2,
        "correct": 2,
        "positives": 1,
        "predicted_positives": 1,
        "true_positive": 1,
        "false_positive": 0,
        "false_negative": 0,
    }


def test_shuffled_window_order_is_seeded_and_complete():
    first = shuffled_window_order(8, seed=13)
    second = shuffled_window_order(8, seed=13)
    other = shuffled_window_order(8, seed=14)

    assert first == second
    assert sorted(first) == list(range(8))
    assert first != other
    assert shuffled_window_order(0, seed=13) == []


def test_training_manifest_skips_missing_audio_and_frame_mismatch(tmp_path):
    good = build_negative_record(
        audio_id="missing",
        source="unit",
        duration_s=0.1,
        frame_hop_s=0.02,
    )
    bad = build_negative_record(
        audio_id="bad",
        source="unit",
        duration_s=0.1,
        frame_hop_s=0.02,
    )
    broken = type(bad)(
        audio_id=bad.audio_id,
        source=bad.source,
        duration_s=bad.duration_s,
        text=bad.text,
        teacher_segments=bad.teacher_segments,
        frame_hop_s=bad.frame_hop_s,
        speech_frames=[0],
        label_quality=bad.label_quality,
    )
    bad_weights = with_frame_weights(
        build_negative_record(
            audio_id="bad-weights",
            source="unit",
            duration_s=0.1,
            frame_hop_s=0.02,
        ),
        [1.0],
    )

    examples, skipped = build_training_examples(
        [good, broken, bad_weights],
        manifest_audio_map={},
        audio_root=tmp_path,
    )

    assert examples == []
    assert [row["reason"] for row in skipped] == [
        "missing_audio_path",
        "frame_count_mismatch",
        "frame_weight_count_mismatch",
    ]


def test_prepare_training_manifest_cli_requires_labels():
    import importlib.util

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "prepare_training_manifest.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_prepare_manifest_args", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        module.parse_args([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("parse_args should require --labels")

    args = module.parse_args(["--labels", "labels.jsonl", "--dry-run-batches", "0"])
    assert args.labels == "labels.jsonl"
    assert args.dry_run_batches == 0


def test_train_tiny_cli_requires_labels():
    import importlib.util

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "train_tiny.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_train_tiny_args", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        module.parse_args([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("parse_args should require --labels")

    args = module.parse_args(["--labels", "labels.jsonl", "--max-steps", "1"])
    assert args.labels == "labels.jsonl"
    assert args.max_steps == 1


def test_build_feature_cache_cli_requires_labels():
    import importlib.util

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "build_feature_cache.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_build_feature_cache_args", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        module.parse_args([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("parse_args should require --labels")

    args = module.parse_args(["--labels", "labels.jsonl", "--ptm", "whisper-ja-1.5b"])
    assert args.labels == "labels.jsonl"
    assert args.ptm == "whisper-ja-1.5b"

    args = module.parse_args(
        [
            "--labels",
            "labels.jsonl",
            "--ptm",
            "qwen3-asr-0.6b",
            "--model-path",
            "models/Qwen-Qwen3-ASR-0.6B",
            "--no-download",
            "--dtype",
            "bfloat16",
        ]
    )
    assert args.ptm == "qwen3-asr-0.6b"
    assert args.model_path == "models/Qwen-Qwen3-ASR-0.6B"
    assert args.no_download is True
    assert args.dtype == "bfloat16"


def test_build_feature_cache_run_supports_qwen_low_rate_features(tmp_path, monkeypatch):
    import importlib.util
    import json

    import soundfile as sf
    import torch

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "build_feature_cache.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_build_feature_cache_run", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeExtractor:
        model_path = "fake-qwen"
        device = "cpu"

        def __init__(self) -> None:
            self.model = torch.nn.Linear(1, 1)

        def extract_batch(self, audios, *, sample_rate: int):
            assert sample_rate == 16000
            return [np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32) for _audio in audios]

        def close(self) -> None:
            pass

    audio_path = tmp_path / "clip.wav"
    sf.write(str(audio_path), np.zeros(1600, dtype=np.float32), 16000)
    labels_path = tmp_path / "labels.jsonl"
    write_jsonl(
        labels_path,
        [
            build_negative_record(
                audio_id="clip",
                source="unit",
                duration_s=0.1,
                frame_hop_s=0.02,
            )
        ],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps([{"audio_id": "clip", "audio": str(audio_path)}], ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "build_ptm_feature_extractor", lambda _config: FakeExtractor())
    monkeypatch.setattr(
        module,
        "extract_mfcc",
        lambda _audio, *, sample_rate, config: np.ones((5, 4), dtype=np.float32),
    )

    module.run(
        module.parse_args(
            [
                "--labels",
                str(labels_path),
                "--manifest",
                str(manifest_path),
                "--ptm",
                "qwen3-asr-0.6b",
                "--device",
                "cpu",
                "--output-dir",
                str(tmp_path / "cache"),
            ]
        )
    )

    rows = json.loads((tmp_path / "cache" / "feature_manifest.json").read_text(encoding="utf-8"))
    assert rows[0]["frame_count"] == 5
    assert rows[0]["whisper_dim"] == 3
    assert rows[0]["mfcc_dim"] == 4
    cached_whisper, cached_mfcc = np.load(rows[0]["feature_path"])["whisper"], np.load(rows[0]["feature_path"])["mfcc"]
    assert cached_whisper.shape == (5, 3)
    assert cached_mfcc.shape == (5, 4)


def test_train_addition_bilstm_cli_requires_inputs():
    import importlib.util

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "train_addition_bilstm.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_train_addition_args", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        module.parse_args([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("parse_args should require inputs")

    args = module.parse_args(
        [
            "--labels",
            "labels.jsonl",
            "--feature-manifest",
            "feature_manifest.json",
            "--max-steps",
            "1",
        ]
    )
    assert args.labels == ["labels.jsonl"]
    assert args.feature_manifest == ["feature_manifest.json"]
    assert args.max_steps == 1
    assert args.log_interval_steps == 0
    assert args.batch_size == 1
    assert args.positive_loss_weight == 1.0

    args = module.parse_args(
        [
            "--labels",
            "labels.jsonl",
            "--feature-manifest",
            "feature_manifest.json",
            "--positive-loss-weight",
            "2.0",
        ]
    )
    assert args.positive_loss_weight == 2.0


def test_evaluate_addition_bilstm_cli_requires_inputs():
    import importlib.util

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "evaluate_addition_bilstm.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_eval_addition_args", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        module.parse_args([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("parse_args should require inputs")

    args = module.parse_args(
        [
            "--labels",
            "labels.jsonl",
            "--feature-manifest",
            "feature_manifest.json",
            "--checkpoint",
            "model.pt",
        ]
    )
    assert args.checkpoint == "model.pt"
    assert args.threshold == 0.5

    args = module.parse_args(
        [
            "--labels",
            "labels.jsonl",
            "--feature-manifest",
            "feature_manifest.json",
            "--checkpoint",
            "model.pt",
            "--threshold",
            "0.35",
        ]
    )
    assert args.threshold == 0.35


def test_export_addition_predictions_cli_writes_jsonl(tmp_path):
    import importlib.util
    import json

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "export_addition_predictions.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_export_addition_predictions", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    feature_path = tmp_path / "feature.npz"
    np.savez_compressed(
        feature_path,
        whisper=np.ones((6, 8), dtype=np.float32),
        mfcc=np.ones((6, 4), dtype=np.float32),
    )
    labels_path = tmp_path / "labels.jsonl"
    record = build_teacher_record(
        audio_id="clip",
        source="unit",
        duration_s=0.12,
        teacher_segments={"a": [(0.0, 0.06)], "b": [(0.0, 0.06)]},
        frame_hop_s=0.02,
    )
    write_jsonl(labels_path, [record])
    feature_manifest_path = tmp_path / "feature_manifest.json"
    feature_manifest_path.write_text(
        json.dumps(
            [
                {
                    "audio_id": "clip",
                    "feature_path": str(feature_path),
                    "label_index": 0,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    train_metrics = train_addition_fusion_classifier(
        records=[record],
        feature_manifest_rows=[
            {
                "audio_id": "clip",
                "feature_path": str(feature_path),
                "label_index": 0,
            }
        ],
        output_dir=tmp_path / "train",
        config=FeatureTrainConfig(
            max_steps=1,
            fusion_dim=8,
            hidden_dim=4,
            layers=1,
            max_trainable_parameters=10_000,
        ),
    )

    module.run(
        module.parse_args(
            [
                "--labels",
                str(labels_path),
                "--feature-manifest",
                str(feature_manifest_path),
                "--checkpoint",
                train_metrics.checkpoint,
                "--threshold",
                "0.0",
                "--output-dir",
                str(tmp_path / "predictions"),
            ]
        )
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "predictions" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads((tmp_path / "predictions" / "prediction_metrics.json").read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["audio_id"] == "clip"
    assert rows[0]["speech_frames"] == [1, 1, 1, 1, 1, 1]
    assert rows[0]["probability_summary"]["count"] == 6.0
    assert summary["rows"] == 1
    assert summary["metrics"]["predicted_positive_ratio"] == 1.0


def test_export_addition_predictions_cli_requires_inputs():
    import importlib.util

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "export_addition_predictions.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_export_addition_args", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        module.parse_args([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("parse_args should require inputs")

    args = module.parse_args(
        [
            "--labels",
            "labels.jsonl",
            "--feature-manifest",
            "feature_manifest.json",
            "--checkpoint",
            "model.pt",
            "--threshold",
            "0.05",
        ]
    )
    assert args.threshold == 0.05


def test_export_fusionvad_operating_point_wraps_predictions_and_recall(tmp_path):
    import importlib.util
    import json

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "export_fusionvad_operating_point.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_export_operating_point", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    feature_path = tmp_path / "feature.npz"
    np.savez_compressed(
        feature_path,
        whisper=np.ones((4, 8), dtype=np.float32),
        mfcc=np.ones((4, 4), dtype=np.float32),
    )
    labels_path = tmp_path / "labels.jsonl"
    record = build_supervised_record(
        audio_id="clip",
        source="unit",
        duration_s=0.08,
        speech_segments=[(0.02, 0.06)],
        frame_hop_s=0.02,
    )
    write_jsonl(labels_path, [record])
    feature_manifest_path = tmp_path / "feature_manifest.json"
    feature_rows = [{"audio_id": "clip", "feature_path": str(feature_path), "label_index": 0}]
    feature_manifest_path.write_text(json.dumps(feature_rows, ensure_ascii=False), encoding="utf-8")
    train_metrics = train_addition_fusion_classifier(
        records=[record],
        feature_manifest_rows=feature_rows,
        output_dir=tmp_path / "train",
        config=FeatureTrainConfig(
            max_steps=1,
            fusion_dim=8,
            hidden_dim=4,
            layers=1,
            max_trainable_parameters=10_000,
        ),
    )

    args = module.parse_args(
        [
            "--labels",
            str(labels_path),
            "--feature-manifest",
            str(feature_manifest_path),
            "--checkpoint",
            train_metrics.checkpoint,
            "--threshold",
            "0.0",
            "--pad-s",
            "0.02",
            "--output-dir",
            str(tmp_path / "op"),
        ]
    )
    assert args.operating_point == "fusionvad-ja-v1.5-posw2"
    module.run(args)

    summary = json.loads((tmp_path / "op" / "operating_point_summary.json").read_text(encoding="utf-8"))
    recall = json.loads((tmp_path / "op" / "high_recall_metrics.json").read_text(encoding="utf-8"))
    assert summary["threshold"] == 0.0
    assert summary["pad_s"] == 0.02
    assert summary["padded"]["recall"] == recall["recall"]
    assert recall["prediction_threshold"] == 0.0
    assert recall["threshold"] == 0.0
    assert (tmp_path / "op" / "frame-predictions" / "predictions.jsonl").exists()


def test_calibrate_addition_threshold_cli_requires_inputs():
    import importlib.util

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "calibrate_addition_threshold.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_calibrate_addition_args", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        module.parse_args([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("parse_args should require inputs")

    args = module.parse_args(
        [
            "--labels",
            "labels.jsonl",
            "--feature-manifest",
            "feature_manifest.json",
            "--checkpoint",
            "model.pt",
            "--step",
            "0.1",
        ]
    )
    assert args.step == 0.1


def test_evaluate_vad_baselines_cli_defaults_to_current_backends():
    import importlib.util

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "evaluate_vad_baselines.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_eval_baselines_args", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        module.parse_args([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("parse_args should require --labels")

    args = module.parse_args(["--labels", "labels.jsonl", "--backend", "fusion_lite"])
    assert args.backend == ["fusion_lite"]


def test_materialize_hf_audio_cli_defaults_to_galgame():
    import importlib.util

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "materialize_hf_audio.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_materialize_hf_args", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.parse_args(["--limit", "2"])
    assert args.dataset == "litagin/Galgame_Speech_ASR_16kHz"
    assert args.split == "train"
    assert args.start_index == 0
    assert args.limit == 2
    assert args.shuffle_buffer_size == 0
    assert args.shuffle_seed == 13

    args = module.parse_args(["--start-index", "5", "--limit", "2"])
    assert args.start_index == 5

    args = module.parse_args(["--limit", "2", "--shuffle-buffer-size", "128", "--shuffle-seed", "29"])
    assert args.shuffle_buffer_size == 128
    assert args.shuffle_seed == 29


def test_build_galgame_weak_labels_cli_requires_manifest():
    import importlib.util

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "build_galgame_weak_labels.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_galgame_weak_args", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        module.parse_args([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("parse_args should require --manifest")

    args = module.parse_args(["--manifest", "hf_audio_manifest.json"])
    assert args.manifest == "hf_audio_manifest.json"
    assert args.teacher_name == "galgame_weak_positive"


def test_build_galgame_synthetic_timeline_writes_exact_labels(tmp_path):
    import importlib.util
    import json

    import soundfile as sf

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "build_galgame_synthetic_timeline.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_galgame_synthetic_timeline", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    first_audio = tmp_path / "first.wav"
    second_audio = tmp_path / "second.wav"
    sf.write(str(first_audio), np.ones(3200, dtype=np.float32) * 0.1, 16000)
    sf.write(str(second_audio), np.ones(3200, dtype=np.float32) * 0.2, 16000)
    source_manifest = tmp_path / "hf_audio_manifest.json"
    source_manifest.write_text(
        json.dumps(
            [
                {"audio_id": "first", "audio": str(first_audio), "text": "a", "input": "src:0"},
                {"audio_id": "second", "audio": str(second_audio), "text": "b", "input": "src:1"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    module.build_synthetic_timeline(
        module.parse_args(
            [
                "--manifest",
                str(source_manifest),
                "--count",
                "1",
                "--speech-clips-per-example",
                "2",
                "--frame-hop-s",
                "0.1",
                "--max-speech-s",
                "0.2",
                "--gap-min-s",
                "0.1",
                "--gap-max-s",
                "0.1",
                "--leading-gap-min-s",
                "0.1",
                "--leading-gap-max-s",
                "0.1",
                "--trailing-gap-min-s",
                "0.1",
                "--trailing-gap-max-s",
                "0.1",
                "--output-dir",
                str(tmp_path / "synthetic"),
            ]
        )
    )

    records = read_jsonl(tmp_path / "synthetic" / "labels.jsonl")
    manifest = json.loads((tmp_path / "synthetic" / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "synthetic" / "synthetic_timeline_summary.json").read_text(encoding="utf-8"))
    audio, sample_rate = sf.read(str(tmp_path / "synthetic" / "audio" / "galgame-synth-000000.wav"), dtype="float32")

    assert len(records) == 1
    assert records[0].label_quality == "supervised"
    assert records[0].speech_frames == [0, 1, 1, 0, 1, 1, 0]
    assert records[0].text == "a b"
    assert manifest[0]["source_audio_ids"] == ["first", "second"]
    assert sample_rate == 16000
    assert audio.shape[0] == 11200
    assert summary["records"] == 1
    assert summary["speech_frame_ratio"] == 4 / 7


def test_build_galgame_synthetic_timeline_uses_real_negative_gap_and_background(tmp_path):
    import importlib.util
    import json

    import soundfile as sf

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "build_galgame_synthetic_timeline.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_galgame_synthetic_timeline_real_gap", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    first_audio = tmp_path / "first.wav"
    second_audio = tmp_path / "second.wav"
    negative_audio = tmp_path / "negative.wav"
    sf.write(str(first_audio), np.ones(3200, dtype=np.float32) * 0.1, 16000)
    sf.write(str(second_audio), np.ones(3200, dtype=np.float32) * 0.2, 16000)
    sf.write(str(negative_audio), np.ones(16000, dtype=np.float32) * -0.25, 16000)
    source_manifest = tmp_path / "hf_audio_manifest.json"
    source_manifest.write_text(
        json.dumps(
            [
                {"audio_id": "first", "audio": str(first_audio), "text": "a", "input": "src:0"},
                {"audio_id": "second", "audio": str(second_audio), "text": "b", "input": "src:1"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    negative_manifest = tmp_path / "negative_manifest.json"
    negative_manifest.write_text(
        json.dumps(
            [{"audio_id": "neg", "audio": str(negative_audio), "source": "unit-negative"}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    module.build_synthetic_timeline(
        module.parse_args(
            [
                "--manifest",
                str(source_manifest),
                "--count",
                "1",
                "--speech-clips-per-example",
                "2",
                "--frame-hop-s",
                "0.1",
                "--max-speech-s",
                "0.2",
                "--gap-min-s",
                "0.1",
                "--gap-max-s",
                "0.1",
                "--leading-gap-min-s",
                "0.1",
                "--leading-gap-max-s",
                "0.1",
                "--trailing-gap-min-s",
                "0.1",
                "--trailing-gap-max-s",
                "0.1",
                "--negative-manifest",
                str(negative_manifest),
                "--negative-gap-prob",
                "1.0",
                "--background-manifest",
                str(negative_manifest),
                "--background-mix-prob",
                "1.0",
                "--background-snr-db-min",
                "10",
                "--background-snr-db-max",
                "10",
                "--output-dir",
                str(tmp_path / "synthetic"),
            ]
        )
    )

    records = read_jsonl(tmp_path / "synthetic" / "labels.jsonl")
    summary = json.loads((tmp_path / "synthetic" / "synthetic_timeline_summary.json").read_text(encoding="utf-8"))
    details = [
        json.loads(line)
        for line in (tmp_path / "synthetic" / "synthetic_timeline_details.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert records[0].speech_frames == [0, 1, 1, 0, 1, 1, 0]
    assert summary["gap_mode_counts"] == {"real_negative": 3}
    assert summary["negative_rows"] == 1
    assert summary["background_rows"] == 1
    assert summary["background_mix_count"] == 1
    assert details[0]["background_mix"]["audio_id"] == "neg"
    assert [item["gap"] for item in details[0]["sources"] if item.get("mode") == "real_negative"] == [
        "leading",
        "middle-0",
        "trailing",
    ]


def test_build_galgame_synthetic_timeline_speech_label_pad_expands_labels(tmp_path):
    import importlib.util
    import json

    import soundfile as sf

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "build_galgame_synthetic_timeline.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_galgame_synthetic_timeline_label_pad", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    audio_path = tmp_path / "speech.wav"
    sf.write(str(audio_path), np.ones(3200, dtype=np.float32) * 0.1, 16000)
    source_manifest = tmp_path / "hf_audio_manifest.json"
    source_manifest.write_text(
        json.dumps([{"audio_id": "speech", "audio": str(audio_path), "text": "a"}], ensure_ascii=False),
        encoding="utf-8",
    )

    module.build_synthetic_timeline(
        module.parse_args(
            [
                "--manifest",
                str(source_manifest),
                "--count",
                "1",
                "--speech-clips-per-example",
                "1",
                "--frame-hop-s",
                "0.05",
                "--max-speech-s",
                "0.1",
                "--gap-min-s",
                "0",
                "--gap-max-s",
                "0",
                "--leading-gap-min-s",
                "0.1",
                "--leading-gap-max-s",
                "0.1",
                "--trailing-gap-min-s",
                "0.1",
                "--trailing-gap-max-s",
                "0.1",
                "--speech-label-pad-s",
                "0.05",
                "--output-dir",
                str(tmp_path / "synthetic"),
            ]
        )
    )

    records = read_jsonl(tmp_path / "synthetic" / "labels.jsonl")
    manifest = json.loads((tmp_path / "synthetic" / "manifest.json").read_text(encoding="utf-8"))

    assert records[0].speech_frames == [0, 1, 1, 1, 1, 0]
    assert manifest[0]["actual_speech_segments"] == [{"end": 0.2, "start": 0.1}]
    assert manifest[0]["speech_segments"] == [{"end": 0.25, "start": 0.05}]


def test_build_galgame_synthetic_timeline_cli_requires_manifest():
    import importlib.util

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "build_galgame_synthetic_timeline.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_galgame_synthetic_timeline_args", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        module.parse_args([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("parse_args should require --manifest")

    args = module.parse_args(["--manifest", "hf_audio_manifest.json", "--count", "3"])
    assert args.manifest == "hf_audio_manifest.json"
    assert args.count == 3
    assert args.speech_clips_per_example == 2
    assert args.negative_gap_prob == 0.0
    assert args.background_mix_prob == 0.0
    assert args.speech_label_pad_s == 0.0


def test_build_local_video_audit_candidates_writes_manifest(tmp_path, monkeypatch):
    import importlib.util
    import json
    from types import SimpleNamespace

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "build_local_video_audit_candidates.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_local_video_audit", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "b.mp4").write_bytes(b"video")
    (video_dir / "a.mkv").write_bytes(b"video")
    (video_dir / "ignore.txt").write_text("", encoding="utf-8")

    commands = []

    def fake_run(command, check=True, capture_output=False, text=False):
        commands.append(command)
        if command[0] == "ffprobe":
            return SimpleNamespace(stdout="120.0\n")
        if command[0] == "ffmpeg":
            __import__("pathlib").Path(command[-1]).write_bytes(b"RIFF")
            return SimpleNamespace(stdout="")
        raise AssertionError(command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module.build_candidates(
        module.parse_args(
            [
                "--input",
                str(video_dir),
                "--clips-per-video",
                "2",
                "--clip-duration-s",
                "4",
                "--exclude-head-s",
                "10",
                "--exclude-tail-s",
                "10",
                "--seed",
                "7",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "audit_candidates.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "out" / "candidate_summary.json").read_text(encoding="utf-8"))

    assert len(rows) == 4
    assert len(manifest) == 4
    assert summary["clips"] == 4
    assert summary["videos"] == 2
    assert rows[0]["source"] == "local-video-heldout"
    assert rows[0]["label_quality"] == "manual_pending"
    assert rows[0]["duration_s"] == 4.0
    assert __import__("pathlib").Path(tmp_path / "out" / rows[0]["audio"]).exists() or __import__(
        "pathlib"
    ).Path(rows[0]["audio"]).exists()
    assert sum(1 for command in commands if command[0] == "ffprobe") == 2
    assert sum(1 for command in commands if command[0] == "ffmpeg") == 4


def test_build_local_video_audit_candidates_cli_requires_input():
    import importlib.util

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "build_local_video_audit_candidates.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_local_video_audit_args", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        module.parse_args([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("parse_args should require --input")

    args = module.parse_args(["--input", "video", "--clips-per-video", "3"])
    assert args.input == ["video"]
    assert args.clips_per_video == 3
    assert args.source == "local-video-heldout"


def test_build_teacher_pseudo_labels_cli_requires_manifest():
    import importlib.util

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "build_teacher_pseudo_labels.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_teacher_pseudo_args", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        module.parse_args([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("parse_args should require --manifest")

    args = module.parse_args(
        [
            "--manifest",
            "hf_audio_manifest.json",
            "--backend",
            "whisperseg-adaptive",
            "--backend",
            "fusion_lite",
            "--limit",
            "8",
            "--teacher-weight",
            "0.4",
        ]
    )
    assert args.manifest == "hf_audio_manifest.json"
    assert args.backend == ["whisperseg-adaptive", "fusion_lite"]
    assert args.limit == 8
    assert args.teacher_weight == 0.4


def test_export_qwen_asr_sft_filters_and_copies_audio(tmp_path):
    import importlib.util
    import json

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "export_qwen_asr_sft.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_export_qwen_asr_sft", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"RIFF")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "audio_id": "keep",
                    "audio": str(audio_path),
                    "text": " こんにちは\n世界 ",
                    "source": "unit",
                    "duration_s": 1.0,
                },
                {
                    "audio_id": "missing_text",
                    "audio": str(audio_path),
                    "text": "",
                    "duration_s": 1.0,
                },
                {
                    "audio_id": "too_long",
                    "audio": str(audio_path),
                    "text": "長い",
                    "duration_s": 99.0,
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    module.run(
        module.parse_args(
            [
                "--manifest",
                str(manifest_path),
                "--split",
                "train",
                "--copy-audio",
                "--max-duration-s",
                "30",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads((tmp_path / "out" / "train_summary.json").read_text(encoding="utf-8"))
    skipped = json.loads((tmp_path / "out" / "train_skipped.json").read_text(encoding="utf-8"))

    assert len(rows) == 1
    assert rows[0]["text"] == "こんにちは 世界"
    assert rows[0]["language"] == "Japanese"
    assert (tmp_path / "out" / "audio" / "keep.wav").exists()
    assert summary["kept"] == 1
    assert summary["skip_counts"] == {"duration_too_long": 1, "missing_text": 1}
    assert [row["reason"] for row in skipped] == ["missing_text", "duration_too_long"]


def test_prepare_qwen_asr_sft_dataset_merges_sources_and_hard_negatives(tmp_path):
    import importlib.util
    import json

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "prepare_qwen_asr_sft_dataset.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_prepare_qwen_asr_sft", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"RIFF")
    asr_manifest = tmp_path / "asr.json"
    asr_manifest.write_text(
        json.dumps(
            [
                {"audio_id": "asr-val", "audio": str(audio_path), "txt": " 検証 ", "duration_s": 1.0},
                {"audio_id": "asr-test", "audio": str(audio_path), "txt": "テスト", "duration_s": 1.0},
                {"audio_id": "asr-train", "audio": str(audio_path), "txt": "こんにちは\n世界", "duration_s": 1.0},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    ser_manifest = tmp_path / "ser.json"
    ser_manifest.write_text(
        json.dumps(
            [
                {"audio_id": "ser-train", "audio": str(audio_path), "text": "感情", "duration_s": 1.0, "cls": "3"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    hard_negative = tmp_path / "hard.jsonl"
    hard_negative.write_text(
        json.dumps({"audio_id": "neg", "audio": str(audio_path), "text": "", "duration_s": 1.0}) + "\n",
        encoding="utf-8",
    )

    assert module.qwen_asr_sft_text(transcript=" こんにちは\n世界 ", language="Japanese") == (
        "language Japanese<asr_text>こんにちは 世界"
    )
    args = module.parse_args(
        [
            "--mode",
            "smoke",
            "--asr-manifest",
            str(asr_manifest),
            "--ser-manifest",
            str(ser_manifest),
            "--asr-train-limit",
            "1",
            "--asr-val-limit",
            "1",
            "--asr-test-limit",
            "1",
            "--ser-train-limit",
            "1",
            "--ser-val-limit",
            "0",
            "--ser-test-limit",
            "0",
            "--hard-negative-jsonl",
            str(hard_negative),
            "--hard-negative-limit",
            "1",
            "--output-root",
            str(tmp_path / "out"),
        ]
    )
    module.run(args)

    train_rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "qwen-sft" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    val_rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "qwen-sft" / "val.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    test_rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "qwen-sft" / "test.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads((tmp_path / "out" / "qwen_sft_dataset_summary.json").read_text(encoding="utf-8"))
    train_manifest = [
        json.loads(line)
        for line in (tmp_path / "out" / "manifest" / "train.manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["split_counts"] == {"test": 1, "train": 3, "val": 1}
    assert val_rows == [{"audio": str(tmp_path / "out" / "audio" / "galgame-asr" / "galgame-asr-asr-val.wav"), "text": "language Japanese<asr_text>検証"}]
    assert test_rows[0]["text"] == "language Japanese<asr_text>テスト"
    assert [row["text"] for row in train_rows] == [
        "language Japanese<asr_text>こんにちは 世界",
        "language Japanese<asr_text>感情",
        "language Japanese<asr_text>",
    ]
    assert {row["source_key"] for row in train_manifest} == {"galgame-asr", "galgame-ser", "hard-negative"}
    assert any(row["metadata"].get("cls") == "3" for row in train_manifest)


def test_prepare_qwen_asr_sft_dataset_can_store_hf_ogg_bytes(tmp_path):
    import importlib.util
    import io

    import soundfile as sf

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "prepare_qwen_asr_sft_dataset.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_prepare_qwen_asr_sft_ogg", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
    audio_path, duration_s, sample_rate, error = module.write_hf_audio(
        row={"ogg": buffer.getvalue()},
        target_audio_dir=tmp_path,
        audio_id="sample",
        audio_format="ogg",
    )

    assert error is None
    assert audio_path == str(tmp_path / "sample.ogg")
    assert (tmp_path / "sample.ogg").read_bytes() == buffer.getvalue()
    assert duration_s == 0.1
    assert sample_rate == 16000


def test_prepare_qwen_asr_sft_dataset_cli_defaults_are_cloud_safe():
    import importlib.util

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "prepare_qwen_asr_sft_dataset.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_prepare_qwen_asr_sft_args", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.parse_args([])
    assert args.mode == "smoke"
    assert args.shuffle_buffer_size == 128
    assert args.asr_train_limit == 40
    assert args.ser_train_limit == 10
    assert args.hf_audio_format == "wav"
    assert args.hf_xet_high_performance is True

    args = module.parse_args(["--mode", "full"])
    assert args.output_root.endswith("datasets/train/qwen3-asr-ja-galgame/v1-full")
    assert args.asr_train_limit == 0
    assert args.ser_train_limit == 0
    assert args.asr_val_limit == 1000
    assert args.ser_val_limit == 500
    full_plan = module.SourcePlan(
        source_key="asr",
        dataset="dataset",
        manifest=None,
        enabled=True,
        train_limit=0,
        val_limit=1,
        test_limit=1,
    )
    assert module.source_is_complete({"val": 1, "test": 1}, plan=full_plan) is False


def test_prepare_qwen_asr_cloud_assets_script_downloads_model_and_data():
    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "prepare_qwen_asr_cloud_assets.sh"
    )
    content = script_path.read_text(encoding="utf-8")

    assert "Qwen/Qwen3-ASR-1.7B" in content
    assert "models/Qwen-Qwen3-ASR-1.7B" in content
    assert ".venv/bin/huggingface-cli download" in content
    assert "prepare_qwen_asr_sft_dataset.py" in content
    assert "HF_XET_HIGH_PERFORMANCE" in content
    assert "SFT_HF_AUDIO_FORMAT=\"${SFT_HF_AUDIO_FORMAT:-ogg}\"" in content
    assert "SFT_ASR_TRAIN_LIMIT=\"${SFT_ASR_TRAIN_LIMIT:-200000}\"" in content
    assert "--no-ser" in content


def test_export_manual_audit_asr_sft_candidates_splits_empty_and_review(tmp_path):
    import importlib.util
    import json

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "export_manual_audit_asr_sft_candidates.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_export_manual_audit_asr_sft_candidates", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"RIFF")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "audio_id": "neg",
                    "audio": str(audio_path),
                    "duration_s": 1.0,
                    "source": "unit",
                    "manual_reason": "manual_negative_asr_text",
                    "label_quality": "negative",
                    "text": "ASR: んっ…\nraw: んっ…",
                },
                {
                    "audio_id": "speech",
                    "audio": str(audio_path),
                    "duration_s": 2.0,
                    "source": "unit",
                    "manual_reason": "no_overlap_asr_text",
                    "label_quality": "supervised",
                    "text": "ASR: 逃げて\nraw: …逃げて",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert module.parse_candidate_asr_text("ASR: んっ…\nraw: x") == "んっ…"
    module.run(
        module.parse_args(
            [
                "--manifest",
                str(manifest_path),
                "--split",
                "train",
                "--copy-audio",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
    )

    empty_rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "train_empty_hard_negative.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    review_rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "train_speech_review.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads((tmp_path / "out" / "train_summary.json").read_text(encoding="utf-8"))

    assert empty_rows[0]["audio_id"] == "neg"
    assert empty_rows[0]["text"] == ""
    assert empty_rows[0]["label_type"] == "empty_hard_negative"
    assert review_rows[0]["candidate_asr_text"] == "逃げて"
    assert review_rows[0]["text"] == ""
    assert review_rows[0]["label_type"] == "needs_manual_transcript"
    assert summary["empty_hard_negative_records"] == 1
    assert summary["speech_review_records"] == 1
    assert (tmp_path / "out" / "audio" / "neg.wav").exists()


def test_probe_qwen_asr_text_distance_and_selection(tmp_path):
    import importlib.util

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "probe_qwen_asr.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_probe_qwen_asr", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"")
    rows = [
        {"audio_id": "a", "audio": str(audio_path), "text": "こんにちは。"},
        {"audio_id": "b", "audio": str(audio_path), "text": ""},
        {"audio_id": "c", "audio": str(tmp_path / "missing.wav"), "text": "x"},
    ]

    assert module.normalize_asr_eval_text("こ ん、にちは。") == "こんにちは"
    assert module.levenshtein_distance("abc", "adc") == 1
    distance = module.text_distance("こんにちは。", "こんにちわ")
    assert distance.reference_chars == 5
    assert distance.hypothesis_chars == 5
    assert distance.distance == 1
    assert distance.cer == 0.2
    assert [row["audio_id"] for row in module.select_rows(rows, limit=1)] == ["a"]

    try:
        module.parse_args([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("parse_args should require --manifest")


def test_vad_recall_metrics_reports_padding_tradeoff(tmp_path):
    import importlib.util
    import json

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "vad_recall_metrics.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_vad_recall_metrics", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    labels_path = tmp_path / "labels.jsonl"
    write_jsonl(
        labels_path,
        [
            build_supervised_record(
                audio_id="clip",
                source="unit",
                duration_s=0.08,
                speech_segments=[(0.02, 0.06)],
                frame_hop_s=0.02,
            )
        ],
    )
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps({"audio_id": "clip", "speech_frames": [0, 1, 0, 0]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    assert module.padded_predictions([0, 1, 0, 0], pad_frames=1) == [1, 1, 1, 0]
    assert module.count_missed_speech_segments([1, 1, 0, 1], [0, 1, 0, 0]) == 2
    module.run(
        module.parse_args(
            [
                "--labels",
                str(labels_path),
                "--predictions",
                str(predictions_path),
                "--pad-s",
                "0.02",
                "--frame-hop-s",
                "0.02",
                "--output",
                str(tmp_path / "recall.json"),
            ]
        )
    )

    summary = json.loads((tmp_path / "recall.json").read_text(encoding="utf-8"))
    assert summary["evaluated"] == 1
    assert summary["recall"] == 1.0
    assert summary["missed_speech_seconds"] == 0.0
    assert summary["extra_audio_seconds"] == 0.02
    assert summary["extra_audio_ratio"] == 1.5


def test_select_audit_candidates_outputs_stratified_rows(tmp_path):
    import importlib.util
    import json

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "select_audit_candidates.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_select_audit_candidates", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    records = [
        build_weighted_teacher_record(
            audio_id="conflict",
            source="unit",
            duration_s=1.0,
            text="a",
            teacher_segments={"a": [(0.0, 0.8)], "b": []},
            frame_hop_s=0.5,
        ),
        build_weighted_teacher_record(
            audio_id="clean",
            source="unit",
            duration_s=1.0,
            text="b",
            teacher_segments={"a": [(0.0, 1.0)], "b": [(0.0, 1.0)]},
            frame_hop_s=0.5,
        ),
    ]
    labels_path = tmp_path / "labels.jsonl"
    write_jsonl(labels_path, records)
    audit_path = tmp_path / "audit.jsonl"
    audit_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "audio_id": "conflict",
                        "audio": "conflict.wav",
                        "duration_s": 1.0,
                        "label_quality": "teacher_conflict",
                        "frames": 2,
                        "active_frames": 0,
                        "ignored_frames": 2,
                        "active_frame_ratio": 0.0,
                        "ignored_frame_ratio": 1.0,
                        "conflict_frames": 2,
                        "conflict_frame_ratio": 1.0,
                        "weighted_speech_frames": 0,
                        "weighted_negative_frames": 0,
                    }
                ),
                json.dumps(
                    {
                        "audio_id": "clean",
                        "audio": "clean.wav",
                        "duration_s": 1.0,
                        "label_quality": "teacher_agree",
                        "frames": 2,
                        "active_frames": 2,
                        "ignored_frames": 0,
                        "active_frame_ratio": 1.0,
                        "ignored_frame_ratio": 0.0,
                        "conflict_frames": 0,
                        "conflict_frame_ratio": 0.0,
                        "weighted_speech_frames": 2,
                        "weighted_negative_frames": 0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    module.run(
        module.parse_args(
            [
                "--labels",
                str(labels_path),
                "--audit",
                str(audit_path),
                "--per-bucket",
                "1",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
    )

    summary = json.loads((tmp_path / "out" / "audit_candidate_summary.json").read_text(encoding="utf-8"))
    assert summary["candidates"] == 2
    assert summary["reason_counts"]["teacher_conflict_high"] == 1
    assert (tmp_path / "out" / "audit_candidates.csv").exists()


def test_generate_manual_audit_html_embeds_candidates(tmp_path):
    import importlib.util
    import json
    import wave

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "generate_manual_audit_html.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_generate_manual_audit_html", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    audio_path = tmp_path / "clip.wav"
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 1600)
    candidates_path = tmp_path / "audit_candidates.jsonl"
    candidates_path.write_text(
        json.dumps(
            {
                "audio_id": "clip",
                "audio": str(audio_path),
                "duration_s": 0.1,
                "source": "unit",
                "text": "テスト",
                "reason": "teacher_conflict_high",
                "label_quality": "teacher_conflict",
                "teacher_segments": {"a": [{"start": 0.0, "end": 0.1, "score": 0.9}]},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_html = tmp_path / "audit" / "manual_audit.html"

    module.run(
        module.parse_args(
            [
                "--candidates",
                str(candidates_path),
                "--output-html",
                str(output_html),
                "--dataset-id",
                "unit-dataset",
                "--copy-audio",
            ]
        )
    )

    html = output_html.read_text(encoding="utf-8")
    assert "FusionVAD-JA 人工审计标注" in html
    assert "manual_labels.jsonl" in html
    assert "teacher_conflict_high" in html
    assert "テスト" in html
    assert "快速审计优先使用四类结果" in html
    assert "startToHereBtn" in html
    assert "hereToEndBtn" in html
    assert "Teacher 并集" in html
    assert "function hasTeacherData" in html
    assert "只设起点时默认到音频末尾" in html
    assert "function hasTimeValue" in html
    assert "function commitPendingStart" in html
    assert "addSegment(0, end)" in html
    assert (tmp_path / "audit" / "audio" / "clip.wav").exists()

    no_teacher_html = module.html_template(
        title="无 teacher 审计",
        dataset_id="unit-no-teacher",
        output_jsonl_name="manual_labels.jsonl",
        candidates=[
            {
                "audio_id": "clip",
                "audio": str(audio_path),
                "audio_url": "audio/clip.wav",
                "duration_s": 0.1,
                "source": "unit",
                "text": "ASR: んっ",
                "reason": "manual_negative_asr_text",
                "label_quality": "negative",
                "teacher_segments": {},
            }
        ],
    )
    assert "Teacher 并集" not in no_teacher_html
    assert "Teacher 交集" not in no_teacher_html
    assert "凡是希望送入 ASR" in no_teacher_html


def test_convert_manual_audit_labels_builds_strong_labels(tmp_path):
    import importlib.util
    import json
    import wave

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "convert_manual_audit_labels.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_convert_manual_audit_labels", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    audio_path = tmp_path / "clip.wav"
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 32000)
    manual_path = tmp_path / "manual_labels.jsonl"
    rows = [
        {
            "audio_id": "positive",
            "audio": str(audio_path),
            "duration_s": 2.0,
            "source": "unit",
            "text": "テスト",
            "reason": "clean_teacher_agree",
            "label_quality": "manual",
            "speech_segments": [{"end": 0.1}, {"start": 0.25, "end": 1.25}, {"start": 1.5}],
            "reviewed": True,
            "skip_reason": "",
            "notes": "",
        },
        {
            "audio_id": "negative",
            "audio": str(audio_path),
            "duration_s": 2.0,
            "source": "unit",
            "text": "…",
            "reason": "teacher_conflict_high",
            "label_quality": "manual",
            "speech_segments": [],
            "reviewed": True,
            "skip_reason": "",
            "notes": "",
        },
        {
            "audio_id": "skipped",
            "audio": str(audio_path),
            "duration_s": 2.0,
            "source": "unit",
            "text": "",
            "reason": "ignored_ratio_high",
            "label_quality": "manual",
            "speech_segments": [],
            "reviewed": True,
            "skip_reason": "uncertain",
            "notes": "",
        },
    ]
    manual_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    module.run(
        module.parse_args(
            [
                "--input",
                str(manual_path),
                "--output-dir",
                str(tmp_path / "strong"),
            ]
        )
    )

    records = read_jsonl(tmp_path / "strong" / "labels.jsonl")
    summary = json.loads((tmp_path / "strong" / "manual_label_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "strong" / "manifest.json").read_text(encoding="utf-8"))

    assert [record.label_quality for record in records] == ["supervised", "negative"]
    assert records[0].teacher_segments["supervised"][0].start == 0.0
    assert records[0].teacher_segments["supervised"][-1].end == 2.0
    assert sum(records[0].speech_frames) == 81
    assert sum(records[1].speech_frames) == 0
    assert summary["records"] == 2
    assert summary["skipped"] == 1
    assert summary["label_quality_counts"] == {"negative": 1, "supervised": 1}
    assert manifest[0]["audio"] == str(audio_path)


def test_build_synthetic_negatives_cli_validates_count():
    import importlib.util

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "build_synthetic_negatives.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_synthetic_negative_args", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        module.parse_args(["--count", "0"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("parse_args should reject non-positive --count")

    args = module.parse_args(["--count", "3", "--duration-s", "0.5"])
    assert args.count == 3
    assert args.duration_s == 0.5
    assert args.source == "synthetic-negative"


def test_write_training_manifest_outputs_jsonl(tmp_path):
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"")
    record = build_negative_record(
        audio_id="clip",
        source="unit",
        duration_s=0.1,
        frame_hop_s=0.02,
    )
    examples, skipped = build_training_examples(
        [record],
        manifest_audio_map={"clip": str(audio_path)},
    )
    path = tmp_path / "training_manifest.jsonl"

    write_training_manifest(path=path, examples=examples)

    assert skipped == []
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


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


def test_label_jsonl_round_trip_preserves_frame_weights(tmp_path):
    path = tmp_path / "weighted-labels.jsonl"
    record = with_frame_weights(
        build_negative_record(
            audio_id="clip",
            source="unit",
            duration_s=0.5,
            frame_hop_s=0.25,
        ),
        [0.0, 0.5],
    )

    write_jsonl(path, [record])
    rows = read_jsonl(path)

    assert rows == [record]
    assert effective_frame_weights(rows[0]) == [0.0, 0.5]


def test_evaluate_vad_asr_downstream_helpers():
    import importlib.util

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "evaluate_vad_asr_downstream.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_evaluate_vad_asr_downstream", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.clean_vad_name("whisperseg-adaptive") == "whisperseg_adaptive"
    assert module.padded_frames([0, 1, 0, 0], pad_frames=1) == [1, 1, 1, 0]
    segments = module.frames_to_segments([0, 1, 1, 0, 1], frame_hop_s=0.1, duration_s=0.45)
    assert [(round(seg.start, 2), round(seg.end, 2)) for seg in segments] == [(0.1, 0.3), (0.4, 0.45)]

    merged = module.merge_segments(
        [SpeechSegment(0.0, 0.2), SpeechSegment(0.3, 0.4), SpeechSegment(0.8, 0.82)],
        duration_s=1.0,
        merge_gap_s=0.15,
        min_segment_s=0.05,
    )
    assert [(round(seg.start, 2), round(seg.end, 2)) for seg in merged] == [(0.0, 0.4)]

    summary = module.summarize_text_rows(
        [
            {"text": "abc", "raw_text": "abcd", "manual_overlap_s": 0.0, "manual_overlap_ratio": 0.0, "label_quality": "negative", "asr_generation": {"error_kind": None}},
            {"text": "", "raw_text": "", "manual_overlap_s": 0.2, "manual_overlap_ratio": 0.5, "label_quality": "supervised", "asr_generation": {"error_kind": "timeout"}},
        ]
    )
    assert summary["chunk_count"] == 2
    assert summary["nonempty_chunk_count"] == 1
    assert summary["negative_record_text_chars"] == 3
    assert summary["no_speech_overlap_text_chars"] == 3
    assert summary["asr_error_counts"] == {"": 1, "timeout": 1}

    args = module.parse_args(
        [
            "--labels",
            "labels.jsonl",
            "--manifest",
            "manifest.json",
            "--fusionvad-predictions",
            "predictions.jsonl",
        ]
    )
    assert args.vad == ["whisperseg-adaptive", "fusion_lite", "fusionvad"]
    assert args.asr_backend == "anime-whisper"


def test_select_asr_hard_negative_candidates_outputs_review_rows(tmp_path):
    import importlib.util
    import json

    script_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "fusionvad_ja"
        / "select_asr_hard_negative_candidates.py"
    )
    spec = importlib.util.spec_from_file_location("fusionvad_ja_select_asr_hard_negative_candidates", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    audio_path = tmp_path / "chunk.wav"
    audio_path.write_bytes(b"")
    asr_path = tmp_path / "anime_whisper_outputs.jsonl"
    rows = [
        {
            "audio_id": "neg",
            "chunk_index": 0,
            "chunk_path": str(audio_path),
            "duration_s": 1.0,
            "label_quality": "negative",
            "manual_overlap_s": 0.0,
            "manual_overlap_ratio": 0.0,
            "text": "んっ…",
            "raw_text": "んっ…",
            "vad": "fusionvad",
        },
        {
            "audio_id": "punct",
            "chunk_index": 0,
            "chunk_path": str(audio_path),
            "duration_s": 1.0,
            "label_quality": "negative",
            "manual_overlap_s": 0.0,
            "manual_overlap_ratio": 0.0,
            "text": "…",
            "raw_text": "…",
            "vad": "fusionvad",
        },
        {
            "audio_id": "nooverlap",
            "chunk_index": 1,
            "chunk_path": str(audio_path),
            "duration_s": 2.0,
            "label_quality": "supervised",
            "manual_overlap_s": 0.0,
            "manual_overlap_ratio": 0.0,
            "text": "やばい",
            "raw_text": "やばい",
            "vad": "fusionvad",
        },
        {
            "audio_id": "low",
            "chunk_index": 2,
            "chunk_path": str(audio_path),
            "duration_s": 2.0,
            "label_quality": "supervised",
            "manual_overlap_s": 0.2,
            "manual_overlap_ratio": 0.05,
            "text": "はい",
            "raw_text": "はい",
            "vad": "fusionvad",
        },
    ]
    asr_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    assert module.informative_text("… ー") == ""
    assert module.informative_text("んっ…") == "んっ"
    module.run(
        module.parse_args(
            [
                "--asr-outputs",
                str(asr_path),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
    )

    candidates = [
        json.loads(line)
        for line in (tmp_path / "out" / "audit_candidates.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads((tmp_path / "out" / "audit_candidate_summary.json").read_text(encoding="utf-8"))

    assert [row["reason"] for row in candidates] == [
        "manual_negative_asr_text",
        "no_overlap_asr_text",
        "low_overlap_asr_text",
    ]
    assert candidates[0]["audio"].endswith("chunk.wav")
    assert candidates[0]["audio_id"] == "neg__fusionvad__chunk000"
    assert "ASR: んっ…" in candidates[0]["text"]
    assert summary["rows"] == 4
    assert summary["candidates"] == 3


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
    assert audit.head_silence_s == 0.0
    assert audit.tail_silence_s == 0.0


def test_audit_audio_normalizes_input_sample_rate_and_reports_edge_silence():
    audio = np.concatenate(
        [
            np.zeros(4000, dtype=np.float32),
            np.ones(4000, dtype=np.float32) * 0.5,
        ]
    )

    normalized, sample_rate = normalize_audio_16k_mono(audio, 8000)
    audit = audit_audio(
        audio_id="clip",
        source="unit",
        audio=audio,
        sample_rate=8000,
    )

    assert sample_rate == 16000
    assert len(normalized) == 16000
    assert audit.sample_rate == 16000
    assert audit.duration_s == 1.0
    assert 0.48 <= audit.head_silence_s <= 0.52
    assert audit.tail_silence_s == 0.0
