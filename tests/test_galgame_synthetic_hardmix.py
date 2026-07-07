from __future__ import annotations

import numpy as np
import pytest

from tools.boundary.ja.build_galgame_synthetic_timeline import (
    parse_args,
    sample_binary_hardmix_layout,
    sample_half_open_int,
    stable_source_partition,
)


def test_half_open_integer_sampling_never_reaches_upper_bound() -> None:
    rng = np.random.default_rng(7)

    values = {
        sample_half_open_int(rng, minimum=2, maximum=5)
        for _ in range(1000)
    }

    assert values == {2, 3, 4}


def test_binary_hardmix_supports_touching_speech_and_consecutive_noise() -> None:
    rng = np.random.default_rng(19)
    layouts = [
        sample_binary_hardmix_layout(
            rng,
            speech_count_min=1,
            speech_count_max=5,
            edge_noise_count_min=0,
            edge_noise_count_max=5,
            inter_noise_count_min=0,
            inter_noise_count_max=5,
        )
        for _ in range(500)
    ]

    patterns = [str(layout["pattern"]) for layout in layouts]
    assert any("11" in pattern for pattern in patterns)
    assert any("00" in pattern for pattern in patterns)
    assert any(pattern.startswith("00001") for pattern in patterns)
    for layout in layouts:
        assert layout["tokens"].count(1) == layout["speech_count"]
        assert all(token in (0, 1) for token in layout["tokens"])


def test_binary_hardmix_cli_uses_explicit_half_open_ranges() -> None:
    args = parse_args(
        [
            "--manifest",
            "unused.json",
            "--timeline-pattern-mode",
            "binary_hardmix",
            "--hardmix-speech-count-min",
            "2",
            "--hardmix-speech-count-max",
            "6",
            "--hardmix-inter-noise-count-min",
            "0",
            "--hardmix-inter-noise-count-max",
            "4",
        ]
    )

    assert args.timeline_pattern_mode == "binary_hardmix"
    assert (args.hardmix_speech_count_min, args.hardmix_speech_count_max) == (2, 6)
    assert (args.hardmix_inter_noise_count_min, args.hardmix_inter_noise_count_max) == (
        0,
        4,
    )


def test_binary_hardmix_cli_rejects_empty_half_open_range() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--manifest",
                "unused.json",
                "--hardmix-edge-noise-count-min",
                "3",
                "--hardmix-edge-noise-count-max",
                "3",
            ]
        )


def test_source_partition_is_stable_and_disjoint() -> None:
    rows = [{"audio_id": f"clip-{index}"} for index in range(1000)]
    first = [
        stable_source_partition(row, train_ratio=0.85, val_ratio=0.10)
        for row in rows
    ]
    second = [
        stable_source_partition(row, train_ratio=0.85, val_ratio=0.10)
        for row in reversed(rows)
    ]

    assert first == list(reversed(second))
    assert set(first) == {"train", "val", "test"}


def test_align_unit_loudness_pulls_units_to_shared_target() -> None:
    from tools.boundary.ja.build_galgame_synthetic_timeline import align_unit_loudness

    rng = np.random.default_rng(3)
    target_rms = 0.05
    loud = (np.sin(np.linspace(0.0, 200.0, 8000)) * 0.5).astype(np.float32)
    quiet = (np.sin(np.linspace(0.0, 200.0, 8000)) * 0.02).astype(np.float32)

    loud_aligned, loud_detail = align_unit_loudness(
        loud, rng=rng, target_rms=target_rms, jitter_db=0.0
    )
    quiet_aligned, quiet_detail = align_unit_loudness(
        quiet, rng=rng, target_rms=target_rms, jitter_db=0.0
    )

    loud_rms = float(np.sqrt(np.mean(np.square(loud_aligned))))
    quiet_rms = float(np.sqrt(np.mean(np.square(quiet_aligned))))
    # Both units land on the shared target: no loudness step at the boundary.
    assert abs(loud_rms - target_rms) / target_rms < 0.05
    assert abs(quiet_rms - target_rms) / target_rms < 0.05
    assert loud_detail["scale_db"] < 0 < quiet_detail["scale_db"]

    # An extreme boost is clamped at +/-20 dB instead of amplifying noise.
    barely_audible = (np.sin(np.linspace(0.0, 200.0, 8000)) * 0.001).astype(
        np.float32
    )
    _clamped, clamped_detail = align_unit_loudness(
        barely_audible, rng=rng, target_rms=target_rms, jitter_db=0.0
    )
    assert clamped_detail["scale_db"] == 20.0

    silent = np.zeros(8000, dtype=np.float32)
    unchanged, detail = align_unit_loudness(
        silent, rng=rng, target_rms=target_rms, jitter_db=0.0
    )
    assert detail is None
    assert np.array_equal(unchanged, silent)


def test_empirical_gap_samples_stays_inside_pool_and_bounds() -> None:
    from tools.boundary.ja.build_galgame_synthetic_timeline import empirical_gap_samples

    rng = np.random.default_rng(11)
    pool = [0.2, 0.5, 3.0, 9.0]

    seconds = [
        empirical_gap_samples(
            rng=rng,
            sample_rate=16000,
            duration_pool_s=pool,
            min_s=0.1,
            max_s=4.0,
        )
        / 16000.0
        for _ in range(500)
    ]

    assert min(seconds) >= 0.1
    assert max(seconds) <= 4.0
    # The 9.0s pool entry must be clipped to max_s, not dropped.
    assert any(value > 3.8 for value in seconds)
    with pytest.raises(ValueError):
        empirical_gap_samples(
            rng=rng, sample_rate=16000, duration_pool_s=[], min_s=0.1, max_s=4.0
        )


def test_choose_same_source_row_only_returns_matching_group() -> None:
    from tools.boundary.ja.build_galgame_synthetic_timeline import (
        choose_same_source_row,
        speech_group_key,
    )

    rows = [
        {"audio_id": "a-0", "input": "video-a.mkv"},
        {"audio_id": "a-1", "input": "video-a.mkv"},
        {"audio_id": "b-0", "input": "video-b.mkv"},
    ]
    rng = np.random.default_rng(5)

    row, index = choose_same_source_row(
        source_rows=rows,
        previous_group_key=speech_group_key(rows[0]),
        used_source_indices={0},
        reuse_sources=False,
        rng=rng,
    )
    assert index == 1
    assert row["audio_id"] == "a-1"

    row, index = choose_same_source_row(
        source_rows=rows,
        previous_group_key=speech_group_key(rows[0]),
        used_source_indices={0, 1},
        reuse_sources=False,
        rng=rng,
    )
    assert row is None and index is None


def test_build_gap_prefers_post_speech_flagged_negatives(tmp_path) -> None:
    import soundfile as sf

    from tools.boundary.ja.build_galgame_synthetic_timeline import build_gap

    rng = np.random.default_rng(23)
    for name in ("breath", "music"):
        sf.write(
            str(tmp_path / f"{name}.wav"),
            (np.sin(np.linspace(0.0, 100.0, 16000)) * 0.1).astype(np.float32),
            16000,
        )
    rows = [
        {
            "audio_id": "neg-breath",
            "audio": str(tmp_path / "breath.wav"),
            "background_type": "breathing+moaning",
        },
        {
            "audio_id": "neg-music",
            "audio": str(tmp_path / "music.wav"),
            "background_type": "music",
        },
    ]

    chosen = []
    for _ in range(40):
        _audio, mode, detail = build_gap(
            samples=1600,
            sample_rate=16000,
            index=0,
            rng=rng,
            noise_rms=0.01,
            hum_rms=0.02,
            negative_rows=rows,
            negative_gap_prob=1.0,
            randomize_synthetic_mode=True,
            preferred_flags=["breath", "moan"],
            preferred_flag_prob=1.0,
        )
        assert mode == "real_negative"
        assert detail["preferred_flags_applied"] is True
        chosen.append(detail["audio_id"])

    assert set(chosen) == {"neg-breath"}


def test_background_switch_crossfades_inside_speech_unit(tmp_path) -> None:
    import soundfile as sf

    from boundary.ja import TeacherSegment
    from tools.boundary.ja.build_galgame_synthetic_timeline import mix_background_audio

    rng = np.random.default_rng(31)
    duration_samples = 16000 * 8
    time = np.arange(duration_samples, dtype=np.float32) / 16000.0
    sf.write(str(tmp_path / "bg1.wav"), (0.1 * np.sin(2 * np.pi * 100 * time)).astype(np.float32), 16000)
    sf.write(str(tmp_path / "bg2.wav"), (0.1 * np.sin(2 * np.pi * 900 * time)).astype(np.float32), 16000)
    rows = [
        {"audio_id": "bg1", "audio": str(tmp_path / "bg1.wav")},
        {"audio_id": "bg2", "audio": str(tmp_path / "bg2.wav")},
    ]
    audio = (0.2 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    speech_segments = [TeacherSegment(start=2.0, end=6.0, score=1.0)]

    mixed, detail = mix_background_audio(
        audio,
        background_rows=rows,
        rng=rng,
        snr_db_min=10.0,
        snr_db_max=10.0,
        switch_prob=1.0,
        speech_segments=speech_segments,
        sample_rate=16000,
    )

    assert mixed.shape == audio.shape
    switch = detail["switch"]
    # The bed switch must land INSIDE the speech unit, away from boundaries.
    assert 2.0 < switch["switch_s"] < 6.0
    assert switch["crossfade_s"] > 0.5
