from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUBTITLE_QC = ROOT / "subtitle_qc"
if str(SUBTITLE_QC) not in sys.path:
    sys.path.insert(0, str(SUBTITLE_QC))

import generate  # noqa: E402
import review_content  # noqa: E402


def test_review_content_candidate_label_uses_positive_allowlist() -> None:
    assert review_content.candidate_label_from_path("sample-video-b", Path("sample-video-b.fusion_lite.srt")) == "fusion_lite"
    assert (
        review_content.candidate_label_from_path("sample-video-b", Path("sample-video-b.whisper_ja_1_5b_fusion_lite.srt"))
        == "fusion_lite"
    )
    assert (
        review_content.candidate_label_from_path("sample-video-b", Path("sample-video-b.whisperseg-adaptive.bilingual.json"))
        == "whisperseg_adaptive"
    )

    with pytest.raises(SystemExit, match="fusion_lite_sigmoid"):
        review_content.candidate_label_from_path("sample-video-b", Path("sample-video-b.fusion_lite_sigmoid.srt"))
    with pytest.raises(SystemExit, match="silero"):
        review_content.candidate_label_from_path("sample-video-b", Path("sample-video-b.silero.srt"))


def test_review_content_auto_discovery_errors_on_unknown_vad_artifact(tmp_path: Path) -> None:
    video = tmp_path / "sample-video-b.mp4"
    artifact_dir = tmp_path / "sample-video-b"
    artifact_dir.mkdir()
    (artifact_dir / "sample-video-b.fusion_lite.srt").write_text("", encoding="utf-8")
    (artifact_dir / "sample-video-b.fusion_lite_sigmoid.srt").write_text("", encoding="utf-8")

    with pytest.raises(SystemExit, match="fusion_lite_sigmoid"):
        review_content.discover_candidates(video, all_candidates=False, modes=None)


def test_review_content_explicit_candidate_allows_trial_labels(tmp_path: Path) -> None:
    path = tmp_path / "sample-video-b.lite_gate08.srt"
    path.write_text("", encoding="utf-8")

    spec = review_content.parse_candidate_spec(f"lite_gate08={path}", "sample-video-b", set())

    assert spec.label == "lite_gate08"
    assert spec.path == path.resolve()


def test_generate_discover_modes_normalizes_requested_modes() -> None:
    assert generate.discover_modes("sample-video-b", ["fusion-lite", "whisperseg-adaptive", "fusion_lite"]) == [
        "fusion_lite",
        "whisperseg_adaptive",
    ]

    with pytest.raises(SystemExit, match="fusion_lite_sigmoid"):
        generate.discover_modes("sample-video-b", ["fusion_lite_sigmoid"])


def test_generate_discover_modes_uses_allowlist_for_prefixed_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video_dir = tmp_path / "video" / "sample-video-b"
    video_dir.mkdir(parents=True)
    (video_dir / "sample-video-b.whisper_ja_1_5b_fusion_lite.bilingual.json").write_text("{}", encoding="utf-8")
    (video_dir / "sample-video-b.whisper_ja_1_5b_whisperseg_adaptive.bilingual.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(generate, "PROJECT_ROOT", tmp_path)

    assert generate.discover_modes("sample-video-b", None) == [
        "whisper_ja_1_5b_whisperseg_adaptive",
        "whisper_ja_1_5b_fusion_lite",
    ]


def test_generate_discover_modes_errors_on_unknown_auto_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video_dir = tmp_path / "video" / "sample-video-b"
    video_dir.mkdir(parents=True)
    (video_dir / "sample-video-b.fusion_lite_sigmoid.bilingual.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(generate, "PROJECT_ROOT", tmp_path)

    with pytest.raises(SystemExit, match="fusion_lite_sigmoid"):
        generate.discover_modes("sample-video-b", None)


def test_generate_resolve_base_mode_accepts_canonical_base_for_prefixed_modes() -> None:
    modes = ["whisper_ja_1_5b_whisperseg_adaptive", "whisper_ja_1_5b_fusion_lite"]

    assert generate.resolve_base_mode("whisperseg_adaptive", modes) == "whisper_ja_1_5b_whisperseg_adaptive"
    assert generate.resolve_base_mode("whisperseg-adaptive", modes) == "whisper_ja_1_5b_whisperseg_adaptive"
    assert generate.resolve_base_mode("fusion_lite", modes) == "whisper_ja_1_5b_fusion_lite"

    with pytest.raises(SystemExit, match="base mode"):
        generate.resolve_base_mode("silero", modes)
