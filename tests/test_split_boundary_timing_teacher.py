import pytest

from tools.audits import generate_split_boundary_timing_audit_html as audit
from tools.boundary.ja import label_split_boundary_timing_with_omni as timing


def test_timing_teacher_selects_most_left_shifted_boundaries(tmp_path) -> None:
    path = tmp_path / "refined.jsonl"
    path.write_text(
        '{"island_id":"a","window_id":"w","duration_s":5,"cuts":[{"time_s":1,"teacher_time_s":1.1,"projected_candidate_time_s":1.05,"cut_refiner_delta_s":-0.2}]}\n'
        '{"island_id":"b","window_id":"w","duration_s":5,"cuts":[{"time_s":2,"teacher_time_s":2.1,"projected_candidate_time_s":2.05,"cut_refiner_delta_s":0.1}]}\n',
        encoding="utf-8",
    )

    selected = timing.select_boundaries(path, 1)

    assert selected[0]["island_id"] == "a"
    assert selected[0]["refiner_delta_s"] == -0.2


def test_timing_teacher_requires_safe_order() -> None:
    selected = {"boundary_id": "b", "duration_s": 5.0}
    timing._validate(
        {"boundary_id": "b", "status": "ok", "left_speech_end_s": 1.0, "safe_cut_time_s": 1.2, "right_speech_start_s": 1.4, "confidence": 0.9},
        selected,
    )
    with pytest.raises(ValueError, match="left_end"):
        timing._validate(
            {"boundary_id": "b", "status": "ok", "left_speech_end_s": 1.4, "safe_cut_time_s": 1.2, "right_speech_start_s": 1.0, "confidence": 0.9},
            selected,
        )


def test_timing_audit_has_four_time_references_and_two_players(tmp_path, monkeypatch) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        '{"boundary_id":"i#b001","island_id":"i","duration_s":5,"coarse_time_s":2,"projected_time_s":2.1,"refined_time_s":1.9,"left_speech_end_s":2.2,"safe_cut_time_s":2.3,"right_speech_start_s":2.4,"confidence":0.9,"flags":[],"reason":"ok"}\n',
        encoding="utf-8",
    )
    semantic = tmp_path / "semantic.jsonl"
    semantic.write_text(
        '{"island_id":"i","cuts":[{"time_s":1.0},{"time_s":2.0},{"time_s":4.0}]}\n',
        encoding="utf-8",
    )
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "i.wav").write_bytes(b"wav")
    monkeypatch.setattr(
        audit,
        "_segmented",
        lambda _source, output, _cut, _duration: output.write_bytes(b"seg"),
    )
    monkeypatch.setattr(audit, "update_audit_entrypoints", lambda **_kwargs: None)

    audit.build_audit(
        labels=labels,
        semantic_labels=semantic,
        request_audio_dir=audio_dir,
        output_dir=tmp_path / "audit",
    )
    page = (tmp_path / "audit" / "index.html").read_text(encoding="utf-8")

    assert "只显示当前最终切点集合" in page
    assert "当前全部切点" in page
    assert "本卡目标 safe cut" in page
    assert "Omni semantic" not in page
    assert "Active Refiner" not in page
    assert page.count("<audio") == 2
    assert "当前切点总数" in page
