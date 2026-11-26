from __future__ import annotations

import json

from tools.omni import build_grok_stt_srt as grok_srt


def _word(
    text: str,
    start: float,
    end: float,
    *,
    speaker: int = 0,
    chunk_id: str = "film-0000",
) -> dict:
    return {
        "film_id": "film",
        "chunk_id": chunk_id,
        "text": text,
        "start_s": start,
        "end_s": end,
        "speaker": speaker,
        "confidence": 0.9,
    }


def test_nonoverlap_speaker_change_cuts_but_overlap_does_not():
    nonoverlap = grok_srt.build_cues(
        [_word("前", 0.0, 0.4), _word("後", 0.5, 0.9, speaker=1)]
    )
    overlap = grok_srt.build_cues(
        [_word("前", 0.0, 0.6), _word("後", 0.5, 0.9, speaker=1)]
    )

    assert [cue["text"] for cue in nonoverlap] == ["前", "後"]
    assert nonoverlap[1]["cut_reason"] == "speaker_change_nonoverlap"
    assert [cue["text"] for cue in overlap] == ["前後"]


def test_run_uses_subtitle_layer_and_writes_japanese_srt(tmp_path):
    words_path = tmp_path / "words.jsonl"
    words = [
        _word("前", 0.0, 0.4),
        _word("半", 0.5, 0.9),
        _word("後", 1.8, 2.2),
        _word("半", 2.3, 2.7),
    ]
    words_path.write_text(
        "\n".join(json.dumps(word, ensure_ascii=False) for word in words),
        encoding="utf-8",
    )

    summary = grok_srt.run(
        words_path=words_path,
        output_dir=tmp_path / "out",
        pause_s=0.8,
        translate=False,
        max_workers=1,
        target_lang="简体中文",
        cache_dir=None,
    )

    output = tmp_path / "out" / "film.Grok-STT-diarized.ja.srt"
    assert output.exists()
    assert "前半" in output.read_text(encoding="utf-8-sig")
    assert summary["film"]["source_utterance_count"] == 2
    assert summary["film"]["subtitle_layer_cue_count"] == 2
    sidecar = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["blocks"][0]["words"][0]["timestamp_kind"] == "grok_stt_word"


def test_unrenderable_tight_cue_is_dropped_before_writer_can_overlap():
    blocks = [
        {"start": 1.0, "end": 1.02, "ja_text": "い"},
        {"start": 1.04, "end": 2.0, "ja_text": "つも"},
    ]

    kept, dropped = grok_srt._drop_unrenderable_tight_cues(blocks)

    assert dropped == 1
    assert [block["ja_text"] for block in kept] == ["つも"]
