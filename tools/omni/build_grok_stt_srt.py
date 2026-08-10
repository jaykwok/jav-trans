from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from core.config import load_config  # noqa: E402

load_config()

from llm import translator  # noqa: E402
from subtitles.options import SubtitleOptions  # noqa: E402
from subtitles.writer import prepare_srt_blocks, write_bilingual_srt  # noqa: E402


SENTENCE_END = re.compile(r"[。！？!?…]+[」』】）)]*$")


def join_text(parts: list[str]) -> str:
    result = ""
    for raw in parts:
        text = str(raw or "").strip()
        if not text:
            continue
        if (
            result
            and result[-1].isascii()
            and result[-1].isalnum()
            and text[0].isascii()
            and text[0].isalnum()
        ):
            result += " "
        result += text
    return result


def should_split(
    group: list[dict[str, Any]],
    current: dict[str, Any],
    *,
    pause_s: float,
) -> tuple[bool, str]:
    if not group:
        return False, ""
    previous = group[-1]
    if previous["chunk_id"] != current["chunk_id"]:
        return True, "request_chunk_boundary"
    gap = float(current["start_s"]) - float(previous["end_s"])
    if previous.get("speaker") != current.get("speaker") and gap >= 0.0:
        return True, "speaker_change_nonoverlap"
    duration = float(previous["end_s"]) - float(group[0]["start_s"])
    if gap >= pause_s and duration >= 0.45:
        return True, "pause"
    if SENTENCE_END.search(str(previous.get("text") or "")) and (
        gap >= 0.12 or duration >= 3.0
    ):
        return True, "sentence_end"
    return False, ""


def build_cues(
    words: list[dict[str, Any]],
    *,
    pause_s: float = 0.8,
) -> list[dict[str, Any]]:
    cues: list[dict[str, Any]] = []
    group: list[dict[str, Any]] = []
    next_reason = "film_start"

    def flush() -> None:
        nonlocal group
        if not group:
            return
        cues.append(
            {
                "start_s": float(group[0]["start_s"]),
                "end_s": float(group[-1]["end_s"]),
                "text": join_text([str(word["text"]) for word in group]),
                "speaker": group[0].get("speaker"),
                "chunk_id": group[0]["chunk_id"],
                "cut_reason": next_reason,
                "word_count": len(group),
                "words": [
                    {
                        "word": str(word["text"]),
                        "start": float(word["start_s"]),
                        "end": float(word["end_s"]),
                        "speaker": word.get("speaker"),
                        "confidence": word.get("confidence"),
                        "timestamp_kind": "grok_stt_word",
                    }
                    for word in group
                ],
            }
        )
        group = []

    for word in words:
        split, reason = should_split(group, word, pause_s=pause_s)
        if split:
            flush()
            next_reason = reason
        group.append(word)
    flush()
    return [cue for cue in cues if cue["text"] and cue["end_s"] > cue["start_s"]]


def _source_blocks(cues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "start": float(cue["start_s"]),
            "end": float(cue["end_s"]),
            "acoustic_start": float(cue["start_s"]),
            "acoustic_end": float(cue["end_s"]),
            "ja_text": str(cue["text"]),
            "zh_text": str(cue["text"]),
            "words": list(cue["words"]),
            "source_segment_ids": [index],
            "grok_speaker": cue.get("speaker"),
            "grok_chunk_id": cue.get("chunk_id"),
            "grok_cut_reason": cue.get("cut_reason"),
        }
        for index, cue in enumerate(cues)
    ]


def _translation_cues(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for cue_id, block in enumerate(blocks):
        item = dict(block)
        source = str(item.get("ja_text") or item.get("text") or "").strip()
        item["cue_id"] = cue_id
        item["text"] = source
        item["ja_text"] = source
        item["zh_text"] = source
        item["words"] = list(item.get("words") or [])
        item["continues_from_previous"] = bool(item.get("continues_from_previous"))
        item["continues_into_next"] = bool(item.get("continues_into_next"))
        item.setdefault("source_segment_ids", [cue_id])
        normalized.append(item)
    return normalized


def _drop_unrenderable_tight_cues(
    blocks: list[dict[str, Any]],
    *,
    minimum_s: float = 0.05,
) -> tuple[list[dict[str, Any]], int]:
    """Drop cues whose next onset leaves less than the SRT writer's 50ms floor."""
    kept: list[dict[str, Any]] = []
    dropped = 0
    for index, block in enumerate(blocks):
        if index + 1 < len(blocks):
            available = float(blocks[index + 1]["start"]) - float(block["start"])
            if available + 1e-9 < minimum_s:
                dropped += 1
                continue
        kept.append(block)
    return kept, dropped


def _write_blocks_json(path: Path, blocks: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps({"blocks": blocks}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run(
    *,
    words_path: Path,
    output_dir: Path,
    pause_s: float,
    translate: bool,
    max_workers: int,
    target_lang: str,
    cache_dir: Path | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_film: dict[str, list[dict[str, Any]]] = {}
    for line in words_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        word = json.loads(line)
        by_film.setdefault(str(word["film_id"]), []).append(word)

    options = SubtitleOptions.from_env()
    summary: dict[str, Any] = {}
    for film_id, words in sorted(by_film.items()):
        words.sort(key=lambda word: (float(word["start_s"]), float(word["end_s"])))
        source_cues = build_cues(words, pause_s=pause_s)
        planned = prepare_srt_blocks(
            _source_blocks(source_cues),
            options=options,
            mode="bilingual",
        )
        planned, unrenderable_cue_count = _drop_unrenderable_tight_cues(planned)
        cues = _translation_cues(planned)
        translation_timings: list[dict[str, Any]] = []
        retry_events: list[dict[str, Any]] = []
        if translate:
            effective_cache_dir = cache_dir or output_dir / "translation-cache"
            effective_cache_dir.mkdir(parents=True, exist_ok=True)
            translations, translation_timings, retry_events = translator.translate_segments(
                cues,
                global_context=translator.generate_global_context(cues),
                max_workers=max_workers,
                cache_path=str(effective_cache_dir / f"{film_id}.jsonl"),
                target_lang=target_lang,
                glossary=os.getenv("TRANSLATION_GLOSSARY", ""),
                reasoning_effort=os.getenv("LLM_REASONING_EFFORT", "none"),
                api_format=os.getenv("LLM_API_FORMAT", "chat"),
            )
            if len(translations) != len(cues):
                raise RuntimeError(
                    f"{film_id}: translated {len(translations)} of {len(cues)} cues"
                )
            for cue, translated in zip(cues, translations):
                cue["zh_text"] = str(translated or "").strip()
            output = output_dir / f"{film_id}.Grok-STT-diarized.zh-ja.srt"
        else:
            for cue in cues:
                cue["zh_text"] = ""
            output = output_dir / f"{film_id}.Grok-STT-diarized.ja.srt"

        written = write_bilingual_srt(cues, str(output), options=options)
        _write_blocks_json(output.with_suffix(".json"), written)
        reasons: dict[str, int] = {}
        for cue in source_cues:
            reason = str(cue["cut_reason"])
            reasons[reason] = reasons.get(reason, 0) + 1
        summary[film_id] = {
            "output": str(output),
            "word_count": len(words),
            "source_utterance_count": len(source_cues),
            "subtitle_layer_cue_count": len(cues),
            "unrenderable_tight_cue_count": unrenderable_cue_count,
            "written_cue_count": len(written),
            "source_utterance_start_reasons": reasons,
            "first_start_s": written[0]["start"] if written else None,
            "last_end_s": written[-1]["end"] if written else None,
            "translated": translate,
            "target_lang": target_lang if translate else None,
            "translation_batch_count": len(translation_timings),
            "translation_retry_event_count": len(retry_events),
            "subtitle_options": options.signature(),
        }

    (output_dir / "grok_srt_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build production-layout SRT files from Grok STT word JSONL."
    )
    parser.add_argument("--words", required=True, help="Combined Grok word JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pause-s", type=float, default=0.8)
    parser.add_argument("--translate", action="store_true")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--target-lang", default=os.getenv("TARGET_LANG", "简体中文"))
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    if args.pause_s < 0 or args.max_workers <= 0:
        parser.error("pause-s must be non-negative and max-workers must be positive")
    return args


def main() -> None:
    args = parse_args()
    summary = run(
        words_path=Path(args.words).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        pause_s=float(args.pause_s),
        translate=bool(args.translate),
        max_workers=int(args.max_workers),
        target_lang=str(args.target_lang).strip() or "简体中文",
        cache_dir=(
            Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
