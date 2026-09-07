from __future__ import annotations

import json

from llm.context import SourceContext, split_batches
from llm import cache, engine, repair
from llm.profiles.json_v3 import JsonProfile
from llm.profiles.base import ProfileContext


def _cues(count=8):
    return [{"start": float(i), "end": i + 0.8, "text": f"台詞{i}"} for i in range(count)]


def _join(cues, start, end):
    for index in range(start, end):
        cues[index]["continues_into_next"] = True
        cues[index + 1]["continues_from_previous"] = True


def test_batch_boundary_does_not_separate_a_confirmed_unit():
    cues = _cues()
    _join(cues, 2, 4)
    batches = split_batches(cues, 4)
    assert [len(batch) for batch in batches] == [2, 4, 2]
    assert [cue for batch in batches for cue in batch] == cues


def test_oversized_units_and_long_single_cues_still_make_progress():
    cues = _cues()
    _join(cues, 0, 7)
    cues[0]["text"] = "長" * 9000
    batches = split_batches(cues, 3, max_source_chars=20)
    assert batches and all(0 < len(batch) <= 3 for batch in batches)
    assert [id(cue) for batch in batches for cue in batch] == [id(cue) for cue in cues]
    assert SourceContext.build(cues).units == (tuple(range(8)),)


def test_focus_includes_a_whole_unit_beyond_the_neighbour_radius():
    cues = _cues(12)
    _join(cues, 3, 8)
    focus = SourceContext.build(cues).focus([6], radius=1)
    assert focus["semantic_units"][0]["ids"] == list(range(3, 9))
    assert {row["id"] for row in focus["items"]} == set(range(3, 9))
    assert [row["id"] for row in focus["items"] if row["role"] == "translate"] == [6]


def test_focus_never_returns_the_film_opening_instead_of_tail_neighbours():
    cues = _cues(100)
    context = SourceContext.build(cues)
    messages = JsonProfile().build_messages(
        cues[90:92], ids=[90, 91],
        ctx=ProfileContext(global_context="only the opening fits", total_count=100, source_context=context),
    )
    task = messages[-1]["content"]
    assert "台詞89" in task and "台詞92" in task
    assert "台詞0" not in task


def test_context_budget_does_not_truncate_requested_source():
    cues = _cues()
    cues[4]["text"] = "長" * 5000
    focus = SourceContext.build(cues).focus([4], context_chars=0)
    assert len(focus["items"]) == 1
    assert focus["items"][0]["ja"] == cues[4]["text"]


def test_repair_keeps_continuation_and_complete_unit_information():
    cues = _cues()
    _join(cues, 1, 6)
    items = repair._build_repair_context_items(cues, ["译文"] * 8, [4], {4: ["source_echo"]})
    by_id = {item["id"]: item for item in items}
    assert by_id[4]["cont_prev"] and by_id[4]["cont_next"]
    assert {1, 2, 3, 4, 5, 6}.issubset(by_id)
    assert by_id[3]["role"] == "context_only"


def test_repeated_source_does_not_require_identical_translation():
    cues = [{"text": "大丈夫"}] * 3
    ids, _ = repair._select_translation_repair_ids(cues, ["没事", "没事", "不用了"])
    assert ids == []


def test_memory_keys_distinguish_occurrence_and_changed_context():
    kwargs = dict(prompt_version="v4", model_name="model", context_signature="scene-a")
    first = cache._translation_memory_key("大丈夫", occurrence_id=1, **kwargs)
    second = cache._translation_memory_key("大丈夫", occurrence_id=2, **kwargs)
    assert first != second
    kwargs["context_signature"] = "scene-b"
    assert first != cache._translation_memory_key("大丈夫", occurrence_id=1, **kwargs)


def test_memory_ignores_time_offsets_but_tracks_context_boundaries():
    cues = _cues(3)
    original = SourceContext.build(cues).signature
    shifted = [{**cue, "start": cue["start"] + 100, "end": cue["end"] + 100} for cue in cues]
    assert SourceContext.build(shifted).signature == original
    shifted[-1]["start"] += 20
    shifted[-1]["end"] += 20
    assert SourceContext.build(shifted).signature != original


def test_source_outside_the_batch_and_sampling_changes_invalidate_reuse(monkeypatch):
    cues = _cues()
    profile = JsonProfile()
    scope = engine.cache_scope(SourceContext.build(cues), profile, "model")
    cues[-1]["text"] = "申し出を断る場面"
    assert scope != engine.cache_scope(SourceContext.build(cues), profile, "model")
    from llm import settings

    scope = engine.cache_scope(SourceContext.build(cues), profile, "model")
    monkeypatch.setattr(settings, "TRANSLATION_TEMPERATURE", 0.123)
    assert scope != engine.cache_scope(SourceContext.build(cues), profile, "model")


def test_observed_renderings_are_not_counted_as_user_glossary(tmp_path):
    from pipeline.quality import collect_glossary_pairs

    (tmp_path / "translation_global_glossary.json").write_text(
        json.dumps({"terms": [{"ja": "大丈夫", "zh": "没事"}]}), encoding="utf-8"
    )
    assert collect_glossary_pairs(str(tmp_path), "sample-a", glossary="東京-东京", console=None) == [("東京", "东京")]
