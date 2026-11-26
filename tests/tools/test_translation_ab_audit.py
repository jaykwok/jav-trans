"""The translation A/B has to stay blind, aligned, and honest about ties."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits import evaluate_translation_ab_audit as evaluate  # noqa: E402
from tools.audits import generate_translation_ab_audit_html as generate  # noqa: E402


def _bilingual(path: Path, cues: list[tuple[float, float, str, str]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "blocks": [
                    {"start": start, "end": end, "ja_text": ja, "zh_text": zh}
                    for start, end, ja, zh in cues
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


BASE = [
    (0.0, 2.0, "おはよう", "早上好"),
    (2.5, 4.0, "気持ちいい", "好舒服"),
    (5.0, 7.5, "そいついきまってるな", "那家伙来劲了"),
    (8.0, 9.0, "うん", "嗯"),
]


def test_arms_must_be_the_same_cues(tmp_path):
    """Two differently-decoded runs would compare the ASR, not the translator."""
    left = generate.load_cues(_bilingual(tmp_path / "a.json", BASE))
    shifted = list(BASE)
    shifted[2] = (5.0, 7.5, "そいつ行きまってるな", "那家伙来劲了吧")
    right = generate.load_cues(_bilingual(tmp_path / "b.json", shifted))

    with pytest.raises(SystemExit) as failure:
        generate.require_same_cue_set({"none": left, "medium": right})
    message = str(failure.value)
    assert "cue 2" in message
    assert "Only the Chinese may differ" in message

    dropped = generate.load_cues(_bilingual(tmp_path / "c.json", BASE[:3]))
    with pytest.raises(SystemExit) as shortfall:
        generate.require_same_cue_set({"none": left, "medium": dropped})
    assert "cue count" in str(shortfall.value)


def test_only_cues_whose_translations_differ_are_eligible(tmp_path):
    left = generate.load_cues(_bilingual(tmp_path / "a.json", BASE))
    variant = list(BASE)
    variant[1] = (2.5, 4.0, "気持ちいい", "舒服死了")
    variant[3] = (8.0, 9.0, "うん", "嗯嗯")
    right = generate.load_cues(_bilingual(tmp_path / "b.json", variant))

    rows = generate.eligible_rows({"none": left, "medium": right}, min_ja_chars=4)
    # cue 1 differs and is long enough; cue 3 differs but 「うん」 is too short;
    # cues 0 and 2 are identical in both arms and carry no preference.
    assert [row["index"] for row in rows] == [1]
    assert rows[0]["none"] == "好舒服"
    assert rows[0]["medium"] == "舒服死了"


def test_leading_arm_is_balanced():
    flags = generate.balanced_first_arm(10, ("none", "medium"), seed=7)
    assert sorted(flags) == ["medium"] * 5 + ["none"] * 5
    odd = generate.balanced_first_arm(7, ("none", "medium"), seed=7)
    assert abs(odd.count("none") - odd.count("medium")) == 1


def test_rendered_page_carries_no_arm_identity():
    """The blind is structural: a row may only carry what a card displays.

    Searching the document for the arm names cannot work - an arm named `none`
    matches the core CSS, and one named `flash` could occur inside a subtitle.
    """
    rows = [
        {
            "row_id": "translation-ab-001",
            "span": "00:02.500–00:04.000",
            "ja": "気持ちいい",
            "clip_src": "media/translation-ab-001.mp3",
            "arm_1_text": "好舒服",
            "arm_2_text": "舒服死了",
        }
    ]
    page = generate.render_page(rows, title="翻译配置 · 匿名人工 A/B")
    assert "好舒服" in page and "舒服死了" in page
    generate.assert_page_is_blind(rows, page)

    leaked = [{**rows[0], "arm_1": "medium"}]
    with pytest.raises(AssertionError, match="arm_1"):
        generate.assert_page_is_blind(leaked, generate.render_page(rows, title="x"))


def test_materialize_replaces_existing_clip_when_output_dir_is_reused(tmp_path, monkeypatch):
    output_dir = tmp_path / "audit"
    clip = output_dir / "media" / "translation-ab-001.mp3"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"OLD CLIP")
    calls: list[bool] = []

    def fake_slice_audio_clip(*, output_path, force, **_kwargs):
        calls.append(force)
        output_path.write_bytes(b"NEW CLIP")

    monkeypatch.setattr(generate, "slice_audio_clip", fake_slice_audio_clip)
    rows = [
        {
            "index": 0,
            "start_s": 1.0,
            "end_s": 2.0,
            "ja": "これは十分な長さ",
            "none": "甲",
            "medium": "乙",
        }
    ]

    generate.materialize(
        rows,
        arm_names=("none", "medium"),
        audio=tmp_path / "source.wav",
        output_dir=output_dir,
        pad_s=0.35,
        seed=1,
    )

    assert calls == [True]
    assert clip.read_bytes() == b"NEW CLIP"


def _answers() -> list[dict]:
    return [
        {"row_id": "r1", "cue_index": 1, "arm_1": "none", "arm_2": "medium", "ja": "a"},
        {"row_id": "r2", "cue_index": 2, "arm_1": "medium", "arm_2": "none", "ja": "b"},
        {"row_id": "r3", "cue_index": 3, "arm_1": "none", "arm_2": "medium", "ja": "c"},
        {"row_id": "r4", "cue_index": 4, "arm_1": "medium", "arm_2": "none", "ja": "d"},
    ]


def test_unblinding_follows_the_card_order_not_the_button():
    """`arm_1_better` means different arms on different cards; that is the point."""
    verdicts = [
        {"row_id": "r1", "verdict": "arm_1_better"},   # none
        {"row_id": "r2", "verdict": "arm_1_better"},   # medium
        {"row_id": "r3", "verdict": "arm_2_better"},   # medium
        {"row_id": "r4", "verdict": "arm_2_better"},   # none
    ]
    result = evaluate.build(_answers(), verdicts)
    assert result["wins"] == {"none": 2, "medium": 2}
    assert result["decisive"] == 4
    assert result["sign_test_p"] == 1.0


def test_ties_are_not_half_a_win_and_unreviewed_is_reported():
    verdicts = [
        {"row_id": "r1", "verdict": "arm_1_better"},   # none
        {"row_id": "r2", "verdict": "equivalent_good"},
        {"row_id": "r3", "verdict": "equivalent_bad"},
    ]
    result = evaluate.build(_answers(), verdicts)
    assert result["wins"] == {"none": 1, "medium": 0}
    assert result["decisive"] == 1
    assert result["equivalent_good"] == 1 and result["equivalent_bad"] == 1
    assert result["reviewed"] == 3 and result["unreviewed"] == 1
    assert result["win_share_decisive"] == {"none": 1.0, "medium": 0.0}
    # One decisive card cannot claim anything: the interval spans almost everything.
    low, high = result["win_share_ci95"]["none"]
    assert low < 0.3 and high == 1.0


def test_a_verdict_for_an_unknown_card_is_refused():
    with pytest.raises(ValueError, match="unknown row"):
        evaluate.build(_answers(), [{"row_id": "nope", "verdict": "arm_1_better"}])


def test_an_unknown_verdict_is_refused():
    with pytest.raises(ValueError, match="unknown verdict"):
        evaluate.build(_answers(), [{"row_id": "r1", "verdict": "arm_1_beter"}])


def test_wilson_interval_matches_a_known_value():
    assert evaluate.wilson_interval(0, 0) is None
    low, high = evaluate.wilson_interval(30, 40)
    assert 0.59 < low < 0.60
    assert 0.85 < high < 0.86
