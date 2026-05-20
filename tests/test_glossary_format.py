from llm.glossary import normalize_glossary_text
from pipeline.quality import parse_glossary_pairs_from_text


def test_parse_glossary_pairs_uses_dash_separator():
    assert parse_glossary_pairs_from_text("ちんぽ-肉棒, チンポ-肉棒\nねこ-猫") == [
        ("ちんぽ", "肉棒"),
        ("チンポ", "肉棒"),
        ("ねこ", "猫"),
    ]


def test_parse_glossary_pairs_ignores_arrow_separator():
    assert parse_glossary_pairs_from_text("ちんぽ→肉棒, チンポ->肉棒") == []


def test_normalize_glossary_text_strips_pair_edges():
    raw = " ちんぽ - 肉棒 \n\n チンポ- 肉棒  , ねこ -猫 "

    assert normalize_glossary_text(raw) == "ちんぽ-肉棒\nチンポ-肉棒\nねこ-猫"
