"""Conservative Japanese sentence/clause boundaries, before translation.

Punctuation is source evidence; a silence or a character count alone is not.
The small clause recognizer deliberately leaves unfamiliar constructions intact
instead of pretending to be a Japanese parser. Timing eligibility is checked
separately against measured source words by the subtitle planner.
"""

from __future__ import annotations

import re


_TERMINALS = frozenset("。！？!?")
_COMMAS = frozenset("、，,；;")
_PAIRS = dict(zip("「『（【〈《〔［｛“‘", "」』）】〉》〕］｝”’", strict=True))
_CLOSERS = frozenset(_PAIRS.values())
_DECORATION = _TERMINALS | _COMMAS | _CLOSERS | frozenset("….")

# Clear predicates and completed subordinate clauses. These are permissions,
# not a mandate to split: the whole sentence must also exceed a display target
# and the speaker must pause at the comma. A noun list or topic marker is not
# a clause ("東京、横浜" / "私は、...").
_PREDICATE = r"(?:です|ます|でした|ました|ません(?:でした)?|だった|だ|ない|なかった|たい|たかった|でしょう|だろう|[いうくぐすつぬぶむるた])"
_CLAUSE_END = re.compile(
    r"(?:(?:です|ます|でした|ました|ません(?:でした)?|でしょう)(?:よ|ね|よね|か)?"
    rf"|{_PREDICATE}(?:から|ので|のに|けど|けれど(?:も)?|が|し)"
    r"|(?:たら|なら|れば))$"
)
# The right side can reverse or qualify the apparent predicate on the left.
# Keeping these constructions together matters more than fitting a line.
_DEPENDENT_TAIL = re.compile(
    r"^(?:わけ|訳|はず|筈|つもり|こと|事|もの|ところ|よう|かもしれ|"
    r"という|って(?:こと|わけ|意味)|とは|だけ|しか|ほど|くらい|ぐらい|"
    r"の(?:で|に|は|が|を|か)|は(?:ない|ありません)|"
    r"ならない|なりません|いけない|いけません|だめ|ダメ|駄目|いい)"
)
_QUOTE_CONTINUATION = re.compile(r"^(?:と|って|の|に|を|が|は|も|など|から|まで)")


def source_boundary_hints(text: str) -> dict[int, str]:
    """Return source character offsets backed by sentence/clause syntax.

    Closing quotes travel with their sentence. Punctuation within a quotation
    or parenthesis cannot separate it from the outer sentence, nor can a quoted
    sentence be detached from a following reporting particle.
    """
    result: dict[int, str] = {}
    stack: list[str] = []
    for offset, char in enumerate(text, start=1):
        if char in _PAIRS:
            stack.append(_PAIRS[char])
        elif char in _CLOSERS and stack and char == stack[-1]:
            stack.pop()
        if stack or offset == len(text):
            continue
        if char not in _DECORATION and not char.isspace():
            continue
        right = text[offset:offset + 128].lstrip()
        if not right or right[0] in _DECORATION:
            continue
        left = text[max(0, offset - 128):offset].rstrip()
        unquoted = left.rstrip("".join(_CLOSERS)).rstrip()
        if not unquoted:
            continue
        if unquoted[-1] in _TERMINALS:
            if left[-1] in _CLOSERS and _QUOTE_CONTINUATION.match(right):
                continue
            result[offset] = "sentence_punctuation"
        elif unquoted[-1] in _COMMAS and left == unquoted:
            clause = unquoted[:-1].rstrip()
            if len(clause) >= 4 and _CLAUSE_END.search(clause) and not _DEPENDENT_TAIL.match(right):
                result[offset] = "clause_with_pause"
    return result
