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
_CLOSERS_STR = "".join(_CLOSERS)
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


def _enclosed_depth_deltas(text: str) -> list[int]:
    """Pre-scan matched quote/bracket spans so unclosed openers never blind splits.

    Each matched pair ``(open_idx, close_idx)`` increments depth at
    ``open_idx`` and decrements it at ``close_idx``, masking internal
    punctuation while allowing a boundary immediately after ``close_idx``.
    Stray closers and unclosed openers are ignored in O(N) amortized time.
    """
    deltas = [0] * len(text)
    stack: list[tuple[int, str]] = []
    expected_counts: dict[str, int] = {}
    for index, char in enumerate(text):
        closer = _PAIRS.get(char)
        if closer is not None:
            stack.append((index, closer))
            expected_counts[closer] = expected_counts.get(closer, 0) + 1
            continue
        if char in _CLOSERS and expected_counts.get(char, 0) > 0:
            while stack:
                open_index, expected = stack.pop()
                expected_counts[expected] -= 1
                if expected == char:
                    deltas[open_index] += 1
                    deltas[index] -= 1
                    break
    return deltas


def source_boundary_hints(text: str) -> dict[int, str]:
    """Return source character offsets backed by sentence/clause syntax.

    Closing quotes travel with their sentence. Punctuation within a matched
    quotation or parenthesis cannot separate it from the outer sentence, nor
    can a quoted sentence be detached from a following reporting particle.
    Unclosed opening quotes/brackets are treated as ordinary characters so
    they cannot suppress subsequent sentence boundaries.
    """
    length = len(text)
    if length < 2:
        return {}
    deltas = _enclosed_depth_deltas(text)
    result: dict[int, str] = {}
    depth = 0
    for index, char in enumerate(text):
        depth += deltas[index]
        offset = index + 1
        if depth > 0 or offset == length:
            continue
        if char not in _DECORATION and not char.isspace():
            continue
        # Collapse punctuation + following whitespace run to a single rightmost
        # offset so zero-width spaces never create consecutive duplicate cuts.
        if text[offset].isspace():
            continue
        right = text[offset:offset + 128].lstrip()
        if not right or right[0] in _DECORATION:
            continue
        left = text[max(0, offset - 128):offset].rstrip()
        unquoted = left.rstrip(_CLOSERS_STR).rstrip()
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
