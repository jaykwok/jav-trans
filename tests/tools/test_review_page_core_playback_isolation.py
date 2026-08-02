"""Only one clip may be audible at a time on an audit page.

Two clips playing at once is not heard as two clips. It is heard as the new one
having started partway in, because the old one is already partway in - and an
auditor who thinks a clip began at two seconds judges what they heard, which is
a mixture of two different spans. That corrupts the verdict silently, so the
isolation is pinned here rather than left to the player's internal bookkeeping.

`stop()` cannot do this job alone. It tracks the single element the span player
is driving, and knows nothing about a card the auditor started from its own
native `<audio controls>`. The rule therefore hangs off the `play` event in the
capture phase, where it applies no matter what started the playback.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.audits.review_page_core import (  # noqa: E402
    AUDIO_SPAN_PLAYER_JS,
    AuditReviewPageSpec,
    render_audit_review_page,
)


def _body(name: str) -> str:
    """The source of one top-level function in the player."""
    start = AUDIO_SPAN_PLAYER_JS.index(f"function {name}(")
    depth, index = 0, AUDIO_SPAN_PLAYER_JS.index("{", start)
    for position in range(index, len(AUDIO_SPAN_PLAYER_JS)):
        character = AUDIO_SPAN_PLAYER_JS[position]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return AUDIO_SPAN_PLAYER_JS[start : position + 1]
    raise AssertionError(f"unbalanced braces reading {name}")


def test_starting_a_clip_silences_every_other_card() -> None:
    body = _body("pauseOtherAudio")
    assert "querySelectorAll('audio')" in body
    assert "other===keep" in body, "the clip being started must not pause itself"
    assert "other.pause()" in body


def test_a_silenced_card_is_rewound_rather_than_left_mid_clip() -> None:
    """Otherwise returning to it resumes where the auditor abandoned it."""
    assert "other.currentTime=0" in _body("pauseOtherAudio")


def test_the_span_player_silences_the_others_before_it_starts() -> None:
    body = _body("play")
    assert "pauseOtherAudio(audio)" in body
    assert body.index("pauseOtherAudio(audio)") < body.index("waitForMetadata")


def test_native_controls_are_covered_by_a_capture_phase_listener() -> None:
    """The span player never sees playback the auditor starts from the element
    itself, so the rule cannot live inside `play()` alone."""
    listener = AUDIO_SPAN_PLAYER_JS[
        AUDIO_SPAN_PLAYER_JS.index("document.addEventListener('play'") :
    ]
    listener = listener[: listener.index("async function play")]
    assert listener.rstrip().endswith("},true);"), "must run in the capture phase"
    assert "pauseOtherAudio(target)" in listener
    assert "activeAudio&&activeAudio!==target" in listener, (
        "playback of the element the span player is already driving must not "
        "cancel that span player"
    )


def test_a_clip_parked_at_its_end_rewinds_before_replaying() -> None:
    """`stop(safeEnd)` leaves the element at the cut's end, which mp3 encoder
    padding puts just short of `duration`. Pressing play there runs out only the
    padding, which sounds like a clip that refuses to play."""
    listener = AUDIO_SPAN_PLAYER_JS[
        AUDIO_SPAN_PLAYER_JS.index("document.addEventListener('play'") :
    ]
    listener = listener[: listener.index("async function play")]
    assert "target.duration-.05" in listener
    assert "target.currentTime=0" in listener


def test_the_rewind_does_not_disturb_a_span_that_ends_mid_file() -> None:
    """A span page stops partway through a long file on purpose; only playback
    sitting within a hair of `duration` may be rewound."""
    listener = AUDIO_SPAN_PLAYER_JS[
        AUDIO_SPAN_PLAYER_JS.index("document.addEventListener('play'") :
    ]
    listener = listener[: listener.index("async function play")]
    guard = re.search(r"if\((.*?)\)\{\s*target\.currentTime=0;", listener, re.S)
    assert guard is not None
    assert "target.currentTime>=target.duration-.05" in guard.group(1)
    assert "Number.isFinite(target.duration)" in guard.group(1)


def test_every_rendered_page_carries_the_isolation() -> None:
    page = render_audit_review_page(
        AuditReviewPageSpec(
            title="t", intro_html="", body_html="", adapter_css="", adapter_js=""
        )
    )
    assert "pauseOtherAudio" in page
    assert "document.addEventListener('play'" in page
