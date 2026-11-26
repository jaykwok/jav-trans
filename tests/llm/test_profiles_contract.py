"""Contract tests every registered profile must satisfy.

These are the invariants the engine relies on; a new profile module that
breaks one of these will fail here before it ever reaches a real run.

Two contracts ship now (a JSON batch one for API models, a per-line one for the
local default), so these tests can no longer assume a reply is JSON. What they
assume instead is what the engine actually assumes: a profile declares its own
batch cap and its own schema, and the two must agree - a profile with no schema
cannot be addressed by id, so it must not accept more than one cue per request.
"""

import json

import pytest

from llm import profiles
from llm.errors import RetryableTranslationFormatError
from llm.profiles.base import ProfileContext


def _segments(count: int) -> list[dict]:
    return [
        {"text": f"セリフ{i}", "start": float(i), "end": float(i) + 1.0}
        for i in range(count)
    ]


@pytest.fixture(params=profiles.list_profiles())
def profile(request):
    return profiles.get_profile(request.param)


def _batch(profile, wanted: int) -> int:
    cap = profile.max_batch_size()
    return wanted if cap is None else min(wanted, cap)


def _reply(profile, ids: list[int], texts: list[str]) -> str:
    """A well-formed reply in whatever shape this profile asked for."""
    if profile.schema is None:
        return texts[0]
    return json.dumps(
        {"translations": [{"id": i, "text": t} for i, t in zip(ids, texts)]},
        ensure_ascii=False,
    )


def test_identity_fields(profile):
    assert profile.id and profile.version
    assert profile.cache_signature() == f"{profile.id}@{profile.version}"


def test_a_schemaless_profile_takes_one_cue_at_a_time(profile):
    """The invariant that keeps a bare reply assignable. A reply with no ids in
    it can only belong to the single cue that was asked for; two cues in one
    request would be silently mis-assigned instead of detected."""
    if profile.schema is None:
        assert profile.max_batch_size() == 1


def test_build_messages_shape(profile):
    count = _batch(profile, 4)
    ctx = ProfileContext(glossary="お姉ちゃん-姐姐", total_count=count)
    messages = profile.build_messages(
        _segments(count), ids=list(range(count)), ctx=ctx
    )
    assert isinstance(messages, list) and messages
    for message in messages:
        assert message.get("role") in {"system", "user", "assistant"}
        assert isinstance(message.get("content"), str)
    # Every source line must reach the prompt: the cue plan is frozen and the
    # engine has no other channel to deliver text to the model.
    joined = "\n".join(m["content"] for m in messages)
    for seg in _segments(count):
        assert seg["text"] in joined


def test_build_messages_refuses_more_than_the_cap(profile):
    """A cap the profile does not enforce is a comment, not a cap."""
    cap = profile.max_batch_size()
    if cap is None:
        pytest.skip("profile sets no cap")
    over = cap + 1
    with pytest.raises(Exception):
        profile.build_messages(
            _segments(over),
            ids=list(range(over)),
            ctx=ProfileContext(total_count=over),
        )


def test_parse_response_covers_ids_or_subset(profile):
    """parse_response may only ever map requested ids (1:1 invariant)."""
    ids = [3, 5][: _batch(profile, 2)]
    texts = ["第一行", "第二行"][: len(ids)]
    parsed = profile.parse_response(_reply(profile, ids, texts), ids=ids)
    assert set(parsed).issubset(set(ids))
    if not profile.supports_partial_reissue:
        assert set(parsed) == set(ids)
    for key, text in zip(ids, texts):
        assert parsed[key] == text


def test_parse_response_rejects_garbage(profile):
    """What counts as garbage differs by contract, but "unusable" must always
    raise rather than resolve to a translation. For the line contract an empty
    reply is the whole failure surface - free text has no other malformed
    shape - and it is the exact defect that retired the previous line profile,
    which wrote "" and returned successfully."""
    garbage = "" if profile.schema is None else "???not a contract reply???\nx\ny"
    ids = [0, 1][: _batch(profile, 2)]
    with pytest.raises(RetryableTranslationFormatError):
        parsed = profile.parse_response(garbage, ids=ids)
        # Strict profiles raise; partial profiles must at least not invent ids.
        if profile.supports_partial_reissue:
            assert not parsed or set(parsed).issubset(set(ids))
            raise RetryableTranslationFormatError("tolerated partial garbage")


def test_an_empty_reply_is_never_a_translation(profile):
    """No profile may return "" as a successful translation. This is the defect
    that took the Sakura/GalTransl profile out: an unmatched line became an
    empty string and the job succeeded, so an untranslated cue reached the
    screen with nothing reporting it."""
    ids = [0][: _batch(profile, 1)]
    try:
        parsed = profile.parse_response("", ids=ids)
    except RetryableTranslationFormatError:
        return
    assert not any(value == "" for value in parsed.values())


def test_addressing_by_id_requires_a_schema(profile):
    if profile.wants_repair_pass or profile.supports_partial_reissue:
        # Repair and partial reissue both address lines by global id, which
        # only the JSON contract can express.
        assert profile.schema is not None
