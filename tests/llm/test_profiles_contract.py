"""Contract tests every registered profile must satisfy.

These are the invariants the engine relies on; a new profile module that
breaks one of these will fail here before it ever reaches a real run.
"""

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


def test_identity_fields(profile):
    assert profile.id and profile.version
    assert profile.cache_signature() == f"{profile.id}@{profile.version}"


def test_sampling_returns_dict(profile):
    sampling = profile.sampling(8)
    assert isinstance(sampling, dict)
    for key in sampling:
        assert key in {"temperature", "top_p", "max_tokens"}


def test_build_messages_shape(profile):
    ctx = ProfileContext(glossary="お姉ちゃん-姐姐", total_count=4)
    messages = profile.build_messages(_segments(4), ids=[0, 1, 2, 3], ctx=ctx)
    assert isinstance(messages, list) and messages
    for message in messages:
        assert message.get("role") in {"system", "user", "assistant"}
        assert isinstance(message.get("content"), str)
    # Every source line must reach the prompt: the cue plan is frozen and the
    # engine has no other channel to deliver text to the model.
    joined = "\n".join(m["content"] for m in messages)
    for seg in _segments(4):
        assert seg["text"] in joined


def test_parse_response_covers_ids_or_subset(profile):
    """parse_response may only ever map requested ids (1:1 invariant)."""
    ids = [3, 5]
    if profile.id == "sakura_galtransl":
        reply = "第一行\n第二行"
    else:
        reply = '{"translations":[{"id":3,"text":"第一行"},{"id":5,"text":"第二行"}]}'
    parsed = profile.parse_response(reply, ids=ids)
    assert set(parsed).issubset(set(ids))
    if not profile.supports_partial_reissue:
        assert set(parsed) == set(ids)
    assert parsed[3] == "第一行"
    assert parsed[5] == "第二行"


def test_parse_response_rejects_garbage(profile):
    with pytest.raises(RetryableTranslationFormatError):
        parsed = profile.parse_response("???not a contract reply???\nx\ny", ids=[0, 1])
        # Strict profiles raise; partial profiles must at least not invent ids.
        if profile.supports_partial_reissue:
            assert not parsed or set(parsed).issubset({0, 1})
            raise RetryableTranslationFormatError("tolerated partial garbage")


def test_history_flags_consistent(profile):
    if profile.needs_history:
        assert profile.history_limit >= 0
    if profile.wants_repair_pass or profile.supports_partial_reissue:
        # Repair and partial reissue both address lines by global id, which
        # only the JSON contract can express.
        assert profile.schema is not None
