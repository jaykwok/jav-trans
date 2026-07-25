from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading

import httpx
import pytest

from tools.omni.audio_teacher_transport import (
    GoogleAIStudioAudioTeacherTransport,
    create_audio_teacher_transport,
)
from tools.omni.gemini_native import (
    GEMINI_INTERACTIONS_ENDPOINT,
    GEMINI_NATIVE_EXECUTION_CONTRACT,
    GEMINI_NATIVE_MODEL,
    GeminiNativeAudioClient,
    GeminiNativeError,
    build_interaction_request,
    pacific_quota_date,
    pacific_rpd_reset_at,
    parse_comma_separated_api_keys,
)


def _audio(tmp_path: Path) -> Path:
    path = tmp_path / "sample.wav"
    path.write_bytes(b"RIFF" + b"\x00" * 128)
    return path


def _success_payload(text: str = '{"ok":true}') -> dict:
    return {
        "id": "interaction-test",
        "status": "completed",
        "steps": [
            {"type": "thought", "signature": "opaque-signature"},
            {
                "type": "model_output",
                "content": [{"type": "text", "text": text}],
            },
        ],
        "usage": {
            "total_input_tokens": 100,
            "total_output_tokens": 20,
            "total_thought_tokens": 40,
            "total_tokens": 160,
        },
    }


def test_native_request_matches_interactions_audio_contract(tmp_path: Path) -> None:
    body = build_interaction_request(
        audio_path=_audio(tmp_path),
        system_prompt="system",
        prompt="prompt",
        model=GEMINI_NATIVE_MODEL,
        thinking_level="medium",
        max_output_tokens=8192,
        response_schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        },
    )
    assert body["model"] == "gemini-3.6-flash"
    assert body["system_instruction"] == "system"
    assert body["input"][0] == {"type": "text", "text": "prompt"}
    assert body["input"][1]["type"] == "audio"
    assert body["input"][1]["mime_type"] == "audio/wav"
    assert body["input"][1]["data"]
    assert body["generation_config"] == {
        "max_output_tokens": 8192,
        "thinking_level": "medium",
        "thinking_summaries": "auto",
    }
    assert body["store"] is False
    assert body["response_format"]["mime_type"] == "application/json"
    wire = json.dumps(body)
    assert "temperature" not in wire
    assert "top_p" not in wire
    assert "top_k" not in wire


def test_native_client_rotates_only_after_http_429(tmp_path: Path) -> None:
    headers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        headers.append(request.headers["x-goog-api-key"])
        if len(headers) == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "60"},
                json={"error": {"status": "RESOURCE_EXHAUSTED"}},
            )
        return httpx.Response(200, json=_success_payload())

    logs: list[str] = []
    client = GeminiNativeAudioClient(
        api_keys=("key-one", "key-two"),
        min_request_interval_s=0.0,
        transport=httpx.MockTransport(handler),
        log=logs.append,
    )
    response = client.call_json(
        audio_path=_audio(tmp_path),
        system_prompt="system",
        prompt="prompt",
        response_schema=None,
    )
    assert response.parsed == {"ok": True}
    assert response.key_slot == 2
    assert headers == ["key-one", "key-two"]
    assert logs == [
        "gemini_key_rotate reason=http_429 from_slot=1/2 to_slot=2/2"
    ]
    assert response.raw["key_slot"] == 2
    assert response.raw["key_count"] == 2
    assert response.raw["usage"]["total_thought_tokens"] == 40
    assert "key-one" not in json.dumps(response.raw)
    assert "key-two" not in json.dumps(response.raw)


def test_native_client_does_not_rotate_on_auth_error(tmp_path: Path) -> None:
    headers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        headers.append(request.headers["x-goog-api-key"])
        return httpx.Response(
            401,
            json={"error": {"status": "UNAUTHENTICATED", "message": "bad key"}},
        )

    client = GeminiNativeAudioClient(
        api_keys=("key-one", "key-two"),
        min_request_interval_s=0.0,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(GeminiNativeError, match="HTTP 401"):
        client.call_json(
            audio_path=_audio(tmp_path),
            system_prompt="system",
            prompt="prompt",
            response_schema=None,
        )
    assert headers == ["key-one"]


def test_native_client_assigns_concurrent_calls_to_distinct_key_slots(
    tmp_path: Path,
) -> None:
    barrier = threading.Barrier(2)
    headers: list[str] = []
    header_lock = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        with header_lock:
            headers.append(request.headers["x-goog-api-key"])
        barrier.wait(timeout=2.0)
        return httpx.Response(200, json=_success_payload())

    client = GeminiNativeAudioClient(
        api_keys=("key-one", "key-two"),
        min_request_interval_s=0.0,
        quota_state_path=tmp_path / "gemini.quota.json",
        transport=httpx.MockTransport(handler),
    )

    def call() -> int:
        return client.call_json(
            audio_path=_audio(tmp_path),
            system_prompt="system",
            prompt="prompt",
            response_schema=None,
        ).key_slot

    with ThreadPoolExecutor(max_workers=2) as executor:
        slots = sorted(future.result() for future in [executor.submit(call), executor.submit(call)])
    assert slots == [1, 2]
    assert sorted(headers) == ["key-one", "key-two"]
    state = json.loads(
        (tmp_path / "gemini.quota.json").read_text(encoding="utf-8")
    )
    assert sorted(
        item["requests_started"] for item in state["keys"].values()
    ) == [1, 1]


def test_native_profile_accepts_two_comma_separated_keys(tmp_path: Path) -> None:
    assert parse_comma_separated_api_keys(" a, b, a ") == ("a", "b")
    env_file = tmp_path / "gemini"
    env_file.write_text(
        "GEMINI_API_KEY=key-one,key-two\nGEMINI_MODEL=gemini-3.6-flash\n",
        encoding="utf-8",
    )
    transport = create_audio_teacher_transport(
        profile="gemini",
        env_file=env_file,
    )
    assert isinstance(transport, GoogleAIStudioAudioTeacherTransport)
    assert transport.model == GEMINI_NATIVE_MODEL
    assert transport.api_key_count == 2
    assert transport.execution_contract == GEMINI_NATIVE_EXECUTION_CONTRACT
    assert transport.client.endpoint == GEMINI_INTERACTIONS_ENDPOINT


def test_native_rpd_budget_rotates_proactively_and_persists(tmp_path: Path) -> None:
    headers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        headers.append(request.headers["x-goog-api-key"])
        return httpx.Response(200, json=_success_payload())

    quota = tmp_path / "gemini.quota.json"
    now = lambda: datetime(2026, 7, 25, 18, tzinfo=timezone.utc)
    client = GeminiNativeAudioClient(
        api_keys=("key-one", "key-two"),
        min_request_interval_s=0.0,
        daily_request_limit=1,
        quota_state_path=quota,
        now_utc=now,
        transport=httpx.MockTransport(handler),
    )
    first = client.call_json(
        audio_path=_audio(tmp_path),
        system_prompt="system",
        prompt="prompt",
        response_schema=None,
    )
    second = client.call_json(
        audio_path=_audio(tmp_path),
        system_prompt="system",
        prompt="prompt",
        response_schema=None,
    )
    assert first.key_slot == 1
    assert second.key_slot == 2
    assert headers == ["key-one", "key-two"]
    assert first.raw["daily_requests_started_after"] == 1
    assert quota.is_file()
    wire = quota.read_text(encoding="utf-8")
    assert "key-one" not in wire and "key-two" not in wire
    state = json.loads(wire)
    assert sorted(
        item["requests_started"] for item in state["keys"].values()
    ) == [1, 1]

    resumed = GeminiNativeAudioClient(
        api_keys=("key-one", "key-two"),
        min_request_interval_s=0.0,
        daily_request_limit=1,
        quota_state_path=quota,
        now_utc=now,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(GeminiNativeError, match="1 RPD budget"):
        resumed.call_json(
            audio_path=_audio(tmp_path),
            system_prompt="system",
            prompt="prompt",
            response_schema=None,
        )
    assert headers == ["key-one", "key-two"]


def test_native_rpd_counts_failed_outbound_requests(tmp_path: Path) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            401,
            json={"error": {"status": "UNAUTHENTICATED", "message": "bad key"}},
        )

    quota = tmp_path / "gemini.quota.json"
    now = lambda: datetime(2026, 7, 25, 18, tzinfo=timezone.utc)
    client = GeminiNativeAudioClient(
        api_keys=("key-one",),
        min_request_interval_s=0.0,
        daily_request_limit=1,
        quota_state_path=quota,
        now_utc=now,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(GeminiNativeError, match="HTTP 401"):
        client.call_json(
            audio_path=_audio(tmp_path),
            system_prompt="system",
            prompt="prompt",
            response_schema=None,
        )
    with pytest.raises(GeminiNativeError, match="1 RPD budget"):
        client.call_json(
            audio_path=_audio(tmp_path),
            system_prompt="system",
            prompt="prompt",
            response_schema=None,
        )
    assert calls == 1
    state = json.loads(quota.read_text(encoding="utf-8"))
    item = next(iter(state["keys"].values()))
    assert item["requests_started"] == 1


def test_pacific_quota_date_uses_midnight_pacific() -> None:
    assert pacific_quota_date(
        datetime(2026, 7, 25, 6, 59, tzinfo=timezone.utc)
    ) == "2026-07-24"
    assert pacific_quota_date(
        datetime(2026, 7, 25, 7, 0, tzinfo=timezone.utc)
    ) == "2026-07-25"
    assert pacific_rpd_reset_at(
        datetime(2026, 7, 25, 7, 0, tzinfo=timezone.utc)
    ) == datetime(2026, 7, 26, 7, 0, tzinfo=timezone.utc)
    assert pacific_rpd_reset_at(
        datetime(2026, 1, 25, 8, 0, tzinfo=timezone.utc)
    ) == datetime(2026, 1, 26, 8, 0, tzinfo=timezone.utc)
    assert pacific_rpd_reset_at(
        datetime(2026, 3, 7, 8, 0, tzinfo=timezone.utc)
    ) == datetime(2026, 3, 8, 8, 0, tzinfo=timezone.utc)
    assert pacific_rpd_reset_at(
        datetime(2026, 3, 8, 8, 0, tzinfo=timezone.utc)
    ) == datetime(2026, 3, 9, 7, 0, tzinfo=timezone.utc)
    assert pacific_rpd_reset_at(
        datetime(2026, 10, 31, 7, 0, tzinfo=timezone.utc)
    ) == datetime(2026, 11, 1, 7, 0, tzinfo=timezone.utc)
    assert pacific_rpd_reset_at(
        datetime(2026, 11, 1, 7, 0, tzinfo=timezone.utc)
    ) == datetime(2026, 11, 2, 8, 0, tzinfo=timezone.utc)


def test_native_quota_json_exposes_rpm_tpm_rpd_and_reset_state(
    tmp_path: Path,
) -> None:
    now = lambda: datetime(2026, 7, 25, 18, tzinfo=timezone.utc)
    quota = tmp_path / "gemini.quota.json"
    client = GeminiNativeAudioClient(
        api_keys=("key-one",),
        min_request_interval_s=12.5,
        quota_state_path=quota,
        now_utc=now,
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json=_success_payload())
        ),
    )
    client.call_json(
        audio_path=_audio(tmp_path),
        system_prompt="system",
        prompt="prompt",
        response_schema=None,
    )
    status = client.quota_status()
    item = next(iter(status["keys"].values()))
    assert status["rpm_limit"] == 5
    assert status["tpm_limit"] == 250_000
    assert status["daily_request_limit"] == 20
    assert status["rpd_reset_at_utc"] == "2026-07-26T07:00:00.000Z"
    assert item["requests_started"] == 1
    assert item["rpd_remaining"] == 19
    assert item["rpm_requests_in_window"] == 1
    assert item["rpm_remaining"] == 4
    assert item["tpm_tokens_in_window"] == 160
    assert item["tpm_remaining"] == 249_840
    assert item["first_request_at_utc"] == "2026-07-25T18:00:00.000Z"
    assert item["last_request_at_utc"] == "2026-07-25T18:00:00.000Z"
    assert item["rpm_ready_at_utc"] == "2026-07-25T18:00:12.500Z"
    wire = quota.read_text(encoding="utf-8")
    assert "key-one" not in wire


def test_native_quota_cooldown_survives_client_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    moment = [datetime(2026, 7, 25, 18, tzinfo=timezone.utc)]
    waits: list[float] = []
    quota = tmp_path / "gemini.quota.json"
    handler = httpx.MockTransport(
        lambda _request: httpx.Response(200, json=_success_payload())
    )
    first = GeminiNativeAudioClient(
        api_keys=("key-one",),
        min_request_interval_s=12.5,
        quota_state_path=quota,
        now_utc=lambda: moment[0],
        transport=handler,
    )
    first.call_json(
        audio_path=_audio(tmp_path),
        system_prompt="system",
        prompt="prompt",
        response_schema=None,
    )

    def fake_sleep(seconds: float) -> None:
        waits.append(seconds)
        moment[0] += timedelta(seconds=seconds)

    monkeypatch.setattr("tools.omni.gemini_native.time.sleep", fake_sleep)
    resumed = GeminiNativeAudioClient(
        api_keys=("key-one",),
        min_request_interval_s=12.5,
        quota_state_path=quota,
        now_utc=lambda: moment[0],
        transport=handler,
    )
    resumed.call_json(
        audio_path=_audio(tmp_path),
        system_prompt="system",
        prompt="prompt",
        response_schema=None,
    )
    assert waits == [12.5]


def test_native_quota_state_survives_key_addition_and_reordering(
    tmp_path: Path,
) -> None:
    quota = tmp_path / "gemini.quota.json"
    now = lambda: datetime(2026, 7, 25, 18, tzinfo=timezone.utc)
    handler = httpx.MockTransport(
        lambda _request: httpx.Response(200, json=_success_payload())
    )
    first = GeminiNativeAudioClient(
        api_keys=("key-one", "key-two"),
        min_request_interval_s=0.0,
        quota_state_path=quota,
        now_utc=now,
        transport=handler,
    )
    first.call_json(
        audio_path=_audio(tmp_path),
        system_prompt="system",
        prompt="prompt",
        response_schema=None,
    )
    resumed = GeminiNativeAudioClient(
        api_keys=("key-new", "key-two", "key-one"),
        min_request_interval_s=0.0,
        quota_state_path=quota,
        now_utc=now,
        transport=handler,
    )
    state = resumed.quota_status()["keys"]
    fingerprint = lambda key: hashlib.sha256(key.encode()).hexdigest()
    assert state[fingerprint("key-one")]["requests_started"] == 1
    assert state[fingerprint("key-two")]["requests_started"] == 0
    assert state[fingerprint("key-new")]["requests_started"] == 0


def test_native_rpd_resets_when_running_process_crosses_pacific_midnight(
    tmp_path: Path,
) -> None:
    headers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        headers.append(request.headers["x-goog-api-key"])
        return httpx.Response(200, json=_success_payload())

    moments = [datetime(2026, 7, 25, 6, 59, tzinfo=timezone.utc)]
    client = GeminiNativeAudioClient(
        api_keys=("key-one",),
        min_request_interval_s=0.0,
        daily_request_limit=1,
        quota_state_path=tmp_path / "gemini.quota.json",
        now_utc=lambda: moments[0],
        transport=httpx.MockTransport(handler),
    )
    client.call_json(
        audio_path=_audio(tmp_path),
        system_prompt="system",
        prompt="prompt",
        response_schema=None,
    )
    moments[0] = datetime(2026, 7, 25, 7, 0, tzinfo=timezone.utc)
    response = client.call_json(
        audio_path=_audio(tmp_path),
        system_prompt="system",
        prompt="prompt",
        response_schema=None,
    )
    assert headers == ["key-one", "key-one"]
    assert response.raw["pacific_quota_date"] == "2026-07-25"
    assert response.raw["daily_requests_started_after"] == 1
