#!/usr/bin/env python3
"""Google AI Studio native Gemini audio transport.

This module deliberately does not share OpenAI-compatible request code with
OpenRouter or Qwen.  It implements the Gemini Interactions REST contract,
per-key quota pacing, and fail-closed API-key rotation on HTTP 429 only.
"""
from __future__ import annotations

import base64
import calendar
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import httpx


GEMINI_INTERACTIONS_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/interactions"
)
GEMINI_NATIVE_EXECUTION_CONTRACT = (
    "google_ai_interactions_inline_audio_medium_json_v1"
)
GEMINI_NATIVE_MODEL = "gemini-3.6-flash"
GEMINI_NATIVE_RPM_PER_KEY = 5
GEMINI_NATIVE_TPM_PER_KEY = 250_000
GEMINI_NATIVE_RPD_PER_KEY = 20
GEMINI_NATIVE_MIN_REQUEST_INTERVAL_S = 12.5
GEMINI_INLINE_REQUEST_LIMIT_BYTES = 20_000_000
GEMINI_QUOTA_STATE_SCHEMA = "gemini_native_quota_state_v3"


class GeminiNativeError(RuntimeError):
    """A sanitized native Gemini transport or response error."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def parse_comma_separated_api_keys(value: str) -> tuple[str, ...]:
    """Parse and deduplicate a comma-separated key list without logging it."""

    keys: list[str] = []
    seen: set[str] = set()
    for raw in value.split(","):
        key = raw.strip()
        if key and key not in seen:
            keys.append(key)
            seen.add(key)
    if not keys:
        raise ValueError("native Gemini profile contains no API key")
    return tuple(keys)


def first_mapping_value(
    values: Mapping[str, str], names: Sequence[str]
) -> tuple[str, str]:
    """Resolve a value from one loaded profile, avoiding process-env leakage."""

    for name in names:
        value = str(values.get(name) or "").strip()
        if value:
            return name, value
    return "", ""


def audio_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    explicit = {
        ".wav": "audio/wav",
        ".mp3": "audio/mp3",
        ".aiff": "audio/aiff",
        ".aif": "audio/aiff",
        ".aac": "audio/aac",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".m4a": "audio/m4a",
        ".opus": "audio/opus",
    }
    value = explicit.get(suffix) or mimetypes.guess_type(path.name)[0]
    if not value or not value.startswith("audio/"):
        raise ValueError(f"unsupported native Gemini audio format: {path.suffix}")
    return value


def build_interaction_request(
    *,
    audio_path: Path,
    system_prompt: str,
    prompt: str,
    model: str = GEMINI_NATIVE_MODEL,
    thinking_level: str = "medium",
    max_output_tokens: int = 8192,
    response_schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    audio_path = audio_path.resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    if thinking_level not in {"minimal", "low", "medium", "high"}:
        raise ValueError(f"invalid Gemini thinking_level: {thinking_level}")
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")
    body: dict[str, Any] = {
        "model": model,
        "system_instruction": system_prompt,
        "input": [
            {"type": "text", "text": prompt},
            {
                "type": "audio",
                "data": base64.b64encode(audio_path.read_bytes()).decode("ascii"),
                "mime_type": audio_mime_type(audio_path),
            },
        ],
        "generation_config": {
            "max_output_tokens": int(max_output_tokens),
            "thinking_level": thinking_level,
            "thinking_summaries": "auto",
        },
        "store": False,
    }
    if response_schema is not None:
        body["response_format"] = {
            "type": "text",
            "mime_type": "application/json",
            "schema": dict(response_schema),
        }
    encoded_size = len(
        json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    if encoded_size >= GEMINI_INLINE_REQUEST_LIMIT_BYTES:
        raise ValueError(
            "native Gemini inline request is at least 20 MB; use the Files API"
        )
    return body


def extract_interaction_output_text(payload: Mapping[str, Any]) -> str:
    """Extract the final model text from the REST response's step timeline."""

    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise GeminiNativeError("native Gemini response has no steps array")
    for step in reversed(steps):
        if not isinstance(step, Mapping) or step.get("type") != "model_output":
            continue
        content = step.get("content")
        if not isinstance(content, list):
            continue
        texts = [
            str(item.get("text") or "")
            for item in content
            if isinstance(item, Mapping)
            and item.get("type") == "text"
            and item.get("text")
        ]
        if texts:
            return "".join(texts)
    raise GeminiNativeError("native Gemini response has no model text output")


def _retry_after_seconds(response: httpx.Response) -> float:
    raw = response.headers.get("retry-after", "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    try:
        error = response.json().get("error", {})
        details = error.get("details", []) if isinstance(error, Mapping) else []
        for detail in details:
            if not isinstance(detail, Mapping):
                continue
            retry = detail.get("retryDelay")
            if isinstance(retry, str) and retry.endswith("s"):
                return max(0.0, float(retry[:-1]))
    except (ValueError, TypeError, AttributeError):
        pass
    return 60.0


def _safe_error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
        error = payload.get("error", {}) if isinstance(payload, Mapping) else {}
        message = str(error.get("message") or "") if isinstance(error, Mapping) else ""
        status = str(error.get("status") or "") if isinstance(error, Mapping) else ""
        safe = ": ".join(part for part in (status, message) if part)
        if safe:
            return safe[:500]
    except (ValueError, TypeError, AttributeError):
        pass
    return response.text[:500]


def _is_daily_quota_error(response: httpx.Response) -> bool:
    try:
        wire = json.dumps(response.json(), ensure_ascii=False).lower()
    except ValueError:
        wire = response.text.lower()
    return any(
        marker in wire
        for marker in ("perday", "per_day", "requests per day", "rpd")
    )


def pacific_quota_date(now_utc: datetime | None = None) -> str:
    """Return Google's RPD date without depending on an external tzdata wheel."""

    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    year = now.year
    march = calendar.monthcalendar(year, 3)
    second_sunday = [week[calendar.SUNDAY] for week in march if week[calendar.SUNDAY]][1]
    november = calendar.monthcalendar(year, 11)
    first_sunday = [week[calendar.SUNDAY] for week in november if week[calendar.SUNDAY]][0]
    dst_start_utc = datetime(year, 3, second_sunday, 10, tzinfo=timezone.utc)
    dst_end_utc = datetime(year, 11, first_sunday, 9, tzinfo=timezone.utc)
    offset_hours = -7 if dst_start_utc <= now < dst_end_utc else -8
    return (now + timedelta(hours=offset_hours)).date().isoformat()


def _pacific_midnight_offset_hours(local_date: date) -> int:
    march = calendar.monthcalendar(local_date.year, 3)
    second_sunday = [
        week[calendar.SUNDAY] for week in march if week[calendar.SUNDAY]
    ][1]
    november = calendar.monthcalendar(local_date.year, 11)
    first_sunday = [
        week[calendar.SUNDAY] for week in november if week[calendar.SUNDAY]
    ][0]
    dst_start = date(local_date.year, 3, second_sunday)
    dst_end = date(local_date.year, 11, first_sunday)
    return -7 if dst_start < local_date <= dst_end else -8


def pacific_rpd_reset_at(now_utc: datetime | None = None) -> datetime:
    """Return the next Pacific midnight as an aware UTC datetime."""

    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    local_today = date.fromisoformat(pacific_quota_date(now))
    local_tomorrow = local_today + timedelta(days=1)
    local_midnight = datetime.combine(
        local_tomorrow,
        datetime.min.time(),
        tzinfo=timezone(
            timedelta(hours=_pacific_midnight_offset_hours(local_tomorrow))
        ),
    )
    return local_midnight.astimezone(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _utc_iso(value: datetime) -> str:
    return _as_utc(value).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_utc(value: object) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return _as_utc(datetime.fromisoformat(raw.replace("Z", "+00:00")))
    except ValueError:
        return None


def _key_fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _write_quota_state(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


@dataclass(frozen=True)
class GeminiNativeResponse:
    parsed: dict[str, Any]
    raw: dict[str, Any]
    key_slot: int


class GeminiNativeAudioClient:
    """Thread-safe native client with round-robin slots and 429 rotation."""

    def __init__(
        self,
        *,
        api_keys: Sequence[str],
        model: str = GEMINI_NATIVE_MODEL,
        timeout_s: float = 240.0,
        min_request_interval_s: float = GEMINI_NATIVE_MIN_REQUEST_INTERVAL_S,
        endpoint: str = GEMINI_INTERACTIONS_ENDPOINT,
        transport: httpx.BaseTransport | None = None,
        log: Callable[[str], None] | None = None,
        daily_request_limit: int = GEMINI_NATIVE_RPD_PER_KEY,
        quota_state_path: Path | None = None,
        now_utc: Callable[[], datetime] | None = None,
    ) -> None:
        keys = tuple(str(key).strip() for key in api_keys if str(key).strip())
        if not keys:
            raise ValueError("native Gemini requires at least one API key")
        if len(set(keys)) != len(keys):
            raise ValueError("native Gemini API keys must be unique")
        self.api_keys = keys
        self.model = model
        self.timeout_s = float(timeout_s)
        self.min_request_interval_s = float(min_request_interval_s)
        self.endpoint = endpoint
        self.transport = transport
        self.log = log or (lambda _message: None)
        self.daily_request_limit = int(daily_request_limit)
        if self.daily_request_limit <= 0:
            raise ValueError("native Gemini daily_request_limit must be positive")
        self.quota_state_path = quota_state_path.resolve() if quota_state_path else None
        self._now_utc = now_utc or (lambda: datetime.now(timezone.utc))
        self._key_fingerprints = tuple(_key_fingerprint(key) for key in keys)
        self._daily_state = self._load_daily_state()
        self._state_lock = threading.RLock()
        self._slot_condition = threading.Condition(self._state_lock)
        self._in_flight: set[int] = set()
        self._next_key_index = 0
        with self._state_lock:
            self._refresh_daily_state()
            self._save_daily_state()

    def _empty_key_state(self) -> dict[str, Any]:
        return {
            "requests_started": 0,
            "first_request_at_utc": None,
            "last_request_at_utc": None,
            "minute_request_started_at_utc": [],
            "minute_token_events": [],
            "blocked_until_utc": None,
            "exhausted_by_429": False,
        }

    def _empty_daily_state(self) -> dict[str, Any]:
        now = _as_utc(self._now_utc())
        return {
            "schema": GEMINI_QUOTA_STATE_SCHEMA,
            "pacific_date": pacific_quota_date(now),
            "rpd_reset_at_utc": _utc_iso(pacific_rpd_reset_at(now)),
            "rpm_limit": GEMINI_NATIVE_RPM_PER_KEY,
            "tpm_limit": GEMINI_NATIVE_TPM_PER_KEY,
            "daily_request_limit": self.daily_request_limit,
            "keys": {
                fingerprint: self._empty_key_state()
                for fingerprint in self._key_fingerprints
            },
        }

    def _load_daily_state(self) -> dict[str, Any]:
        fresh = self._empty_daily_state()
        path = self.quota_state_path
        if path is None or not path.is_file():
            return fresh
        try:
            saved = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return fresh
        if not isinstance(saved, Mapping) or saved.get("schema") != GEMINI_QUOTA_STATE_SCHEMA:
            return fresh
        keys = saved.get("keys")
        if not isinstance(keys, Mapping):
            return fresh
        same_day = saved.get("pacific_date") == fresh["pacific_date"]
        for fingerprint in self._key_fingerprints:
            item = keys.get(fingerprint)
            if not isinstance(item, Mapping):
                continue
            copied = self._empty_key_state()
            copied.update(
                minute_request_started_at_utc=list(
                    item.get("minute_request_started_at_utc") or ()
                ),
                minute_token_events=list(item.get("minute_token_events") or ()),
                blocked_until_utc=item.get("blocked_until_utc"),
            )
            if same_day:
                copied.update(
                    requests_started=max(
                        0, int(item.get("requests_started") or 0)
                    ),
                    first_request_at_utc=item.get("first_request_at_utc"),
                    last_request_at_utc=item.get("last_request_at_utc"),
                    exhausted_by_429=bool(item.get("exhausted_by_429")),
                )
            fresh["keys"][fingerprint] = copied
        return fresh

    def _update_key_status(self, item: dict[str, Any], *, now: datetime) -> None:
        cutoff = now - timedelta(seconds=60)
        request_times = sorted(
            value
            for value in (
                _parse_utc(raw)
                for raw in item.get("minute_request_started_at_utc") or ()
            )
            if value is not None and cutoff < value <= now + timedelta(seconds=1)
        )
        token_events: list[tuple[datetime, int]] = []
        for event in item.get("minute_token_events") or ():
            if not isinstance(event, Mapping):
                continue
            at = _parse_utc(event.get("at_utc"))
            tokens = max(0, int(event.get("tokens") or 0))
            if at is not None and tokens > 0 and cutoff < at <= now + timedelta(seconds=1):
                token_events.append((at, tokens))
        token_events.sort(key=lambda pair: pair[0])
        blocked_until = _parse_utc(item.get("blocked_until_utc"))
        if blocked_until is not None and blocked_until <= now:
            blocked_until = None

        ready_at = now
        if request_times:
            ready_at = max(
                ready_at,
                request_times[-1] + timedelta(seconds=self.min_request_interval_s),
            )
        if len(request_times) >= GEMINI_NATIVE_RPM_PER_KEY:
            ready_at = max(ready_at, request_times[0] + timedelta(seconds=60))
        token_total = sum(tokens for _at, tokens in token_events)
        if token_total >= GEMINI_NATIVE_TPM_PER_KEY and token_events:
            ready_at = max(ready_at, token_events[0][0] + timedelta(seconds=60))
        if blocked_until is not None:
            ready_at = max(ready_at, blocked_until)

        item["minute_request_started_at_utc"] = [
            _utc_iso(value) for value in request_times
        ]
        item["minute_token_events"] = [
            {"at_utc": _utc_iso(at), "tokens": tokens}
            for at, tokens in token_events
        ]
        item["blocked_until_utc"] = (
            _utc_iso(blocked_until) if blocked_until is not None else None
        )
        item["rpm_requests_in_window"] = len(request_times)
        item["rpm_remaining"] = max(
            0, GEMINI_NATIVE_RPM_PER_KEY - len(request_times)
        )
        item["tpm_tokens_in_window"] = token_total
        item["tpm_remaining"] = max(0, GEMINI_NATIVE_TPM_PER_KEY - token_total)
        item["rpd_remaining"] = max(
            0, self.daily_request_limit - int(item.get("requests_started") or 0)
        )
        item["rpm_ready_at_utc"] = _utc_iso(ready_at)
        item["rpd_reset_at_utc"] = self._daily_state["rpd_reset_at_utc"]

    def _save_daily_state(self) -> None:
        if self.quota_state_path is None:
            return
        now = _as_utc(self._now_utc())
        for item in self._daily_state["keys"].values():
            self._update_key_status(item, now=now)
        self._daily_state["updated_at_utc"] = _utc_iso(now)
        _write_quota_state(self.quota_state_path, self._daily_state)

    def _refresh_daily_state(self) -> None:
        with self._state_lock:
            current_date = pacific_quota_date(self._now_utc())
            if self._daily_state.get("pacific_date") == current_date:
                return
            previous = self._daily_state
            refreshed = self._empty_daily_state()
            previous_keys = previous.get("keys")
            if isinstance(previous_keys, Mapping):
                for fingerprint, item in refreshed["keys"].items():
                    old = previous_keys.get(fingerprint)
                    if not isinstance(old, Mapping):
                        continue
                    item["minute_request_started_at_utc"] = list(
                        old.get("minute_request_started_at_utc") or ()
                    )
                    item["minute_token_events"] = list(
                        old.get("minute_token_events") or ()
                    )
                    item["blocked_until_utc"] = old.get("blocked_until_utc")
            self._daily_state = refreshed
            self._save_daily_state()

    def _daily_item(self, key_index: int) -> dict[str, Any]:
        return self._daily_state["keys"][self._key_fingerprints[key_index]]

    def _daily_available(self, key_index: int) -> bool:
        item = self._daily_item(key_index)
        return (
            not bool(item["exhausted_by_429"])
            and int(item["requests_started"]) < self.daily_request_limit
        )

    def _record_daily_request_start(self, key_index: int) -> tuple[int, str]:
        """Persist the outbound request before sending it.

        RPD is a request budget, not a successful-response budget. Recording
        first also keeps a process crash or post-send network failure from
        silently reusing the same daily slot.
        """

        with self._state_lock:
            self._refresh_daily_state()
            now = _as_utc(self._now_utc())
            item = self._daily_item(key_index)
            item["requests_started"] = int(item["requests_started"]) + 1
            stamp = _utc_iso(now)
            item["first_request_at_utc"] = item.get("first_request_at_utc") or stamp
            item["last_request_at_utc"] = stamp
            item.setdefault("minute_request_started_at_utc", []).append(stamp)
            quota_date = str(self._daily_state["pacific_date"])
            self._save_daily_state()
            return int(item["requests_started"]), quota_date

    def _record_429(
        self,
        key_index: int,
        *,
        retry_s: float,
        daily: bool,
        quota_date: str,
    ) -> None:
        with self._state_lock:
            item = self._daily_item(key_index)
            item["blocked_until_utc"] = _utc_iso(
                _as_utc(self._now_utc()) + timedelta(seconds=max(0.0, retry_s))
            )
            if daily and self._daily_state.get("pacific_date") == quota_date:
                item["exhausted_by_429"] = True
            self._save_daily_state()

    def _reserve_key_slot(self, attempted_slots: set[int]) -> int:
        with self._slot_condition:
            while True:
                self._refresh_daily_state()
                for offset in range(len(self.api_keys)):
                    index = (self._next_key_index + offset) % len(self.api_keys)
                    if (
                        index not in attempted_slots
                        and index not in self._in_flight
                        and self._daily_available(index)
                    ):
                        self._in_flight.add(index)
                        self._next_key_index = (index + 1) % len(self.api_keys)
                        return index
                if any(
                    index not in attempted_slots and self._daily_available(index)
                    for index in range(len(self.api_keys))
                ):
                    self._slot_condition.wait(timeout=1.0)
                    continue
                raise GeminiNativeError(
                    "all native Gemini key slots returned HTTP 429 or reached "
                    f"the {self.daily_request_limit} RPD budget for the current "
                    "Pacific quota day",
                    status_code=429,
                )

    def _release_key_slot(self, key_index: int) -> None:
        with self._slot_condition:
            self._in_flight.discard(key_index)
            self._slot_condition.notify_all()

    def _wait_for_key_quota(self, key_index: int) -> None:
        while True:
            with self._state_lock:
                self._refresh_daily_state()
                now = _as_utc(self._now_utc())
                item = self._daily_item(key_index)
                self._update_key_status(item, now=now)
                ready_at = _parse_utc(item.get("rpm_ready_at_utc")) or now
                wait_s = max(0.0, (ready_at - now).total_seconds())
            if wait_s <= 0:
                return
            self.log(
                f"gemini_quota_wait key_slot={key_index + 1}/{len(self.api_keys)} "
                f"wait_s={wait_s:.1f}"
            )
            time.sleep(wait_s)

    def _record_usage(self, key_index: int, payload: Mapping[str, Any]) -> None:
        usage = payload.get("usage")
        if not isinstance(usage, Mapping):
            return
        total_tokens = int(usage.get("total_tokens") or 0)
        if total_tokens > 0:
            with self._state_lock:
                self._daily_item(key_index).setdefault(
                    "minute_token_events", []
                ).append(
                    {
                        "at_utc": _utc_iso(_as_utc(self._now_utc())),
                        "tokens": total_tokens,
                    }
                )
                self._save_daily_state()

    def quota_status(self) -> dict[str, Any]:
        """Return a secret-free snapshot suitable for logs and status UIs."""

        with self._state_lock:
            self._refresh_daily_state()
            now = _as_utc(self._now_utc())
            for item in self._daily_state["keys"].values():
                self._update_key_status(item, now=now)
            return json.loads(json.dumps(self._daily_state))

    def call_json(
        self,
        *,
        audio_path: Path,
        system_prompt: str,
        prompt: str,
        response_schema: Mapping[str, Any] | None,
        thinking_level: str = "medium",
        max_output_tokens: int = 8192,
    ) -> GeminiNativeResponse:
        body = build_interaction_request(
            audio_path=audio_path,
            system_prompt=system_prompt,
            prompt=prompt,
            model=self.model,
            thinking_level=thinking_level,
            max_output_tokens=max_output_tokens,
            response_schema=response_schema,
        )
        attempted_slots: set[int] = set()
        previous_429_slot: int | None = None
        while len(attempted_slots) < len(self.api_keys):
            key_index = self._reserve_key_slot(attempted_slots)
            if previous_429_slot is not None:
                self.log(
                    f"gemini_key_rotate reason=http_429 "
                    f"from_slot={previous_429_slot + 1}/{len(self.api_keys)} "
                    f"to_slot={key_index + 1}/{len(self.api_keys)}"
                )
                previous_429_slot = None
            attempted_slots.add(key_index)
            try:
                self._wait_for_key_quota(key_index)
                daily_requests, request_quota_date = (
                    self._record_daily_request_start(key_index)
                )
                try:
                    with httpx.Client(
                        timeout=self.timeout_s,
                        transport=self.transport,
                    ) as client:
                        response = client.post(
                            self.endpoint,
                            headers={
                                "x-goog-api-key": self.api_keys[key_index],
                                "Content-Type": "application/json",
                            },
                            json=body,
                        )
                except httpx.HTTPError as error:
                    raise GeminiNativeError(
                        f"native Gemini network error: {type(error).__name__}"
                    ) from error
                if response.status_code == 429:
                    retry_s = _retry_after_seconds(response)
                    self._record_429(
                        key_index,
                        retry_s=retry_s,
                        daily=_is_daily_quota_error(response),
                        quota_date=request_quota_date,
                    )
                    previous_429_slot = key_index
                    continue
                if response.status_code >= 400:
                    raise GeminiNativeError(
                        f"native Gemini HTTP {response.status_code}: "
                        f"{_safe_error_message(response)}",
                        status_code=response.status_code,
                    )
                try:
                    payload = response.json()
                except ValueError as error:
                    raise GeminiNativeError(
                        "native Gemini returned invalid JSON"
                    ) from error
                if not isinstance(payload, Mapping):
                    raise GeminiNativeError(
                        "native Gemini response must be an object"
                    )
                status = str(payload.get("status") or "completed")
                if status != "completed":
                    raise GeminiNativeError(
                        f"native Gemini interaction status is {status!r}"
                    )
                text = extract_interaction_output_text(payload)
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError as error:
                    raise GeminiNativeError(
                        "native Gemini model output is not valid JSON"
                    ) from error
                if not isinstance(parsed, Mapping):
                    raise GeminiNativeError(
                        "native Gemini model JSON output must be an object"
                    )
                self._record_usage(key_index, payload)
                steps = payload.get("steps")
                thought_steps = [
                    step
                    for step in (steps if isinstance(steps, list) else [])
                    if isinstance(step, Mapping) and step.get("type") == "thought"
                ]
                usage = (
                    payload.get("usage")
                    if isinstance(payload.get("usage"), Mapping)
                    else {}
                )
                raw = {
                    "transport": "google_ai_interactions",
                    "endpoint": self.endpoint,
                    "model": self.model,
                    "key_slot": key_index + 1,
                    "key_count": len(self.api_keys),
                    "key_fingerprint_sha256": self._key_fingerprints[key_index],
                    "daily_request_limit": self.daily_request_limit,
                    "daily_requests_started_after": daily_requests,
                    "pacific_quota_date": request_quota_date,
                    "status": status,
                    "usage": dict(usage),
                    "thought_step_count": len(thought_steps),
                    "thought_signature_present": any(
                        bool(step.get("signature")) for step in thought_steps
                    ),
                    "response": dict(payload),
                }
                return GeminiNativeResponse(
                    parsed=dict(parsed), raw=raw, key_slot=key_index + 1
                )
            finally:
                self._release_key_slot(key_index)
        raise GeminiNativeError(
            "native Gemini request exhausted all API key slots", status_code=429
        )
