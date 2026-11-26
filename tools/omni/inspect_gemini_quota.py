#!/usr/bin/env python3
"""Refresh and print secret-free native Gemini quota state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from tools.omni.audio_teacher_transport import (
    GoogleAIStudioAudioTeacherTransport,
    create_audio_teacher_transport,
)


def summarize_quota(status: Mapping[str, Any], *, state_path: Path) -> dict[str, Any]:
    keys = status.get("keys")
    if not isinstance(keys, Mapping):
        raise ValueError("native Gemini quota state has no keys")
    rows = []
    for slot, (fingerprint, item) in enumerate(keys.items(), start=1):
        if not isinstance(item, Mapping):
            raise ValueError("native Gemini quota key state must be an object")
        rows.append(
            {
                "slot": slot,
                "fingerprint_prefix": str(fingerprint)[:12],
                "rpm_requests_in_window": int(
                    item.get("rpm_requests_in_window") or 0
                ),
                "rpm_remaining": int(item.get("rpm_remaining") or 0),
                "tpm_tokens_in_window": int(
                    item.get("tpm_tokens_in_window") or 0
                ),
                "tpm_remaining": int(item.get("tpm_remaining") or 0),
                "rpd_requests_started": int(item.get("requests_started") or 0),
                "rpd_remaining": int(item.get("rpd_remaining") or 0),
                "first_request_at_utc": item.get("first_request_at_utc"),
                "last_request_at_utc": item.get("last_request_at_utc"),
                "rpm_ready_at_utc": item.get("rpm_ready_at_utc"),
                "blocked_until_utc": item.get("blocked_until_utc"),
                "rpd_ready_at_utc": item.get("rpd_ready_at_utc"),
                "rpd_next_release_at_utc": item.get("rpd_next_release_at_utc"),
                "rpd_blocked_until_utc": item.get("rpd_blocked_until_utc"),
                "last_daily_429_at_utc": item.get("last_daily_429_at_utc"),
                "exhausted_by_429": bool(item.get("exhausted_by_429")),
            }
        )
    return {
        "schema": status.get("schema"),
        "state_path": str(state_path),
        "key_count": len(rows),
        "rpd_accounting_mode": status.get("rpd_accounting_mode"),
        "rpd_window_s": status.get("rpd_window_s"),
        "quota_date": status.get("quota_date"),
        "rpd_reset_at_utc": status.get("rpd_reset_at_utc"),
        "rpd_reset_timezone": status.get("rpd_reset_timezone"),
        "rpd_reset_local_time": status.get("rpd_reset_local_time"),
        "rpd_reset_is_advisory": status.get("rpd_reset_is_advisory"),
        "rpd_next_ready_at_utc": status.get("rpd_next_ready_at_utc"),
        "rpm_limit_per_key": status.get("rpm_limit"),
        "tpm_limit_per_key": status.get("tpm_limit"),
        "rpd_limit_per_key": status.get("daily_request_limit"),
        "rpd_remaining_total": sum(row["rpd_remaining"] for row in rows),
        "keys": rows,
    }


def inspect(*, env_file: Path) -> dict[str, Any]:
    transport = create_audio_teacher_transport(
        profile="gemini",
        env_file=env_file,
    )
    if not isinstance(transport, GoogleAIStudioAudioTeacherTransport):
        raise TypeError("gemini profile did not resolve to native transport")
    if transport.quota_state_path is None:
        raise ValueError("native Gemini quota state path is disabled")
    return summarize_quota(
        transport.quota_status(),
        state_path=transport.quota_state_path,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        default="gemini",
        choices=("gemini",),
        help="Named native profile under ~/.config/omni/.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    result = inspect(
        env_file=(Path.home() / ".config" / "omni" / args.env_file).resolve()
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
