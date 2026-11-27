from __future__ import annotations

from pathlib import Path
import sys

import pytest

from tools._common import (
    PROJECT_ROOT,
    SRC_ROOT,
    ensure_src_on_path,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


def test_project_and_src_roots_point_to_repository() -> None:
    expected_root = Path(__file__).resolve().parents[2]
    assert PROJECT_ROOT == expected_root
    assert SRC_ROOT == expected_root / "src"
    assert (PROJECT_ROOT / "tools" / "_common.py").is_file()
    assert SRC_ROOT.is_dir()


def test_ensure_src_on_path_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "path", [])

    roots = ensure_src_on_path()
    assert roots == (PROJECT_ROOT, SRC_ROOT)
    assert str(PROJECT_ROOT) in sys.path
    assert str(SRC_ROOT) in sys.path

    before = list(sys.path)
    ensure_src_on_path()
    assert sys.path == before


def test_json_roundtrip_creates_parents_and_handles_utf8_bom(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "payload.json"
    payload = {"b": "中文", "a": 1}

    returned = write_json(target, payload, sort_keys=True)
    assert returned == target
    raw = target.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert raw.index('"a"') < raw.index('"b"')
    assert read_json(target) == payload

    bom_path = tmp_path / "bom.json"
    bom_path.write_text('{"k": "日本語"}', encoding="utf-8-sig")
    assert read_json(bom_path) == {"k": "日本語"}


def test_jsonl_roundtrip_skips_blank_lines_and_validates_mappings(tmp_path: Path) -> None:
    target = tmp_path / "rows" / "data.jsonl"
    rows = [{"id": 1, "text": "こんにちは"}, {"id": 2, "text": "你好"}]

    returned = write_jsonl(target, rows, sort_keys=True)
    assert returned == target
    assert read_jsonl(target, require_mapping=True) == rows

    sparse = tmp_path / "sparse.jsonl"
    sparse.write_text('\n{"ok": true}\n\n[1, 2]\n', encoding="utf-8-sig")
    assert read_jsonl(sparse) == [{"ok": True}, [1, 2]]
    with pytest.raises(ValueError, match=r"JSONL row must be an object:.*:4"):
        read_jsonl(sparse, require_mapping=True)
