from __future__ import annotations

import json
from pathlib import Path

from tools.audits.audit_nav import write_audit_index, write_latest_audit_entry


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_audit_index_lists_latest_without_auto_refresh(tmp_path: Path):
    audit_root = tmp_path / "agents" / "audits"
    latest_html = _write_text(
        audit_root / "latest-review" / "index.html",
        "<!doctype html><title>latest</title>",
    )
    old_html = _write_text(
        audit_root / "old-review" / "index.html",
        "<!doctype html><title>old</title>",
    )
    (latest_html.parent / "summary.json").write_text(
        json.dumps({"review_item_count": 3, "video_label": "匿名样片 A"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (old_html.parent / "summary.json").write_text(
        json.dumps({"rows": 2, "dataset_id": "old-review"}, ensure_ascii=False),
        encoding="utf-8",
    )

    write_audit_index(audit_root=audit_root, latest_html=latest_html, latest_title="最新审计")

    index = (audit_root / "index.html").read_text(encoding="utf-8")
    assert 'href="latest-review/index.html"' in index
    assert 'href="old-review/index.html"' in index
    assert "speech-boundary-ja/latest-review" not in index
    assert "最新审计" in index
    assert "匿名样片 A" in index
    assert "old-review" in index
    assert "http-equiv" not in index
    assert "window.location" not in index
    assert "location.reload" not in index


def test_latest_audit_entry_is_static_link(tmp_path: Path):
    audit_root = tmp_path / "agents" / "audits"
    latest_html = _write_text(
        audit_root / "latest-review" / "index.html",
        "<!doctype html><title>latest</title>",
    )

    write_latest_audit_entry(audit_root=audit_root, latest_html=latest_html, title="最新审计")

    entry = (audit_root / "latest-audit.html").read_text(encoding="utf-8")
    assert 'href="latest-review/index.html"' in entry
    assert "speech-boundary-ja/latest-review" not in entry
    assert "自动跳转已关闭" in entry
    assert "http-equiv" not in entry
    assert "window.location" not in entry
    assert "location.reload" not in entry
