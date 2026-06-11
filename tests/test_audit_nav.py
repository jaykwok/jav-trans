from __future__ import annotations

import json
from pathlib import Path

from tools.audits.audit_nav import (
    delete_audit_entry,
    write_audit_index,
    write_latest_audit_entry,
)


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
    assert 'class="delete-audit"' in index
    assert "/__audit_api__/delete-audit" in index
    assert "tools/audits/serve_audits.sh" in index
    assert "127.0.0.1:8765" not in index
    assert "audit_nav.py serve" not in index
    assert "audit_nav.py delete --href" in index
    assert "按更新时间倒序排列" in index


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


def test_delete_audit_entry_moves_directory_and_rebuilds_nav(tmp_path: Path):
    audit_root = tmp_path / "agents" / "audits"
    rm_root = tmp_path / "agents" / "rm" / "audit-deletions"
    latest_html = _write_text(
        audit_root / "latest-review" / "index.html",
        "<!doctype html><title>latest</title>",
    )
    old_html = _write_text(
        audit_root / "old-review" / "index.html",
        "<!doctype html><title>old</title>",
    )
    (latest_html.parent / "summary.json").write_text(
        json.dumps({"review_item_count": 3, "title": "Latest"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (old_html.parent / "summary.json").write_text(
        json.dumps({"review_item_count": 2, "title": "Old"}, ensure_ascii=False),
        encoding="utf-8",
    )
    write_latest_audit_entry(audit_root=audit_root, latest_html=latest_html, title="Latest")
    write_audit_index(audit_root=audit_root, latest_html=latest_html, latest_title="Latest")

    result = delete_audit_entry(
        href="old-review/index.html",
        audit_root=audit_root,
        rm_root=rm_root,
    )

    assert result["deleted_href"] == "old-review/index.html"
    assert result["latest_href"] == "latest-review/index.html"
    assert not old_html.exists()
    moved = list(rm_root.glob("*-old-review"))
    assert len(moved) == 1
    assert (moved[0] / "index.html").exists()
    index = (audit_root / "index.html").read_text(encoding="utf-8")
    assert "old-review/index.html" not in index
    assert "latest-review/index.html" in index


def test_delete_latest_audit_entry_picks_remaining_latest(tmp_path: Path):
    audit_root = tmp_path / "agents" / "audits"
    rm_root = tmp_path / "agents" / "rm" / "audit-deletions"
    latest_html = _write_text(
        audit_root / "latest-review" / "index.html",
        "<!doctype html><title>latest</title>",
    )
    remaining_html = _write_text(
        audit_root / "remaining-review" / "index.html",
        "<!doctype html><title>remaining</title>",
    )
    write_latest_audit_entry(audit_root=audit_root, latest_html=latest_html, title="Latest")
    write_audit_index(audit_root=audit_root, latest_html=latest_html, latest_title="Latest")

    result = delete_audit_entry(
        href="latest-review/index.html",
        audit_root=audit_root,
        rm_root=rm_root,
    )

    assert result["deleted_href"] == "latest-review/index.html"
    assert result["latest_href"] == "remaining-review/index.html"
    assert not latest_html.exists()
    latest_entry = (audit_root / "latest-audit.html").read_text(encoding="utf-8")
    assert 'href="remaining-review/index.html"' in latest_entry
    assert 'href="latest-review/index.html"' not in latest_entry
    assert remaining_html.exists()


def test_audit_index_defaults_to_mtime_descending(tmp_path: Path):
    audit_root = tmp_path / "agents" / "audits"
    old_html = _write_text(
        audit_root / "old-review" / "index.html",
        "<!doctype html><title>old</title>",
    )
    new_html = _write_text(
        audit_root / "new-review" / "index.html",
        "<!doctype html><title>new</title>",
    )
    old_time = 1_700_000_000
    new_time = old_time + 100
    import os

    os.utime(old_html.parent, (old_time, old_time))
    os.utime(old_html, (old_time, old_time))
    os.utime(new_html.parent, (new_time, new_time))
    os.utime(new_html, (new_time, new_time))

    write_audit_index(audit_root=audit_root)

    index = (audit_root / "index.html").read_text(encoding="utf-8")
    assert index.index('href="new-review/index.html"') < index.index('href="old-review/index.html"')
