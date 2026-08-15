from __future__ import annotations

import json
from pathlib import Path

from tools.audits.audit_nav import (
    register_external_audit_page,
    unregister_external_audit_page,
    update_audit_entrypoints,
    write_audit_index,
)


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_register_external_audit_page_adds_to_navigation(tmp_path: Path):
    audit_root = tmp_path / "agents" / "audits"
    project_root = tmp_path
    external_page = _write_text(
        tmp_path / "agents" / "temp" / "20260727_test" / "index.html",
        "<!doctype html><title>external</title>",
    )
    (external_page.parent / "summary.json").write_text(
        json.dumps({"review_item_count": 5}, ensure_ascii=False),
        encoding="utf-8",
    )

    ok = register_external_audit_page(
        page_index=external_page,
        title="临时审计",
        audit_root=audit_root,
        project_root=project_root,
    )

    assert ok
    registry = audit_root / "external_pages.jsonl"
    assert registry.exists()
    lines = [line.strip() for line in registry.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["title"] == "临时审计"
    assert "../temp/20260727_test/index.html" in row["href"]

    write_audit_index(audit_root=audit_root, project_root=project_root)
    index = (audit_root / "index.html").read_text(encoding="utf-8")
    assert "临时审计" in index
    assert "../temp/20260727_test/index.html" in index
    assert "5 条" in index
    assert 'class="unregister-external"' in index


def test_unregister_external_audit_page_removes_from_navigation(tmp_path: Path):
    audit_root = tmp_path / "agents" / "audits"
    project_root = tmp_path
    external_page = _write_text(
        tmp_path / "agents" / "temp" / "20260727_test" / "index.html",
        "<!doctype html><title>external</title>",
    )

    register_external_audit_page(
        page_index=external_page,
        title="临时审计",
        audit_root=audit_root,
        project_root=project_root,
    )
    write_audit_index(audit_root=audit_root, project_root=project_root)
    index_before = (audit_root / "index.html").read_text(encoding="utf-8")
    assert "临时审计" in index_before

    ok = unregister_external_audit_page(
        href="../temp/20260727_test/index.html",
        audit_root=audit_root,
    )

    assert ok
    write_audit_index(audit_root=audit_root, project_root=project_root)
    index_after = (audit_root / "index.html").read_text(encoding="utf-8")
    assert "临时审计" not in index_after
    assert "../temp/20260727_test/index.html" not in index_after


def test_external_registry_prunes_missing_targets(tmp_path: Path):
    audit_root = tmp_path / "agents" / "audits"
    project_root = tmp_path
    external_page = _write_text(
        tmp_path / "agents" / "temp" / "20260727_test" / "index.html",
        "<!doctype html><title>external</title>",
    )

    register_external_audit_page(
        page_index=external_page,
        title="临时审计",
        audit_root=audit_root,
        project_root=project_root,
    )
    external_page.unlink()

    write_audit_index(audit_root=audit_root, project_root=project_root)

    index = (audit_root / "index.html").read_text(encoding="utf-8")
    assert "临时审计" not in index
    registry = audit_root / "external_pages.jsonl"
    if registry.exists():
        lines = [line.strip() for line in registry.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 0


def test_register_rejects_pages_under_audit_root(tmp_path: Path):
    audit_root = tmp_path / "agents" / "audits"
    project_root = tmp_path
    page_in_audit = _write_text(
        audit_root / "20260727_test" / "index.html",
        "<!doctype html><title>in audit</title>",
    )

    ok = register_external_audit_page(
        page_index=page_in_audit,
        title="应被拒绝",
        audit_root=audit_root,
        project_root=project_root,
    )

    assert not ok
    registry = audit_root / "external_pages.jsonl"
    assert not registry.exists()


def test_register_rejects_pages_outside_project(tmp_path: Path):
    audit_root = tmp_path / "agents" / "audits"
    project_root = tmp_path
    outside_page = _write_text(
        tmp_path.parent / "outside" / "index.html",
        "<!doctype html><title>outside</title>",
    )

    ok = register_external_audit_page(
        page_index=outside_page,
        title="应被拒绝",
        audit_root=audit_root,
        project_root=project_root,
    )

    assert not ok


def test_generator_written_page_outside_audit_root_still_reaches_the_index(tmp_path: Path):
    """A generated page nobody can navigate to is an audit that never happens.

    Every generator takes `--output-dir` and calls `update_audit_entrypoints`;
    agents/temp is a documented destination. That call used to return silently
    for anything outside the audit root, so the tool printed its summary, the
    page sat on disk, and `agents/audits/index.html` never mentioned it.
    """
    audit_root = tmp_path / "agents" / "audits"
    project_root = tmp_path
    page = _write_text(
        tmp_path / "agents" / "temp" / "20260813_ab" / "index.html",
        "<!doctype html><title>ab</title>",
    )

    update_audit_entrypoints(
        latest_html=page,
        title="翻译配置 · 匿名人工 A/B",
        audit_root=audit_root,
        project_root=project_root,
    )

    index = (audit_root / "index.html").read_text(encoding="utf-8")
    assert "翻译配置 · 匿名人工 A/B" in index
    assert "../temp/20260813_ab/index.html" in index
    latest = (audit_root / "latest-audit.html").read_text(encoding="utf-8")
    assert "../temp/20260813_ab/index.html" in latest


def test_page_inside_audit_root_is_not_added_to_the_external_registry(tmp_path: Path):
    # Scanned entries and registry entries would otherwise both list it.
    audit_root = tmp_path / "agents" / "audits"
    page = _write_text(
        audit_root / "20260813_ab" / "index.html",
        "<!doctype html><title>ab</title>",
    )

    update_audit_entrypoints(
        latest_html=page,
        title="内部审计页",
        audit_root=audit_root,
        project_root=tmp_path,
    )

    assert not (audit_root / "external_pages.jsonl").exists()
    index = (audit_root / "index.html").read_text(encoding="utf-8")
    assert "20260813_ab/index.html" in index
