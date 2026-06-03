from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = PROJECT_ROOT / "agents" / "audits"


def project_rel(value: str | Path | None) -> str:
    if not value:
        return ""
    raw = Path(value)
    try:
        return raw.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return raw.as_posix()


def rel_url(path: Path, *, from_dir: Path) -> str:
    try:
        return path.resolve().relative_to(from_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_for(index_path: Path) -> Mapping[str, Any]:
    for summary_path in (
        index_path.parent / "summary.json",
        index_path.with_suffix(".summary.json"),
    ):
        if not summary_path.exists():
            continue
        try:
            payload = read_json(summary_path)
        except Exception:
            continue
        if isinstance(payload, Mapping):
            return payload
    return {}


def _entry_desc(summary: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in ("review_item_count", "rows", "subtitle_cue_count", "long_chunks"):
        value = summary.get(key)
        if value is not None:
            if key == "review_item_count":
                parts.append(f"{value} 条")
            elif key == "rows":
                parts.append(f"{value} 行")
            elif key == "subtitle_cue_count":
                parts.append(f"{value} 字幕")
            elif key == "long_chunks":
                parts.append(f"{value} long chunks")
    for key in ("video_label", "video", "dataset_id"):
        value = summary.get(key)
        if value:
            parts.append(str(value))
            break
    return " · ".join(parts) if parts else "审计页面"


def _entry_title(index_path: Path, summary: Mapping[str, Any]) -> str:
    if summary.get("title"):
        return str(summary["title"])
    if summary.get("dataset_id"):
        return str(summary["dataset_id"])
    return index_path.parent.name


def _discover_entries(audit_root: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for index_path in sorted(audit_root.glob("**/index.html")):
        if index_path == audit_root / "index.html":
            continue
        summary = _summary_for(index_path)
        entries.append(
            {
                "href": rel_url(index_path, from_dir=audit_root),
                "title": _entry_title(index_path, summary),
                "desc": _entry_desc(summary),
            }
        )
    return entries


def _card(entry: Mapping[str, str], *, latest_href: str) -> str:
    href = str(entry.get("href") or "")
    title = str(entry.get("title") or href)
    desc = str(entry.get("desc") or "审计页面")
    badge = '<span class="badge">最新</span>' if href == latest_href else ""
    return (
        f'  <a class="entry" href="{html.escape(href)}">\n'
        f"    <strong>{html.escape(title)}{badge}</strong>\n"
        f"    <span>{html.escape(desc)}</span>\n"
        "  </a>"
    )


def write_latest_audit_entry(*, audit_root: Path, latest_html: Path, title: str) -> None:
    latest_rel = rel_url(latest_html, from_dir=audit_root)
    audit_root.mkdir(parents=True, exist_ok=True)
    (audit_root / "latest-audit.html").write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>最新审计页入口</title>
</head>
<body>
<h1>最新审计页入口</h1>
<p>自动跳转已关闭，避免审计过程中被浏览器反复刷新。</p>
<p><a href="{html.escape(latest_rel)}">打开 {html.escape(title)}</a></p>
<p><a href="{html.escape(rel_url(audit_root / "index.html", from_dir=audit_root))}">返回审计导航</a></p>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_audit_index(
    *,
    audit_root: Path = AUDIT_ROOT,
    latest_html: Path | None = None,
    latest_title: str = "",
) -> None:
    audit_root.mkdir(parents=True, exist_ok=True)
    entries = _discover_entries(audit_root)
    latest_href = ""
    if latest_html is not None:
        latest_href = rel_url(latest_html, from_dir=audit_root)
        latest_summary = _summary_for(latest_html)
        latest_entry = {
            "href": latest_href,
            "title": latest_title or latest_html.parent.name,
            "desc": _entry_desc(latest_summary) if latest_summary else "最新审计页",
        }
        entries = [latest_entry, *[entry for entry in entries if entry["href"] != latest_href]]
    cards = "\n".join(_card(entry, latest_href=latest_href) for entry in entries)
    latest_meta = f"当前 latest: {project_rel(latest_html)}" if latest_html else "当前 latest: 未指定"
    (audit_root / "index.html").write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>JAVTrans 审计导航</title>
<style>
body {{
  margin: 0;
  background: #f5f6f4;
  color: #1d2421;
  font: 15px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}}
main {{ max-width: 980px; margin: 0 auto; padding: 32px 18px; }}
h1 {{ margin: 0 0 18px; font-size: 24px; }}
.entry {{
  display: block;
  margin: 12px 0;
  padding: 14px 16px;
  border: 1px solid #d8ddd8;
  border-radius: 8px;
  background: #fff;
  color: inherit;
  text-decoration: none;
}}
.entry strong {{
  display: flex;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 4px;
  color: #0f766e;
}}
.entry span {{ color: #66706c; }}
.badge {{
  display: inline-block;
  border: 1px solid #0f766e;
  border-radius: 999px;
  padding: 1px 8px;
  font-size: 12px;
  line-height: 1.4;
}}
.muted {{ color: #66706c; font-size: 13px; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
</style>
</head>
<body>
<main>
  <h1>JAVTrans 审计导航</h1>
{cards}
  <p class="muted">{html.escape(latest_meta)}</p>
  <p class="muted">所有长期审计页统一放在 <code>agents/audits/</code>，从本页进入；不使用自动跳转，避免审计中刷新。</p>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )


def update_audit_entrypoints(*, latest_html: Path, title: str) -> None:
    try:
        latest_html.resolve().relative_to(AUDIT_ROOT.resolve())
    except ValueError:
        return
    write_latest_audit_entry(audit_root=AUDIT_ROOT, latest_html=latest_html, title=title)
    write_audit_index(audit_root=AUDIT_ROOT, latest_html=latest_html, latest_title=title)
