from __future__ import annotations

import argparse
from datetime import datetime
import html
import json
import re
import shutil
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = PROJECT_ROOT / "agents" / "audits"
AUDIT_RM_ROOT = PROJECT_ROOT / "agents" / "rm" / "audit-deletions"
AUDIT_SERVER_COMMAND = "tools/audits/serve_audits.ps1"
ANON_LABELS = {
    "NAMH-055": "匿名样片 A",
    "REAL-988": "匿名样片 B",
    "FJIN-059": "匿名样片 C",
    "867HTTM-0045": "匿名样片 D",
    "BONY-173": "匿名样片 E",
    "HAME-052": "匿名样片 F",
    "MADM-217": "匿名样片 G",
    "MKMP-549": "匿名样片 H",
    "MKMP-577": "匿名样片 I",
    "NMSL-036": "匿名样片 J",
    "SORA-575": "匿名样片 K",
}


def anonymize_display_text(value: str) -> str:
    text = str(value)
    for raw, label in ANON_LABELS.items():
        text = text.replace(raw, label)
    return text


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
    for key in ("review_item_count", "rows", "subtitle_cue_count", "fallback_rows", "long_chunks"):
        value = summary.get(key)
        if value is not None:
            if key == "review_item_count":
                parts.append(f"{value} 条")
            elif key == "rows":
                parts.append(f"{value} 行")
            elif key == "subtitle_cue_count":
                parts.append(f"{value} 字幕")
            elif key == "fallback_rows":
                parts.append(f"{value} fallback rows")
            elif key == "long_chunks":
                parts.append(f"{value} long chunks")
    for key in ("video_label", "video", "dataset_id"):
        value = summary.get(key)
        if value:
            parts.append(anonymize_display_text(str(value)))
            break
    return " · ".join(parts) if parts else "审计页面"


def _entry_title(index_path: Path, summary: Mapping[str, Any]) -> str:
    if summary.get("title"):
        return anonymize_display_text(str(summary["title"]))
    if summary.get("dataset_id"):
        return anonymize_display_text(str(summary["dataset_id"]))
    return anonymize_display_text(index_path.parent.name)


def _audit_entry_mtime(index_path: Path) -> float:
    candidates = [index_path, index_path.parent]
    for summary_path in (index_path.parent / "summary.json", index_path.with_suffix(".summary.json")):
        if summary_path.exists():
            candidates.append(summary_path)
    return max(path.stat().st_mtime for path in candidates if path.exists())


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
                "dir": rel_url(index_path.parent, from_dir=audit_root),
                "mtime": str(_audit_entry_mtime(index_path)),
            }
        )
    entries.sort(key=lambda entry: float(entry["mtime"]), reverse=True)
    return entries


def _card(entry: Mapping[str, str], *, latest_href: str) -> str:
    href = str(entry.get("href") or "")
    title = str(entry.get("title") or href)
    desc = str(entry.get("desc") or "审计页面")
    badge = '<span class="badge">最新</span>' if href == latest_href else ""
    delete_label = f"删除 {title}"
    return (
        f'  <div class="entry" data-href="{html.escape(href)}">\n'
        f'    <a class="entry-main" href="{html.escape(href)}">\n'
        f"      <strong>{html.escape(title)}{badge}</strong>\n"
        f"      <span>{html.escape(desc)}</span>\n"
        "    </a>\n"
        f'    <button class="delete-audit" type="button" data-href="{html.escape(href)}" '
        f'data-title="{html.escape(title)}" aria-label="{html.escape(delete_label)}">删除</button>\n'
        "  </div>"
    )


def _current_latest_href(audit_root: Path) -> str:
    latest_path = audit_root / "latest-audit.html"
    if not latest_path.exists():
        return ""
    try:
        text = latest_path.read_text(encoding="utf-8")
    except Exception:
        return ""
    for href in re.findall(r'href="([^"]+)"', text):
        if href and href != "index.html" and not href.startswith(("#", "http:", "https:")):
            return href
    return ""


def _resolve_audit_entry_dir(*, audit_root: Path, href: str) -> Path:
    if not href.strip():
        raise ValueError("empty audit href")
    if "://" in href or href.startswith(("/", "\\")):
        raise ValueError(f"audit href must be relative: {href}")
    target = (audit_root / href).resolve()
    root = audit_root.resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"audit href escapes audit root: {href}") from exc
    target_dir = target.parent if target.name == "index.html" or target.suffix else target
    if target_dir == root:
        raise ValueError("refusing to delete audit root")
    try:
        target_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"audit directory escapes audit root: {target_dir}") from exc
    if not (target_dir / "index.html").exists():
        raise FileNotFoundError(f"audit index not found: {target_dir / 'index.html'}")
    return target_dir


def _unique_rm_dest(*, rm_root: Path, target_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = rm_root / f"{stamp}-{target_dir.name}"
    dest = base
    suffix = 2
    while dest.exists():
        dest = rm_root / f"{base.name}-{suffix}"
        suffix += 1
    return dest


def _newest_entry_index(audit_root: Path) -> Path | None:
    entries = [
        path for path in audit_root.glob("**/index.html")
        if path != audit_root / "index.html"
    ]
    if not entries:
        return None
    return max(entries, key=_audit_entry_mtime)


def _latest_index_after_delete(*, audit_root: Path, deleted_href: str, previous_latest_href: str) -> Path | None:
    if previous_latest_href and previous_latest_href != deleted_href:
        latest_candidate = (audit_root / previous_latest_href).resolve()
        try:
            latest_candidate.relative_to(audit_root.resolve())
        except ValueError:
            latest_candidate = Path()
        if latest_candidate.exists():
            return latest_candidate
    return _newest_entry_index(audit_root)


def write_empty_latest_audit_entry(*, audit_root: Path) -> None:
    audit_root.mkdir(parents=True, exist_ok=True)
    (audit_root / "latest-audit.html").write_text(
        """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>最新审计页入口</title>
</head>
<body>
<h1>最新审计页入口</h1>
<p>当前没有审计页。</p>
<p><a href="index.html">返回审计导航</a></p>
</body>
</html>
""",
        encoding="utf-8",
    )


def refresh_audit_entrypoints_after_change(
    *,
    audit_root: Path = AUDIT_ROOT,
    latest_html: Path | None = None,
    latest_title: str = "",
) -> Path | None:
    if latest_html is None:
        latest_html = _newest_entry_index(audit_root)
    if latest_html is None:
        write_empty_latest_audit_entry(audit_root=audit_root)
        write_audit_index(audit_root=audit_root)
        return None
    summary = _summary_for(latest_html)
    title = latest_title or _entry_title(latest_html, summary)
    write_latest_audit_entry(audit_root=audit_root, latest_html=latest_html, title=title)
    write_audit_index(audit_root=audit_root, latest_html=latest_html, latest_title=title)
    return latest_html


def delete_audit_entry(
    *,
    href: str,
    audit_root: Path = AUDIT_ROOT,
    rm_root: Path = AUDIT_RM_ROOT,
) -> dict[str, str]:
    target_dir = _resolve_audit_entry_dir(audit_root=audit_root, href=href)
    deleted_href = rel_url(target_dir / "index.html", from_dir=audit_root)
    previous_latest_href = _current_latest_href(audit_root)
    rm_root.mkdir(parents=True, exist_ok=True)
    dest = _unique_rm_dest(rm_root=rm_root, target_dir=target_dir)
    shutil.move(str(target_dir), str(dest))
    latest_html = _latest_index_after_delete(
        audit_root=audit_root,
        deleted_href=deleted_href,
        previous_latest_href=previous_latest_href,
    )
    refreshed_latest = refresh_audit_entrypoints_after_change(
        audit_root=audit_root,
        latest_html=latest_html,
    )
    return {
        "deleted_href": deleted_href,
        "moved_to": project_rel(dest),
        "latest_href": rel_url(refreshed_latest, from_dir=audit_root) if refreshed_latest else "",
    }


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
    if not cards:
        cards = '  <p class="muted">当前没有审计页。新的审计生成后会按更新时间倒序显示在这里。</p>'
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
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 12px;
  align-items: center;
  margin: 12px 0;
  padding: 14px 16px;
  border: 1px solid #d8ddd8;
  border-radius: 8px;
  background: #fff;
  color: inherit;
}}
.entry-main {{
  display: block;
  min-width: 0;
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
.delete-audit {{
  border: 1px solid #efb5af;
  border-radius: 6px;
  background: #fff0ee;
  color: #b42318;
  cursor: pointer;
  padding: 7px 10px;
  white-space: nowrap;
}}
.delete-audit:hover {{ background: #ffe3df; }}
.badge {{
  display: inline-block;
  border: 1px solid #0f766e;
  border-radius: 999px;
  padding: 1px 8px;
  font-size: 12px;
  line-height: 1.4;
}}
.muted {{ color: #66706c; font-size: 13px; }}
.status {{
  margin: 14px 0 0;
  padding: 10px 12px;
  border: 1px solid #d8ddd8;
  border-radius: 8px;
  background: #fff;
  color: #40504a;
  font-size: 13px;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}}
.status:empty {{ display: none; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
</style>
</head>
<body>
<main>
  <h1>JAVTrans 审计导航</h1>
{cards}
  <div class="status" id="deleteStatus"></div>
  <p class="muted">{html.escape(latest_meta)}</p>
  <p class="muted">所有长期审计页统一放在 <code>agents/audits/</code>，从本页进入；按更新时间倒序排列，最上面是最新需要审计的页面。不使用自动跳转。推荐用 <code>{AUDIT_SERVER_COMMAND}</code> 启动轻量审计服务，它从项目根目录提供媒体文件，支持音频 Range seek，不 watch 项目目录也不注入自动刷新脚本。删除按钮只有通过该脚本启动时才能直接移动本地文件并重建导航；否则页面会给出可手动运行的删除命令。删除会移动到 <code>agents/rm/audit-deletions/</code>。</p>
</main>
<script>
const statusBox = document.getElementById("deleteStatus");
function setStatus(text) {{
  statusBox.textContent = text || "";
}}
function deleteCommand(href) {{
  return `PYTHONIOENCODING=utf-8 UV_CACHE_DIR=agents/temp/uv-cache uv run python tools/audits/audit_nav.py delete --href "${{href.replaceAll('"', '\\"')}}"`;
}}
async function copyText(text) {{
  try {{
    await navigator.clipboard.writeText(text);
    return true;
  }} catch (_) {{
    return false;
  }}
}}
async function postDeleteAudit(href) {{
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), 45000);
  try {{
    const response = await fetch("/__audit_api__/delete-audit", {{
      method: "POST",
      headers: {{"Content-Type": "application/json"}},
      body: JSON.stringify({{href}}),
      signal: controller.signal
    }});
    const payload = await response.json().catch(() => ({{}}));
    return {{response, payload}};
  }} finally {{
    window.clearTimeout(timer);
  }}
}}
async function deleteAudit(button) {{
  const href = button.dataset.href || "";
  const title = button.dataset.title || href;
  if (!href) return;
  if (!window.confirm(`删除审计记录？\\n\\n${{title}}\\n\\n文件会移动到 agents/rm/audit-deletions/。`)) return;
  button.disabled = true;
  button.textContent = "删除中";
  try {{
    const {{response, payload}} = await postDeleteAudit(href);
    if (!response.ok || !payload.ok) {{
      throw new Error(payload.error || `HTTP ${{response.status}}`);
    }}
    const card = button.closest(".entry");
    if (card) card.remove();
    setStatus(`已移动到 ${{payload.moved_to || "agents/rm/audit-deletions/"}}。导航页已重建；刷新页面可看到最新状态。耗时 ${{payload.delete_elapsed_s ?? "?"}}s。`);
  }} catch (error) {{
    const command = deleteCommand(href);
    const copied = await copyText(command);
    const reason = error && error.name === "AbortError" ? "删除请求超时，服务端可能仍在移动目录；可稍后刷新页面确认。" : `错误：${{error.message || error}}`;
    setStatus(
      "审计服务删除 API 不可用，浏览器静态页不能直接删除文件。\\n" +
      "请用以下方式从项目根目录启动：" + {json.dumps(AUDIT_SERVER_COMMAND)} + "\\n" +
      "或直接运行删除命令" + (copied ? "（已复制）" : "") + "：\\n" + command + "\\n" +
      reason
    );
    button.disabled = false;
    button.textContent = "删除";
  }}
}}
for (const button of document.querySelectorAll(".delete-audit")) {{
  button.addEventListener("click", event => {{
    event.preventDefault();
    event.stopPropagation();
    deleteAudit(button);
  }});
}}
</script>
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


def _path_arg(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Maintain agents/audits navigation pages.")
    parser.add_argument("--audit-root", default=str(AUDIT_ROOT))
    parser.add_argument("--rm-root", default=str(AUDIT_RM_ROOT))
    subparsers = parser.add_subparsers(dest="command", required=True)

    delete_parser = subparsers.add_parser("delete", help="Move an audit entry to agents/rm and rebuild navigation.")
    delete_parser.add_argument("--href", required=True, help="Audit href from agents/audits/index.html")

    rebuild_parser = subparsers.add_parser("rebuild", help="Rebuild index.html/latest-audit.html")
    rebuild_parser.add_argument("--latest-href", help="Optional latest audit href")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit_root = _path_arg(args.audit_root)
    rm_root = _path_arg(args.rm_root)
    if args.command == "delete":
        result = delete_audit_entry(href=args.href, audit_root=audit_root, rm_root=rm_root)
        print(json.dumps({"ok": True, **result}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "rebuild":
        latest_html = None
        if args.latest_href:
            latest_html = _resolve_audit_entry_dir(audit_root=audit_root, href=args.latest_href) / "index.html"
        refreshed = refresh_audit_entrypoints_after_change(audit_root=audit_root, latest_html=latest_html)
        print(json.dumps({"ok": True, "latest_href": rel_url(refreshed, from_dir=audit_root) if refreshed else ""}, ensure_ascii=False, indent=2))
        return 0
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
