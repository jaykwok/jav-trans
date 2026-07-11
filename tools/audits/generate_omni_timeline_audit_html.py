#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.audit_nav import update_audit_entrypoints  # noqa: E402


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _index(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["item_id"]): row for row in _read_jsonl(path)}


def _audio_href(audio_path: str, output_dir: Path) -> str:
    path = _project_path(audio_path)
    return Path(os.path.relpath(path, output_dir)).as_posix()


def _position(value: float, duration_s: float) -> float:
    return min(100.0, max(0.0, value / max(duration_s, 0.001) * 100.0))


def _matched_units(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        unit
        for unit in row.get("units") or []
        if str(unit.get("status")) == "matched"
        and float(unit.get("end_s") or 0.0) > float(unit.get("start_s") or 0.0)
    ]


def _timeline(units: list[dict[str, Any]], *, duration_s: float) -> str:
    blocks = []
    for unit in units:
        start_s = float(unit["start_s"])
        end_s = float(unit["end_s"])
        left = _position(start_s, duration_s)
        width = max(0.35, _position(end_s, duration_s) - left)
        title = html.escape(
            f"{unit.get('unit_id')} {start_s:.3f}-{end_s:.3f}s "
            f"confidence={float(unit.get('confidence') or 0.0):.3f}"
        )
        blocks.append(
            f'<span class="unit" style="left:{left:.4f}%;width:{width:.4f}%" '
            f'title="{title}">{html.escape(str(unit.get("text") or ""))}</span>'
        )
    return '<div class="timeline">' + "".join(blocks) + "</div>"


def _unit_rows(item: dict[str, Any], omni: dict[str, Any]) -> str:
    omni_index = {str(unit["unit_id"]): unit for unit in omni.get("units") or []}
    rows = []
    for expected in item.get("text_units") or []:
        unit = omni_index.get(str(expected["unit_id"])) or {}
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(expected['unit_id']))}</td>"
            f"<td>{html.escape(str(expected['text']))}</td>"
            f"<td>{html.escape(str(unit.get('status') or 'missing'))}</td>"
            f"<td>{float(unit.get('start_s') or 0.0):.3f}-{float(unit.get('end_s') or 0.0):.3f}</td>"
            f"<td>{float(unit.get('confidence') or 0.0):.3f}</td>"
            "</tr>"
        )
    return "".join(rows)


def build_audit(
    *,
    items_path: Path,
    omni_labels: Path,
    output_dir: Path,
    title: str,
    update_nav: bool,
) -> dict[str, Any]:
    item_index = _index(items_path)
    omni_index = _index(omni_labels)
    unknown_ids = set(omni_index) - set(item_index)
    if unknown_ids:
        raise ValueError(f"Omni labels contain unknown item IDs: {sorted(unknown_ids)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    playback_tracks: dict[str, list[dict[str, Any]]] = {}
    unit_count = 0
    matched_count = 0
    for item_id, omni in sorted(omni_index.items()):
        item = item_index[item_id]
        duration_s = float(item["duration_s"])
        expected_units = list(item.get("text_units") or [])
        matched_units = _matched_units(omni)
        unit_count += len(expected_units)
        matched_count += len(matched_units)
        coverage = len(matched_units) / len(expected_units) if expected_units else 0.0
        playback_tracks[item_id] = [
            {
                "unit_id": str(unit["unit_id"]),
                "text": str(unit["text"]),
                "start_s": float(unit["start_s"]),
                "end_s": float(unit["end_s"]),
            }
            for unit in matched_units
        ]
        cards.append(
            f"""
<section class="item">
  <h2>{html.escape(item_id)}</h2>
  <div class="meta">
    <span>{duration_s:.3f}s</span>
    <span>chunk {int(item['source_chunk_index'])}</span>
    <span>matched {len(matched_units)}/{len(expected_units)}</span>
    <span>coverage {coverage:.1%}</span>
  </div>
  <p class="transcript">{html.escape(str(item.get('transcript') or ''))}</p>
  <div class="player" data-item-id="{html.escape(item_id)}">
    <div class="caption-line" aria-live="off"><span class="caption-text"></span></div>
    <audio controls preload="metadata" src="{html.escape(_audio_href(str(item['audio_path']), output_dir))}"></audio>
  </div>
  <div class="lane"><div class="lane-label">Omni</div>{_timeline(matched_units, duration_s=duration_s)}</div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>ID</th><th>Text</th><th>Status</th><th>Omni time</th><th>Confidence</th></tr></thead>
      <tbody>{_unit_rows(item, omni)}</tbody>
    </table>
  </div>
</section>
"""
        )
    tracks_json = json.dumps(playback_tracks, ensure_ascii=False).replace("</", "<\\/")
    index_path = output_dir / "index.html"
    index_path.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: #202124; background: #f4f5f2; font: 14px/1.45 system-ui,sans-serif; letter-spacing: 0; }}
    header {{ padding: 18px max(18px,calc((100vw - 1200px)/2)); background: #202124; color: #fff; }}
    h1 {{ margin: 0 0 6px; font-size: 22px; }}
    header p {{ margin: 0; color: #d8ddd8; }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 14px 18px 28px; }}
    .item {{ margin: 14px 0; padding: 14px; border: 1px solid #d7dad4; border-radius: 8px; background: #fff; }}
    h2 {{ margin: 0 0 8px; font-size: 16px; overflow-wrap: anywhere; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }}
    .meta span {{ padding: 3px 7px; border: 1px solid #d7dad4; border-radius: 6px; background: #f8f8f5; font-size: 12px; }}
    .transcript {{ margin: 8px 0; font-size: 15px; }}
    .player {{ margin: 8px 0 12px; overflow: hidden; border: 1px solid #d7dad4; border-radius: 8px; background: #f8f8f5; }}
    .caption-line {{ display: grid; min-height: 50px; place-items: center; padding: 8px 12px; border-bottom: 1px solid #d7dad4; background: #202124; color: #fff; text-align: center; }}
    .caption-text {{ max-width: 100%; overflow-wrap: anywhere; font-size: 18px; font-weight: 650; }}
    audio {{ display: block; width: 100%; height: 38px; }}
    .lane {{ display: grid; grid-template-columns: 54px minmax(0,1fr); gap: 8px; align-items: center; margin: 12px 0; }}
    .lane-label {{ font-weight: 650; font-size: 12px; }}
    .timeline {{ position: relative; height: 30px; overflow: hidden; border: 1px solid #d7dad4; background: repeating-linear-gradient(90deg,#fff 0,#fff 9.8%,#eceee9 10%); }}
    .unit {{ position: absolute; top: 4px; height: 20px; overflow: hidden; padding: 1px 3px; border-radius: 3px; background: #287f5a; color: #fff; font-size: 11px; white-space: nowrap; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th,td {{ padding: 5px 6px; border-bottom: 1px solid #e2e4df; text-align: left; white-space: nowrap; }}
    th {{ background: #f4f5f2; }}
    @media (max-width:700px) {{ main {{ padding: 8px; }} .item {{ padding: 10px; }} }}
  </style>
</head>
<body>
  <header><h1>{html.escape(title)}</h1><p>{len(cards)} items · {matched_count}/{unit_count} matched Omni units</p></header>
  <main>{''.join(cards)}</main>
  <script>
    const playbackTracks = {tracks_json};
    function activeUnit(units,currentTime) {{
      let active = null;
      for (const unit of units) {{
        if (currentTime >= unit.start_s && currentTime < unit.end_s) {{
          if (active === null || unit.start_s >= active.start_s) active = unit;
        }}
      }}
      return active;
    }}
    for (const player of document.querySelectorAll(".player")) {{
      const audio = player.querySelector("audio");
      const caption = player.querySelector(".caption-text");
      const render = () => {{
        if (audio.paused && audio.currentTime === 0) {{
          caption.textContent = "";
          return;
        }}
        const unit = activeUnit(playbackTracks[player.dataset.itemId] || [],audio.currentTime);
        caption.textContent = unit ? unit.text : "";
      }};
      for (const eventName of ["timeupdate","seeking","seeked","play","pause"]) audio.addEventListener(eventName,render);
      audio.addEventListener("ended",() => {{ caption.textContent = ""; }});
      render();
    }}
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )
    summary = {
        "schema": "omni_timeline_audit_summary_v1",
        "item_count": len(cards),
        "unit_count": unit_count,
        "matched_unit_count": matched_count,
        "items": str(items_path),
        "omni_labels": str(omni_labels),
        "index_html": str(index_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if update_nav:
        update_audit_entrypoints(latest_html=index_path, title=title)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True)
    parser.add_argument("--omni-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="Omni Timeline Audit")
    parser.add_argument("--update-audit-nav", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build_audit(
                items_path=_project_path(args.items),
                omni_labels=_project_path(args.omni_labels),
                output_dir=_project_path(args.output_dir),
                title=args.title,
                update_nav=args.update_audit_nav,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
