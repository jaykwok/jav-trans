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


def _lane(units: list[dict[str, Any]], *, duration_s: float, kind: str) -> str:
    blocks = []
    for unit in units:
        if kind == "omni" and str(unit.get("status")) != "matched":
            continue
        if kind == "fused" and not bool(unit.get("trainable")):
            continue
        start_s = float(unit.get("start_s") or 0.0)
        end_s = float(unit.get("end_s") or 0.0)
        left = _position(start_s, duration_s)
        width = max(0.35, _position(end_s, duration_s) - left)
        label = html.escape(str(unit.get("text") or unit.get("unit_id") or ""))
        score = float(
            unit.get("alignment_score")
            or unit.get("confidence")
            or unit.get("omni_confidence")
            or 0.0
        )
        title = html.escape(
            f"{unit.get('unit_id')} {start_s:.3f}-{end_s:.3f}s "
            f"score={score:.3f}"
        )
        blocks.append(
            f'<span class="unit {kind}" style="left:{left:.4f}%;width:{width:.4f}%" '
            f'title="{title}">{label}</span>'
        )
    return '<div class="timeline">' + "".join(blocks) + "</div>"


def _unit_rows(
    forced_units: list[dict[str, Any]],
    omni_units: list[dict[str, Any]],
    fused_units: list[dict[str, Any]],
) -> str:
    omni_index = {str(unit["unit_id"]): unit for unit in omni_units}
    fused_index = {str(unit["unit_id"]): unit for unit in fused_units}
    rows = []
    for forced in forced_units:
        unit_id = str(forced["unit_id"])
        omni = omni_index.get(unit_id) or {}
        fused = fused_index.get(unit_id) or {}
        start_delta = fused.get("start_delta_s")
        end_delta = fused.get("end_delta_s")
        start_delta_text = "" if start_delta is None else f"{float(start_delta):.3f}"
        end_delta_text = "" if end_delta is None else f"{float(end_delta):.3f}"
        rows.append(
            "<tr>"
            f"<td>{html.escape(unit_id)}</td>"
            f"<td>{html.escape(str(forced.get('text') or ''))}</td>"
            f"<td>{float(forced.get('start_s') or 0.0):.3f}-{float(forced.get('end_s') or 0.0):.3f}</td>"
            f"<td>{float(forced.get('alignment_score') or 0.0):.3f}</td>"
            f"<td>{html.escape(str(omni.get('status') or 'missing'))}</td>"
            f"<td>{float(omni.get('start_s') or 0.0):.3f}-{float(omni.get('end_s') or 0.0):.3f}</td>"
            f"<td>{float(omni.get('confidence') or 0.0):.3f}</td>"
            f"<td>{html.escape(str(fused.get('source') or 'missing'))}</td>"
            f"<td>{start_delta_text}</td>"
            f"<td>{end_delta_text}</td>"
            "</tr>"
        )
    return "".join(rows)


def _playback_units(units: list[dict[str, Any]], *, kind: str) -> list[dict[str, Any]]:
    rows = []
    for unit in units:
        if kind == "omni" and str(unit.get("status")) != "matched":
            continue
        if kind == "fused" and not bool(unit.get("trainable")):
            continue
        start_s = float(unit.get("start_s") or 0.0)
        end_s = float(unit.get("end_s") or 0.0)
        if end_s <= start_s:
            continue
        rows.append(
            {
                "unit_id": str(unit.get("unit_id") or ""),
                "text": str(unit.get("text") or ""),
                "start_s": start_s,
                "end_s": end_s,
            }
        )
    return rows


def build_audit(
    *,
    forced_labels: Path,
    omni_labels: Path,
    fused_labels: Path,
    output_dir: Path,
    title: str,
    update_nav: bool,
) -> dict[str, Any]:
    forced_index = _index(forced_labels)
    omni_index = _index(omni_labels)
    fused_index = _index(fused_labels)
    if set(forced_index) != set(omni_index) or set(forced_index) != set(fused_index):
        raise ValueError("forced, Omni, and fused item IDs must match exactly")
    output_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    playback_tracks: dict[str, dict[str, list[dict[str, Any]]]] = {}
    total_units = 0
    trainable_units = 0
    for item_id, forced in sorted(forced_index.items()):
        omni = omni_index[item_id]
        fused = fused_index[item_id]
        duration_s = float(forced["duration_s"])
        forced_units = list(forced.get("word_units") or [])
        omni_units = list(omni.get("units") or [])
        fused_units = list(fused.get("units") or [])
        item_tracks = {
            "forced": _playback_units(forced_units, kind="forced"),
            "omni": _playback_units(omni_units, kind="omni"),
            "fused": _playback_units(fused_units, kind="fused"),
        }
        playback_tracks[item_id] = item_tracks
        default_track = "fused" if item_tracks["fused"] else "omni"
        total_units += len(fused_units)
        trainable_units += sum(bool(unit.get("trainable")) for unit in fused_units)
        cards.append(
            f"""
<section class="item">
  <h2>{html.escape(item_id)}</h2>
  <div class="meta">
    <span>{duration_s:.3f}s</span>
    <span>chunk {int(forced['source_chunk_index'])}</span>
    <span>coverage {float(fused.get('trainable_coverage') or 0.0):.1%}</span>
    <span>trainable {html.escape(str(bool(fused.get('item_trainable'))).lower())}</span>
  </div>
  <p class="transcript">{html.escape(str(forced.get('transcript') or ''))}</p>
  <div class="player" data-item-id="{html.escape(item_id)}" data-track="{default_track}">
    <div class="caption-line" aria-live="off"><span class="caption-text"></span></div>
    <audio controls preload="metadata" src="{html.escape(_audio_href(str(forced['audio_path']), output_dir))}"></audio>
    <div class="track-selector" role="group" aria-label="选择单一字幕轨道">
      <button type="button" class="track-button" data-track="forced" aria-pressed="{'true' if default_track == 'forced' else 'false'}">Forced</button>
      <button type="button" class="track-button" data-track="omni" aria-pressed="{'true' if default_track == 'omni' else 'false'}">Omni</button>
      <button type="button" class="track-button" data-track="fused" aria-pressed="{'true' if default_track == 'fused' else 'false'}">Selected</button>
    </div>
  </div>
  <div class="lanes">
    <div class="lane-label">Forced</div>{_lane(forced_units, duration_s=duration_s, kind='forced')}
    <div class="lane-label">Omni</div>{_lane(omni_units, duration_s=duration_s, kind='omni')}
    <div class="lane-label">Selected</div>{_lane(fused_units, duration_s=duration_s, kind='fused')}
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>ID</th><th>Text</th><th>Forced time</th><th>F score</th><th>Omni</th><th>Omni time</th><th>O conf</th><th>Selection</th><th>start Δ</th><th>end Δ</th></tr></thead>
      <tbody>{_unit_rows(forced_units, omni_units, fused_units)}</tbody>
    </table>
  </div>
</section>
"""
        )
    index_path = output_dir / "index.html"
    tracks_json = json.dumps(playback_tracks, ensure_ascii=False).replace("</", "<\\/")
    index_path.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: #202124; background: #f4f5f2; font: 14px/1.45 system-ui, sans-serif; letter-spacing: 0; }}
    header {{ padding: 18px max(18px, calc((100vw - 1200px) / 2)); background: #202124; color: #fff; }}
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
    .track-selector {{ display: flex; gap: 0; padding: 7px 10px 9px; border-top: 1px solid #e2e4df; }}
    .track-button {{ min-width: 72px; padding: 5px 10px; border: 1px solid #b8bdb5; border-right-width: 0; background: #fff; color: #343734; cursor: pointer; }}
    .track-button:first-child {{ border-radius: 6px 0 0 6px; }}
    .track-button:last-child {{ border-right-width: 1px; border-radius: 0 6px 6px 0; }}
    .track-button[aria-pressed="true"] {{ background: #315b9d; color: #fff; border-color: #315b9d; }}
    .lanes {{ display: grid; grid-template-columns: 62px minmax(0,1fr); gap: 6px 8px; align-items: center; margin: 12px 0; }}
    .lane-label {{ font-weight: 650; font-size: 12px; }}
    .timeline {{ position: relative; height: 30px; overflow: hidden; border: 1px solid #d7dad4; background: repeating-linear-gradient(90deg,#fff 0,#fff 9.8%,#eceee9 10%); }}
    .unit {{ position: absolute; top: 4px; height: 20px; overflow: hidden; padding: 1px 3px; border-radius: 3px; color: #fff; font-size: 11px; white-space: nowrap; }}
    .forced {{ background: #b65327; }}
    .omni {{ background: #287f5a; }}
    .fused {{ background: #315b9d; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ padding: 5px 6px; border-bottom: 1px solid #e2e4df; text-align: left; white-space: nowrap; }}
    th {{ background: #f4f5f2; }}
    @media (max-width: 700px) {{ main {{ padding: 8px; }} .item {{ padding: 10px; }} .lanes {{ grid-template-columns: 52px minmax(0,1fr); }} }}
  </style>
</head>
<body>
  <header><h1>{html.escape(title)}</h1><p>{len(cards)} items · {trainable_units}/{total_units} consensus units</p></header>
  <main>{''.join(cards)}</main>
  <script>
    const playbackTracks = {tracks_json};

    function activeUnit(units, currentTime) {{
      let active = null;
      for (const unit of units) {{
        if (currentTime >= unit.start_s && currentTime < unit.end_s) {{
          if (active === null || unit.start_s >= active.start_s) active = unit;
        }}
      }}
      return active;
    }}

    function renderCaption(player) {{
      const audio = player.querySelector("audio");
      const caption = player.querySelector(".caption-text");
      const itemTracks = playbackTracks[player.dataset.itemId] || {{}};
      const units = itemTracks[player.dataset.track] || [];
      const unit = activeUnit(units, audio.currentTime);
      caption.textContent = unit ? unit.text : "";
    }}

    for (const player of document.querySelectorAll(".player")) {{
      const audio = player.querySelector("audio");
      for (const eventName of ["timeupdate", "seeking", "seeked", "play", "pause"]) {{
        audio.addEventListener(eventName, () => renderCaption(player));
      }}
      audio.addEventListener("ended", () => {{
        player.querySelector(".caption-text").textContent = "";
      }});
      for (const button of player.querySelectorAll(".track-button")) {{
        button.addEventListener("click", () => {{
          player.dataset.track = button.dataset.track;
          for (const sibling of player.querySelectorAll(".track-button")) {{
            sibling.setAttribute("aria-pressed", String(sibling === button));
          }}
          renderCaption(player);
        }});
      }}
      renderCaption(player);
    }}
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )
    summary = {
        "schema": "timeline_teacher_audit_summary_v1",
        "item_count": len(cards),
        "unit_count": total_units,
        "trainable_unit_count": trainable_units,
        "forced_labels": str(forced_labels),
        "omni_labels": str(omni_labels),
        "fused_labels": str(fused_labels),
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
    parser.add_argument("--forced-labels", required=True)
    parser.add_argument("--omni-labels", required=True)
    parser.add_argument("--fused-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="Timeline Teacher Fusion Audit")
    parser.add_argument("--update-audit-nav", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build_audit(
                forced_labels=_project_path(args.forced_labels),
                omni_labels=_project_path(args.omni_labels),
                fused_labels=_project_path(args.fused_labels),
                output_dir=_project_path(args.output_dir),
                title=args.title,
                update_nav=args.update_audit_nav,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
