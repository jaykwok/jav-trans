from __future__ import annotations

import json
from pathlib import Path

from tools.audits.generate_candidate_island_outside_prompt_matrix import generate


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_outside_prompt_matrix_renders_four_variants_with_one_audio(tmp_path: Path) -> None:
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"wav")
    source = {
        "source_id": "s",
        "frame_count": 10,
        "duration_s": 0.2,
        "audio": str(audio),
        "audio_sha256": "sha",
    }
    manifest = tmp_path / "manifest.jsonl"
    _write(manifest, [source])
    variants: list[tuple[str, Path]] = []
    for index, name in enumerate(("baseline", "v5", "v6", "custom-2")):
        path = tmp_path / f"variant-{index}.jsonl"
        _write(
            path,
            [
                {
                    **source,
                    "prompt_version": name,
                    "islands": [{"start_frame": index, "end_frame": index + 2}],
                    "unsure_spans": [],
                }
            ],
        )
        variants.append((name, path))

    summary = generate(
        manifest=manifest,
        variants=variants,
        output_dir=tmp_path / "out",
        update_nav=False,
    )
    assert summary["source_count"] == 1
    assert summary["variant_count"] == 4
    assert summary["variant_order"] == ["baseline", "v5", "v6", "custom-2"]
    page = (tmp_path / "out" / "index.html").read_text(encoding="utf-8")
    assert page.count("<audio controls") == 1
    assert 'preload="metadata"' in page
    assert "playToken" in page
    assert "waitForMetadata" in page
    assert "audio.play()" in page
    assert "baseline" in page and "custom-2" in page
