import json
from pathlib import Path

import pytest

from tools.asr.cueqc.select_cueqc_v13_stratified_sources import select


def _source(sample_id: str, partition: str, core_id: str) -> dict:
    return {
        "schema": "source-test-v1",
        "sample_id": sample_id,
        "source_partition": partition,
        "audio": f"{sample_id}.wav",
        "core_spans": [{"core_id": core_id, "text": sample_id}],
        "preserved": {"nested": True},
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _inventory() -> list[dict]:
    rows: list[dict] = []
    for partition, size in (("train", 20), ("val", 4), ("test", 3)):
        for index in range(size):
            sample_id = f"{partition}-{index}"
            rows.append(_source(sample_id, partition, f"core-{sample_id}"))
    return rows


def test_selects_85_10_5_and_prioritizes_reusable_sources(tmp_path: Path) -> None:
    manifest = tmp_path / "sources.jsonl"
    reuse = tmp_path / "runtime.jsonl"
    output = tmp_path / "selected.jsonl"
    _write(manifest, _inventory())
    _write(
        reuse,
        [
            {"sample_id": "train-19", "subisland_id": "train-19-a"},
            {"sample_id": "train-19", "subisland_id": "train-19-b"},
            {"sample_id": "val-3"},
            {"sample_id": "test-2"},
            {"sample_id": "not-in-inventory"},
        ],
    )

    summary = select(
        source_manifest=manifest,
        output=output,
        count=20,
        seed=17,
        reuse_jsonl=reuse,
    )
    rows = _read(output)

    assert summary["partition_counts"] == {"train": 17, "val": 2, "test": 1}
    assert summary["selected_count"] == 20
    assert summary["core_count"] == 20
    assert summary["max_core_use"] == 1
    assert summary["reused_count"] == 3
    assert summary["reused_partition_counts"] == {"train": 1, "val": 1, "test": 1}
    assert {"train-19", "val-3", "test-2"} <= {
        row["sample_id"] for row in rows
    }
    assert all(row["preserved"] == {"nested": True} for row in rows)
    assert json.loads(output.with_suffix(".summary.json").read_text())["seed"] == 17


def test_selection_is_deterministic_for_seed(tmp_path: Path) -> None:
    manifest = tmp_path / "sources.jsonl"
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    third = tmp_path / "third.jsonl"
    _write(manifest, _inventory())

    select(source_manifest=manifest, output=first, count=20, seed=3)
    select(source_manifest=manifest, output=second, count=20, seed=3)
    select(source_manifest=manifest, output=third, count=20, seed=4)

    assert first.read_text() == second.read_text()
    assert first.read_text() != third.read_text()


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [_source("same", "train", "core-a"), _source("same", "val", "core-b")],
            "duplicate sample_id",
        ),
        (
            [_source("a", "train", "same-core"), _source("b", "val", "same-core")],
            "duplicate core_id",
        ),
    ],
)
def test_rejects_non_unique_source_identity(
    tmp_path: Path, rows: list[dict], message: str
) -> None:
    manifest = tmp_path / "sources.jsonl"
    _write(manifest, rows)

    with pytest.raises(ValueError, match=message):
        select(source_manifest=manifest, output=tmp_path / "out.jsonl", count=1)


def test_rejects_insufficient_partition_inventory(tmp_path: Path) -> None:
    manifest = tmp_path / "sources.jsonl"
    rows = [
        *[_source(f"train-{i}", "train", f"train-core-{i}") for i in range(17)],
        _source("val-0", "val", "val-core-0"),
        _source("test-0", "test", "test-core-0"),
    ]
    _write(manifest, rows)

    with pytest.raises(ValueError, match="insufficient val inventory: need 2, found 1"):
        select(source_manifest=manifest, output=tmp_path / "out.jsonl", count=20)
