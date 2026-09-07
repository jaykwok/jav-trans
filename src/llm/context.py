"""Source-backed translation units and bounded, request-local context.

Display cues keep their ids and measured times. Reciprocal continuation marks
are the only evidence used to join them into one semantic unit; pauses never
invent speakers or relationships. This index is shared by first pass, repair,
review and cache identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math


def source_text(segment: dict) -> str:
    return str(segment.get("ja_text") or segment.get("text") or segment.get("ja") or "").strip()


def _time(segment: dict, key: str) -> float:
    try:
        value = float(segment.get(key, 0.0))
        return value if math.isfinite(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


def linked(left: dict, right: dict) -> bool:
    return bool(left.get("continues_into_next") and right.get("continues_from_previous"))


def cue_item(segment: dict, index: int) -> dict:
    start, end = _time(segment, "start"), _time(segment, "end")
    item = {
        "id": index,
        "start": round(start, 3),
        "end": round(end, 3),
        "duration_sec": round(max(0.0, end - start), 3),
        "ja": source_text(segment),
    }
    if segment.get("continues_from_previous"):
        item["cont_prev"] = True
    if segment.get("continues_into_next"):
        item["cont_next"] = True
    return item


def split_batches(
    segments: list[dict], batch_size: int, *, max_source_chars: int = 6000
) -> list[list[dict]]:
    """Bound work by both cue count and source size, preferring unit boundaries.

    An oversized unit is split for capacity, but remains a single unit in
    SourceContext. A pause can move the cut only within the last quarter of a
    reasonably full batch; it is a scheduling hint, not a speaker label.
    """
    if not segments:
        return []
    cap = max(1, batch_size) if batch_size > 0 else len(segments)
    batches: list[list[dict]] = []
    start = 0
    while start < len(segments):
        end, chars = start, 0
        while end < len(segments) and end - start < cap:
            size = len(source_text(segments[end]))
            if end > start and max_source_chars > 0 and chars + size > max_source_chars:
                break
            chars += size
            end += 1
        if end < len(segments) and cap > 1:
            unit_start = end
            while unit_start > start and linked(segments[unit_start - 1], segments[unit_start]):
                unit_start -= 1
            if unit_start > start:
                end = unit_start
            floor = start + max(1, (end - start) * 3 // 4)
            pauses = [
                (_time(segments[i], "start") - _time(segments[i - 1], "end"), i)
                for i in range(floor, end)
                if not linked(segments[i - 1], segments[i])
            ]
            if pauses:
                gap, cut = max(pauses)
                if gap >= 2.5:
                    end = cut
        batches.append(segments[start:end])
        start = end
    return batches


@dataclass(frozen=True)
class SourceContext:
    rows: tuple[dict, ...]
    units: tuple[tuple[int, ...], ...]
    unit_for_id: tuple[int, ...]
    signature: str

    @classmethod
    def build(cls, segments: list[dict], *, external_context: str = "") -> SourceContext:
        rows = tuple(cue_item(segment, index) for index, segment in enumerate(segments))
        units: list[list[int]] = []
        unit_for_id: list[int] = []
        for index, segment in enumerate(segments):
            if not units or not linked(segments[index - 1], segment):
                units.append([])
            units[-1].append(index)
            unit_for_id.append(len(units) - 1)
        # Absolute timing adjustments do not change remembered meaning. A gap
        # crossing the context boundary does change which evidence is visible,
        # so keep that fact without keying memory on every timestamp.
        semantic_rows = [
            {
                **{key: row[key] for key in ("id", "ja", "cont_prev", "cont_next") if key in row},
                "context_break": index > 0 and row["start"] - rows[index - 1]["end"] > 12.0,
            }
            for index, row in enumerate(rows)
        ]
        serialized = json.dumps([semantic_rows, external_context], ensure_ascii=False, separators=(",", ":"))
        return cls(
            rows,
            tuple(tuple(unit) for unit in units),
            tuple(unit_for_id),
            hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24],
        )

    def focus(self, ids: list[int], *, radius: int = 2, context_chars: int = 3000) -> dict:
        """Always keep requested cues, then whole units, then nearest neighbours.

        The budget applies to extra source text. No requested text is truncated
        and no partial context sentence is represented as a complete one.
        """
        requested = set(ids)
        if any(index < 0 or index >= len(self.rows) for index in requested):
            raise ValueError("translation context id outside source")
        selected = set(requested)
        candidates: set[int] = set()
        unit_ids = {self.unit_for_id[index] for index in requested}
        for index in requested:
            for neighbour in range(max(0, index - radius), min(len(self.rows), index + radius + 1)):
                low, high = sorted((index, neighbour))
                if all(self.rows[step]["start"] - self.rows[step - 1]["end"] <= 12.0 for step in range(low + 1, high + 1)):
                    candidates.add(neighbour)
        for unit_id in unit_ids:
            candidates.update(self.units[unit_id])
        ordered = sorted(
            candidates - requested,
            key=lambda index: (
                self.unit_for_id[index] not in unit_ids,
                min(abs(index - target) for target in requested),
                index,
            ),
        )
        remaining = max(0, context_chars)
        for index in ordered:
            size = len(self.rows[index]["ja"])
            if size <= remaining:
                selected.add(index)
                remaining -= size
        units = [
            {"ids": list(unit), "ja": "".join(self.rows[index]["ja"] for index in unit)}
            for unit_id in sorted(unit_ids)
            if len(unit := self.units[unit_id]) > 1 and set(unit).issubset(selected)
        ]
        return {
            "items": [
                {**self.rows[index], "role": "translate" if index in requested else "context_only"}
                for index in sorted(selected)
            ],
            "semantic_units": units,
        }
