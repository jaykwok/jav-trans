"""Disk-backed sequence dataset store.

Sequence NPZ datasets carry a multi-GB ``frame_features`` array that a 16GB-RAM
box cannot materialize (build-time concatenate, merge and training load each
need the whole array resident). This module moves frames into a sidecar
``<stem>.frames.npy`` written incrementally and opened with ``mmap_mode="r"``;
the ``.npz`` keeps every small array plus a format marker. Legacy in-npz
datasets keep loading unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

FRAMES_STORAGE_KEY = "frame_features_storage"
FRAMES_STORAGE_SIDECAR = "sidecar_npy_v1"
FRAMES_SUFFIX = ".frames.npy"
_COPY_CHUNK_ROWS = 16384


def frames_sidecar_path(dataset_path: Path) -> Path:
    dataset_path = Path(dataset_path)
    name = dataset_path.name
    if name.endswith(".npz"):
        name = name[: -len(".npz")]
    return dataset_path.with_name(name + FRAMES_SUFFIX)


class StreamingFrameWriter:
    """Append float32 frame rows and finalize into a memmap-able ``.npy``.

    Rows stream to a raw ``.tmp`` file so peak RAM stays at one append's size;
    ``finalize()`` chunk-copies into a proper ``.npy`` (disk-to-disk).
    """

    def __init__(self, dataset_path: Path) -> None:
        self.sidecar = frames_sidecar_path(dataset_path)
        self.sidecar.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = self.sidecar.with_suffix(".npy.tmp")
        self._handle = self._tmp.open("wb")
        self._rows = 0
        self._row_shape: tuple[int, ...] | None = None

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._rows, *(self._row_shape or ()))

    def append(self, rows: np.ndarray) -> None:
        values = np.ascontiguousarray(rows, dtype=np.float32)
        if values.ndim == 2:
            values = values[None]
        if values.ndim != 3:
            raise ValueError(f"expected (rows, bins, dim) frames, got {values.shape}")
        if self._row_shape is None:
            self._row_shape = tuple(values.shape[1:])
        elif tuple(values.shape[1:]) != self._row_shape:
            raise ValueError(
                f"frame row shape changed: {values.shape[1:]} != {self._row_shape}"
            )
        self._handle.write(values.tobytes())
        self._rows += int(values.shape[0])

    def finalize(self) -> tuple[Path, tuple[int, ...]]:
        self._handle.close()
        if self._rows == 0 or self._row_shape is None:
            self._tmp.unlink(missing_ok=True)
            raise ValueError("no frame rows were appended")
        shape = (self._rows, *self._row_shape)
        target = np.lib.format.open_memmap(
            self.sidecar, mode="w+", dtype=np.float32, shape=shape
        )
        row_values = int(np.prod(self._row_shape))
        with self._tmp.open("rb") as source:
            for start in range(0, self._rows, _COPY_CHUNK_ROWS):
                count = min(_COPY_CHUNK_ROWS, self._rows - start)
                block = np.frombuffer(
                    source.read(count * row_values * 4), dtype=np.float32
                ).reshape(count, *self._row_shape)
                target[start : start + count] = block
        target.flush()
        del target
        self._tmp.unlink(missing_ok=True)
        return self.sidecar, shape


def save_sequence_dataset(
    dataset_path: Path,
    *,
    frames: np.ndarray | None = None,
    frames_finalized: bool = False,
    compress: bool = False,
    **arrays: np.ndarray,
) -> None:
    """Write the small-array ``.npz`` half of a sidecar-format dataset.

    Either pass ``frames`` (written to the sidecar in chunks) or set
    ``frames_finalized`` after a :class:`StreamingFrameWriter` already wrote it.
    """

    dataset_path = Path(dataset_path)
    if frames is not None:
        writer = StreamingFrameWriter(dataset_path)
        values = np.ascontiguousarray(frames, dtype=np.float32)
        for start in range(0, values.shape[0], _COPY_CHUNK_ROWS):
            writer.append(values[start : start + _COPY_CHUNK_ROWS])
        writer.finalize()
    elif not frames_finalized:
        raise ValueError("provide frames or finalize a StreamingFrameWriter first")
    save = np.savez_compressed if compress else np.savez
    save(
        dataset_path,
        **{FRAMES_STORAGE_KEY: np.asarray(FRAMES_STORAGE_SIDECAR)},
        **arrays,
    )


def open_frames_memmap_for_write(
    dataset_path: Path, *, rows: int, row_shape: tuple[int, ...]
) -> np.memmap:
    """Pre-sized sidecar for writers that know the total row count upfront."""

    sidecar = frames_sidecar_path(dataset_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(
        sidecar, mode="w+", dtype=np.float32, shape=(rows, *row_shape)
    )


def load_sequence_arrays(dataset_path: Path) -> dict[str, Any]:
    """Load a sequence dataset; frames come back memmap-backed when sidecar-format.

    Legacy datasets (frames inside the npz) load exactly as before. Returned
    dict holds plain arrays for every npz key plus ``frame_features``.
    """

    dataset_path = Path(dataset_path)
    bundle = np.load(dataset_path, allow_pickle=False)
    arrays: dict[str, Any] = {}
    sidecar_format = False
    for key in bundle.files:
        if key == FRAMES_STORAGE_KEY:
            if str(bundle[key]) != FRAMES_STORAGE_SIDECAR:
                raise ValueError(
                    f"unknown frame storage {bundle[key]!r}: {dataset_path}"
                )
            sidecar_format = True
            continue
        arrays[key] = bundle[key]
    if sidecar_format:
        if "frame_features" in arrays:
            raise ValueError(
                f"sidecar-format dataset also embeds frame_features: {dataset_path}"
            )
        sidecar = frames_sidecar_path(dataset_path)
        if not sidecar.exists():
            raise FileNotFoundError(
                f"missing frames sidecar {sidecar} for {dataset_path}"
            )
        arrays["frame_features"] = np.load(sidecar, mmap_mode="r")
    return arrays


def chunked_frame_stats(
    frames: Any, rows: np.ndarray, *, chunk_rows: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std over (rows, bins) without materializing all rows."""

    rows = np.sort(np.asarray(rows, dtype=np.int64))
    dim = int(frames.shape[2])
    if chunk_rows <= 0:
        # Budget ~256MB per float64 block: a fixed row count sized for 168-dim
        # frames blows past physical RAM on full-dim (2088) input.
        row_bytes = int(frames.shape[1]) * dim * 8
        chunk_rows = max(256, (256 * 1024 * 1024) // max(1, row_bytes))
    total = 0
    running_sum = np.zeros(dim, dtype=np.float64)
    running_sq = np.zeros(dim, dtype=np.float64)
    for start in range(0, rows.size, chunk_rows):
        block = np.asarray(frames[rows[start : start + chunk_rows]], dtype=np.float64)
        running_sum += block.sum(axis=(0, 1))
        # einsum avoids np.square's full-block float64 temporary.
        running_sq += np.einsum("ijk,ijk->k", block, block)
        total += block.shape[0] * block.shape[1]
    mean = running_sum / max(1, total)
    variance = np.maximum(running_sq / max(1, total) - np.square(mean), 0.0)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def transform_window_npz(
    paths,
    transform,
    *,
    workers: int = 6,
    compress: bool = True,
):
    """Load -> transform -> save many per-window npz concurrently.

    Window feature files (``semantic_split_features.npz`` etc.) are zlib-
    compressed; the decompress/compress C calls release the GIL, so a thread
    pool turns the single-threaded zlib bottleneck into near-linear speedup on
    multi-core boxes. ``transform`` receives a dict of every materialized array
    and returns the dict to write back. Returns the count of files written.

    Use this for one-off feature transforms (slicing PTM dims, applying a
    learned projection, re-binning) instead of a serial ``for path in paths``
    loop — the recurring IO pattern that ran ~10 min single-threaded drops to
    seconds here.
    """
    from concurrent.futures import ThreadPoolExecutor

    paths = [Path(path) for path in paths]
    save = np.savez_compressed if compress else np.savez

    def _one(path: Path) -> bool:
        with np.load(path) as handle:
            arrays = {key: np.asarray(handle[key]) for key in handle.files}
        out = transform(arrays)
        if out is None:
            return False
        save(path, **out)
        return True

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(_one, paths))
    return sum(1 for value in results if value)
