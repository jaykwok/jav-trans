from pathlib import Path

import numpy as np
import pytest

from tools.boundary.ja.edge_frame_dataset import load_edge_row


def test_edge_frame_loader_rejects_silent_length_truncation(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    labels = tmp_path / "labels.npz"
    np.savez(
        source,
        ptm=np.zeros((4, 8), dtype=np.float32),
        mfcc=np.zeros((3, 2), dtype=np.float32),
    )
    np.savez(
        labels,
        labels=np.zeros(3, dtype=np.int64),
        weights=np.ones(3, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="mismatched frame counts"):
        load_edge_row(
            {"source_feature_path": str(source), "feature_path": str(labels)}
        )
