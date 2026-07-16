import numpy as np

from boundary.inner_refiner_v1 import InnerEdgeRefinerV1


class _UnexpectedModel:
    def __call__(self, _features):
        raise AssertionError("zero-frame sub-island must abstain before model inference")


def test_zero_frame_subisland_abstains_without_rule_fallback() -> None:
    refiner = InnerEdgeRefinerV1(
        path="inner.pt",
        sha256="sha",
        model=_UnexpectedModel(),
        model_config={"ptm_input_dim": 4},
        feature_config={"raw_ptm_dim": 4},
        normalization={
            "feature_mean": np.zeros(5, dtype=np.float32),
            "feature_std": np.ones(5, dtype=np.float32),
        },
        metadata={},
        device="cpu",
    )

    predictions = refiner.predict_subislands(
        frame_feature_groups=[np.zeros((0, 5), dtype=np.float32)],
        raw_spans=[(1.0, 1.001)],
        frame_hop_s=0.02,
    )

    assert len(predictions) == 1
    assert predictions[0].start_action == "abstain"
    assert predictions[0].end_action == "abstain"
    assert predictions[0].abstain_reason == "no_feature_frames"
    assert predictions[0].start_s == 1.0
    assert predictions[0].end_s == 1.001
