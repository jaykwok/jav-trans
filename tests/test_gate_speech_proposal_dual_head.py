from tools.boundary.ja.gate_speech_proposal_dual_head import match_times


def test_match_times_is_one_to_one_with_frame_tolerance() -> None:
    assert match_times([1.00, 1.02, 2.0], [1.01, 2.03], tolerance_s=0.02) == 1
    assert match_times([1.00, 1.02], [1.01, 1.03], tolerance_s=0.02) == 2
