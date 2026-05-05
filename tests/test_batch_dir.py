import main


def test_batch_dir_cli_dead_code_removed():
    assert not hasattr(main, "_scan_batch_input_dir")
    assert not hasattr(main, "_run_batch_input_dir")
    assert not hasattr(main, "select_video_file")
