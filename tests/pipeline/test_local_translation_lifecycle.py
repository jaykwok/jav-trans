from __future__ import annotations

import pytest

import main


@pytest.mark.parametrize("fails", [False, True])
def test_translation_task_always_closes_local_backend(monkeypatch, fails):
    calls: list[str] = []
    closed: list[object] = []
    artifacts = object()

    monkeypatch.setattr(main, "_reopen_snapshot_run_logger", lambda _artifacts: None)
    monkeypatch.setattr(
        main,
        "_close_artifacts_logger",
        lambda value: closed.append(value),
    )
    monkeypatch.setattr(
        main.llm_backends,
        "reset_backend",
        lambda name=None: calls.append(str(name or "")),
    )

    def run_impl(*_args, **_kwargs):
        if fails:
            raise RuntimeError("translation failed")
        return ["done.srt"]

    monkeypatch.setattr(main, "_run_translation_and_write_impl", run_impl)

    if fails:
        with pytest.raises(RuntimeError, match="translation failed"):
            main.run_translation_and_write("video.mp4", artifacts, ctx=object())
    else:
        assert main.run_translation_and_write(
            "video.mp4", artifacts, ctx=object()
        ) == ["done.srt"]

    assert calls == ["llamacpp"]
    assert closed == [artifacts]
