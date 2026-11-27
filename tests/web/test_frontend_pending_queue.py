from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "web" / "static"


def test_selected_files_render_as_pending_jobs_before_start() -> None:
    index = (STATIC / "index.html").read_text(encoding="utf-8")
    files_js = (STATIC / "js" / "files.js").read_text(encoding="utf-8")
    jobs_js = (STATIC / "js" / "jobsRender.js").read_text(encoding="utf-8")

    assert 'id="btn-submit" disabled>开始任务</button>' in index
    assert 'id="file-list"' not in index
    assert "pendingId: `pending-${nextPendingId++}`" in files_js
    assert "renderPendingSelection();" in files_js
    assert "badge badge-pending" in jobs_js
    assert "等待点击“开始任务”" in jobs_js
    assert "data-remove-pending" in jobs_js


def test_start_only_removes_the_pending_entries_that_were_sent() -> None:
    files_js = (STATIC / "js" / "files.js").read_text(encoding="utf-8")

    assert "const pendingEntries = [...state.files];" in files_js
    assert "const pendingIds = new Set(pendingEntries.map(f => f.pendingId));" in files_js
    assert "state.files = state.files.filter(f => !pendingIds.has(f.pendingId));" in files_js
    assert "btnSubmit.textContent = '开始任务';" in files_js


def test_completed_job_renders_minimal_actions_and_forces_full_progress() -> None:
    jobs_js = (STATIC / "js" / "jobsRender.js").read_text(encoding="utf-8")

    assert "job.status === 'done'" in jobs_js
    assert "pct = 100;" in jobs_js
    assert "const terminalStage = CLEARABLE.has(job.status) ? job.status : null;" in jobs_js
    assert "const activeStage = savingStage || terminalStage || job.progress?.stage" in jobs_js
    assert "done:                '已完成'" in jobs_js
    assert "const progressInfo = terminalStage" in jobs_js
    # Minimalist card actions: play, folder reveal, remove (no SRT open/download buttons).
    assert "data-play=" in jobs_js
    assert "data-folder=" in jobs_js
    assert "data-remove=" in jobs_js
    assert "data-open-artifact" not in jobs_js
    assert "btn-open-artifact" not in jobs_js
    assert "data-dl" not in jobs_js
    assert "下载 SRT" not in jobs_js
