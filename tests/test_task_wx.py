import json
import multiprocessing as mp

from asr import pipeline as asr
from asr import worker as asr_worker


def _recv_with_timeout(conn, timeout_s: float = 10.0):
    assert conn.poll(timeout_s), "timed out waiting for ASR worker response"
    return conn.recv()


def test_aggregate_timeout_fragments_writes_summary_and_removes_fragments(monkeypatch, tmp_path):
    chunk_root = tmp_path / "chunks"
    timeout_dir = tmp_path / "asr_timeouts"
    timeout_dir.mkdir(parents=True)
    job_id = "sample_job"
    fragments = []
    for index in range(3):
        path = timeout_dir / f"timeouts_{index:02d}_run.json"
        path.write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "chunk_index": index,
                    "audio_path": f"work/{job_id}/audio/chunk_{index}.wav",
                }
            ),
            encoding="utf-8",
        )
        fragments.append(path)

    monkeypatch.setattr(asr, "_ASR_CHUNK_ROOT", chunk_root)

    summary_path = asr.aggregate_timeout_fragments(job_id)

    assert summary_path == timeout_dir / f"timeouts_summary_{job_id}.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["job_id"] == job_id
    assert payload["count"] == 3
    assert [record["chunk_index"] for record in payload["records"]] == [0, 1, 2]
    assert all(not path.exists() for path in fragments)
    assert not (tmp_path / "rm").exists()


def test_asr_worker_unknown_op_exits_with_code_1(monkeypatch):
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    monkeypatch.setenv("ASR_WORKER_MOCK", "1")
    process = ctx.Process(
        target=asr_worker.main,
        args=(child_conn, {"device": "cpu"}),
    )
    process.start()
    child_conn.close()
    try:
        ready = _recv_with_timeout(parent_conn)
        assert ready["op"] == "ready"

        parent_conn.send({"op": "definitely_unknown", "job_id": "job-x"})
        response = _recv_with_timeout(parent_conn)
        assert response["op"] == "error"
        assert response["kind"] == "protocol_error"
        assert "unknown op" in response["detail"]

        process.join(timeout=10)
        assert process.exitcode == 1
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        parent_conn.close()
        child_conn.close()

