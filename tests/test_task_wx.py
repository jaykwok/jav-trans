import json
import multiprocessing as mp

from whisper import pipeline as asr
from whisper import worker as asr_worker
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
    monkeypatch.setattr(asr, "_agents_RM_ROOT", tmp_path / "rm")

    summary_path = asr.aggregate_timeout_fragments(job_id)

    assert summary_path == timeout_dir / f"timeouts_summary_{job_id}.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["job_id"] == job_id
    assert payload["count"] == 3
    assert [record["chunk_index"] for record in payload["records"]] == [0, 1, 2]
    assert all(not path.exists() for path in fragments)
    assert len(list((tmp_path / "rm").glob("timeouts_*.json.*"))) == 3


def test_asr_worker_unknown_op_exits_with_code_1(monkeypatch):
    parent_conn, child_conn = mp.Pipe()
    monkeypatch.setenv("ASR_WORKER_MOCK", "1")
    process = mp.Process(
        target=asr_worker.main,
        args=(child_conn, {"device": "cpu"}),
    )
    process.start()
    try:
        ready = parent_conn.recv()
        assert ready["op"] == "ready"

        parent_conn.send({"op": "definitely_unknown", "job_id": "job-x"})
        response = parent_conn.recv()
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

