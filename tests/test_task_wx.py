import json

from asr import pipeline as asr


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

    monkeypatch.setenv("ASR_CHUNK_ROOT", str(chunk_root))

    summary_path = asr.aggregate_timeout_fragments(job_id)

    assert summary_path == timeout_dir / f"timeouts_summary_{job_id}.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["job_id"] == job_id
    assert payload["count"] == 3
    assert [record["chunk_index"] for record in payload["records"]] == [0, 1, 2]
    assert all(not path.exists() for path in fragments)
    assert not (tmp_path / "rm").exists()

