from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Query, Response, status

from llm.preflight import translation_config_problems
from web.models import JobSpec, JobState
from web import pipeline_manager as pm


router = APIRouter()


@router.post("/jobs", status_code=status.HTTP_201_CREATED)
async def post_jobs(spec: JobSpec) -> dict[str, list[str]]:
    # Refuse a job whose translation stage cannot possibly run rather than
    # discovering it after ASR has spent ten minutes on the video.
    if not spec.skip_translation:
        problems = translation_config_problems()
        if problems:
            raise HTTPException(status_code=400, detail="\n".join(problems))
    jobs = await pm.create_job(spec)
    return {"ids": [job.id for job in jobs]}


JOBS_REVISION_HEADER = "X-Jobs-Revision"
# Which run of the service issued that revision. Revisions are only comparable
# within one epoch: the counter is resumed from the records that survived, so a
# restart after everything was deleted starts it over, and without the epoch the
# page reads the lower numbers as older news and keeps its own stale list.
JOBS_EPOCH_HEADER = "X-Jobs-Epoch"


@router.get("/jobs", response_model=list[JobState])
async def get_jobs(
    response: Response,
    status: str | None = Query(default=None),
) -> list[JobState]:
    # The header dates the whole list, which a per-job field cannot: an empty
    # list carries no jobs to read a revision from, and "everything was just
    # deleted" and "this response is from before anything existed" look
    # identical without it.
    jobs, revision = await pm.jobs_snapshot(status=status)
    response.headers[JOBS_REVISION_HEADER] = str(revision)
    response.headers[JOBS_EPOCH_HEADER] = str(pm.state_epoch())
    return jobs


@router.get("/jobs/{job_id}", response_model=JobState)
async def get_job_by_id(job_id: str, response: Response) -> JobState:
    job = await pm.get_job(job_id)
    if job is None:
        # 404 answers two different questions with one word: "there was never
        # such a job" and "the one you are still showing is gone". Only the
        # second lets a page drop the card, so a deletion we still remember says
        # so - and says at which reading, in case an older list arrives after.
        deleted_at = pm.deleted_revision(job_id)
        if deleted_at is not None:
            response.headers[JOBS_REVISION_HEADER] = str(deleted_at)
            response.headers[JOBS_EPOCH_HEADER] = str(pm.state_epoch())
            raise HTTPException(
                status_code=410,
                detail="Job deleted",
                headers={
                    JOBS_REVISION_HEADER: str(deleted_at),
                    JOBS_EPOCH_HEADER: str(pm.state_epoch()),
                },
            )
        raise HTTPException(status_code=404, detail="Job not found")
    # The body carries this job's revision; only the epoch it belongs to has
    # nowhere to live but a header.
    response.headers[JOBS_EPOCH_HEADER] = str(pm.state_epoch())
    return job


@router.post("/jobs/{job_id}/retry", response_model=JobState)
async def retry_job(job_id: str) -> JobState:
    job = await pm.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.spec.skip_translation and await asyncio.to_thread(pm.retry_requires_computation, job):
        problems = translation_config_problems()
        if problems:
            raise HTTPException(status_code=400, detail="\n".join(problems))
    retried = await pm.retry_job(job_id)
    if retried is None:
        raise HTTPException(
            status_code=409,
            detail="Only failed, cancelled, or unsaved jobs can be retried",
        )
    return retried


@router.get("/gpu-state")
async def get_gpu_state() -> dict:
    """Whether an unstoppable child still owns the GPU, and the diagnostics."""
    return pm.gpu_cleanup_status()


@router.post("/gpu-state/retry-cleanup")
async def post_gpu_cleanup_retry() -> dict:
    """重试清理 / 重新检查: one attempt, then the state it left behind."""
    return await pm.retry_gpu_cleanup()


@router.delete("/jobs")
async def delete_finished_jobs() -> dict[str, int]:
    removed = await pm.remove_finished_jobs()
    return {"removed": removed}


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str) -> dict[str, bool]:
    ok = await pm.remove_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"ok": True}
