from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from web.models import JobSpec, JobState
from web import pipeline_manager as pm


router = APIRouter()


@router.post("/jobs", status_code=status.HTTP_201_CREATED)
async def post_jobs(spec: JobSpec) -> dict[str, list[str]]:
    jobs = await pm.create_job(spec)
    return {"ids": [job.id for job in jobs]}


@router.get("/jobs", response_model=list[JobState])
async def get_jobs(status: str | None = Query(default=None)) -> list[JobState]:
    return await pm.list_jobs(status=status)


@router.get("/jobs/{job_id}", response_model=JobState)
async def get_job_by_id(job_id: str) -> JobState:
    job = await pm.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/jobs/{job_id}/retry", response_model=JobState)
async def retry_job(job_id: str) -> JobState:
    job = await pm.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    retried = await pm.retry_job(job_id)
    if retried is None:
        raise HTTPException(
            status_code=409,
            detail="Only failed or cancelled jobs can be retried",
        )
    return retried


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
