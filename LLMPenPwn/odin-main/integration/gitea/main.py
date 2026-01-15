import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from analysis import (clientlib_analysis, combiner_analysis, exploit_analysis, preprocess_exploit_analysis,
                      overview_analysis, static_analysis, trigger_analysis)
from fastapi import FastAPI, HTTPException, Query, Request
from gitea import clone_repository_job, search_repositories
from pydantic import BaseModel, Field
from queueing import JobQueue

WORKERS = int(os.getenv("ODIN_WORKERS", "30"))

queue = JobQueue(workers=WORKERS)
queue.register_handler("git-clone", clone_repository_job)
queue.register_handler("static-analysis", static_analysis)
queue.register_handler("combiner-analysis", combiner_analysis)
queue.register_handler("preprocess-exploit-analysis", preprocess_exploit_analysis)
queue.register_handler("exploit-analysis", exploit_analysis)
queue.register_handler("overview-analysis", overview_analysis)
queue.register_handler("clientlib-analysis", clientlib_analysis)
queue.register_handler("repo-trigger", trigger_analysis)

@asynccontextmanager
async def _lifespan(_: FastAPI):
    queue.start()
    try:
        yield
    finally:
        queue.stop()


app = FastAPI(lifespan=_lifespan)

class EnqueueReq(BaseModel):
    job_type: str = Field(default="static-analysis", alias="type")
    priority: int = 100
    depends_on: Optional[List[str]] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    org: Optional[str] = None
    repo: Optional[str] = None

    class Config:
        validate_by_name = True

    def resolved_payload(self) -> Dict[str, Any]:
        payload = dict(self.payload)
        if self.org:
            payload.setdefault("owner", self.org)
        if self.repo:
            payload.setdefault("repo", self.repo)
        return payload

class EnqueueResp(BaseModel):
    job_id: str
    status: str

@app.get("/")
def root(repo: Optional[str] = Query(None), org: Optional[str] = Query(None)):
    if repo is None or org is None:
        return {"service": "odin-queue", "workers": WORKERS, "job_types": queue.job_types()}

    try:
        job_id = queue.enqueue(
            job_type="repo-trigger",
            payload={"owner": org, "repo": repo},
            priority=50,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"job_id": job_id, "status": "queued"}

@app.get("/search")
def search(q: str = Query(..., description="Search repositories")):
    return search_repositories(q)

@app.post("/jobs", response_model=EnqueueResp)
def enqueue_job(req: EnqueueReq):
    payload = req.resolved_payload()
    depends_on: List[str] = list(req.depends_on or [])

    try:
        job_id = queue.enqueue(
            job_type=req.job_type,
            payload=payload,
            priority=req.priority,
            depends_on=depends_on,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"job_id": job_id, "status": "queued"}

@app.get("/jobs")
def list_jobs():
    return queue.list_jobs()

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/traces/{job_id}")
def get_trace(job_id: str):
    tr = queue.get_trace(job_id)
    if tr is None:
        # Either job missing or no trace yet
        job = queue.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        raise HTTPException(status_code=404, detail="Trace not available yet")
    return {"job_id": job_id, "trace": tr}


@app.middleware("http")
async def ignore_upgrade_header(request: Request, call_next):
    scope = request.scope
    # Force protocol to http if proxy incorrectly sets upgrade
    if scope.get("type") == "http":
        headers = []
        for k, v in scope["headers"]:
            if k.lower() not in (b"upgrade", b"connection"):
                headers.append((k, v))
        scope["headers"] = headers

    response = await call_next(request)
    return response
