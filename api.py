"""
Hand-drawn Animation Agent — FastAPI Web Layer
===============================================
POST /animate            — submit a job (202 Accepted, returns job_id immediately)
GET  /animate/{job_id}   — poll job status / retrieve video URL + plan
GET  /media/{filename}   — serve locally stored media (Telegram / Feishu bots)
POST /feishu/event       — Feishu webhook (image / audio / text → animation)

Async design:
  - Heavy lifting (plan → execute) runs in a BackgroundTask.
  - Job state is in-memory (swap for Redis in production).
  - Image upload is stubbed to /tmp; replace _store_image() with TOS/S3 in production.
"""

from __future__ import annotations

import logging
import os
import uuid
from enum import Enum
from pathlib import Path
from typing import Optional

import aiofiles
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()  # load .env before any os.environ reads

from agent import AgentResult, run_agent
from bots.feishu_bot import handle_event as _feishu_handle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hand-drawn Animation Agent",
    version="2.0.0",
    description=(
        "Upload a hand-drawn sketch (and optionally audio) + description. "
        "The agent plans and executes the right Volcengine tools automatically."
    ),
)


# ---------------------------------------------------------------------------
# In-memory job store  (replace with Redis + Celery for production)
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    pending    = "pending"
    processing = "processing"
    completed  = "completed"
    failed     = "failed"


class Job(BaseModel):
    job_id:          str
    status:          JobStatus       = JobStatus.pending
    video_url:       Optional[str]   = None
    plan:            list[dict]      = []
    enhanced_prompt: Optional[str]   = None
    error:           Optional[str]   = None


_jobs: dict[str, Job] = {}


# ---------------------------------------------------------------------------
# Image / audio storage stub  (replace with TOS/S3 in production)
# ---------------------------------------------------------------------------

async def _store_upload(upload: UploadFile, media_type: str = "image") -> str:
    """
    Save an uploaded file to /tmp and return a mock public URL.
    In production: upload to Volcengine TOS and return the signed URL.
    The Volcengine API must be able to fetch this URL from the public internet.
    """
    suffix   = os.path.splitext(upload.filename or f"upload.bin")[1] or ".bin"
    filename = f"{uuid.uuid4().hex}{suffix}"
    tmp_path = f"/tmp/{filename}"

    async with aiofiles.open(tmp_path, "wb") as f:
        await f.write(await upload.read())

    # TODO: replace with real cloud upload + public URL
    public_url = f"https://your-cdn.example.com/uploads/{filename}"
    logger.info("%s stored locally at %s  (public_url=%s)", media_type, tmp_path, public_url)
    return public_url


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

async def _run_job(
    job_id: str,
    user_request: str,
    assets: dict,
) -> None:
    job = _jobs[job_id]
    job.status = JobStatus.processing

    try:
        result: AgentResult = await run_agent(
            user_request=user_request,
            assets=assets,
        )
        job.status          = JobStatus.completed
        job.video_url       = result.video_url
        job.plan            = result.plan
        job.enhanced_prompt = result.enhanced_prompt
        logger.info("Job %s completed  video_url=%s", job_id, result.video_url)

    except TimeoutError as exc:
        job.status = JobStatus.failed
        job.error  = f"Timeout: {exc}"
        logger.error("Job %s timed out: %s", job_id, exc)

    except RuntimeError as exc:
        job.status = JobStatus.failed
        job.error  = str(exc)
        logger.error("Job %s failed: %s", job_id, exc)

    except Exception as exc:
        job.status = JobStatus.failed
        job.error  = f"Unexpected error: {exc}"
        logger.exception("Job %s unexpected error", job_id)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class SubmitResponse(BaseModel):
    job_id:  str
    status:  JobStatus
    message: str


@app.post(
    "/animate",
    status_code=202,
    response_model=SubmitResponse,
    summary="Submit an animation job",
)
async def submit_animation(
    background_tasks: BackgroundTasks,
    description: str                    = Form(...,  description="Natural-language animation description (any language)"),
    image: Optional[UploadFile]         = File(None, description="Hand-drawn sketch image (PNG/JPG)"),
    image_url: Optional[str]            = Form(None, description="Public URL of the sketch (alternative to file upload)"),
    audio: Optional[UploadFile]         = File(None, description="Audio file for lip-sync (optional, enables audio_portrait tool)"),
    audio_url: Optional[str]            = Form(None, description="Public URL of audio file (alternative to file upload)"),
):
    """
    Submit a hand-drawn animation job.

    Supply either `image` (file upload) or `image_url` (public URL) — not both.
    Optionally supply `audio` / `audio_url` to enable the audio-driven portrait tool.

    Returns **202 Accepted** with a `job_id`.
    Poll `GET /animate/{job_id}` to track progress and retrieve the video URL.
    """
    if not description.strip():
        raise HTTPException(status_code=422, detail="description must not be empty")

    # Resolve image
    if image_url:
        resolved_image_url = image_url
    elif image:
        resolved_image_url = await _store_upload(image, "image")
    else:
        raise HTTPException(status_code=422, detail="Provide either 'image' file or 'image_url'")

    # Resolve audio (optional)
    resolved_audio_url: Optional[str] = None
    if audio_url:
        resolved_audio_url = audio_url
    elif audio:
        resolved_audio_url = await _store_upload(audio, "audio")

    # Build assets dict — the planner sees which keys are present and plans accordingly
    assets: dict = {"image_url": resolved_image_url}
    if resolved_audio_url:
        assets["audio_url"] = resolved_audio_url

    # Create job record
    job_id = uuid.uuid4().hex
    _jobs[job_id] = Job(job_id=job_id)

    background_tasks.add_task(_run_job, job_id, description, assets)
    logger.info("Job %s queued  description='%s'  assets=%s", job_id, description, list(assets))

    return SubmitResponse(
        job_id=job_id,
        status=JobStatus.pending,
        message="Job accepted. Poll GET /animate/{job_id} for status.",
    )


class StatusResponse(BaseModel):
    job_id:          str
    status:          JobStatus
    video_url:       Optional[str]  = None
    plan:            list[dict]     = []
    enhanced_prompt: Optional[str]  = None
    error:           Optional[str]  = None


@app.get(
    "/animate/{job_id}",
    response_model=StatusResponse,
    summary="Poll animation job status",
)
async def get_animation_status(job_id: str):
    """
    Returns the current status of an animation job.

    - **pending / processing** — still running, poll again in a few seconds
    - **completed** — `video_url` is populated, `plan` shows what tools ran
    - **failed** — `error` describes what went wrong
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    return StatusResponse(
        job_id=job.job_id,
        status=job.status,
        video_url=job.video_url,
        plan=job.plan,
        enhanced_prompt=job.enhanced_prompt,
        error=job.error,
    )


@app.get("/media/{filename}", include_in_schema=False)
async def serve_media(filename: str) -> FileResponse:
    """
    Serve locally stored media files (images/audio saved from bot uploads).
    In production: remove this and serve from TOS/S3 directly.
    """
    path = Path(f"/tmp/{filename}")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


@app.get("/health")
async def health():
    counts = {s.value: 0 for s in JobStatus}
    for j in _jobs.values():
        counts[j.status.value] += 1
    return {"status": "ok", **counts}


@app.post("/feishu/event", include_in_schema=False)
async def feishu_event(request: Request):
    """
    Feishu webhook — receives image / audio / text messages and runs the agent.
    Also handles the one-time URL verification challenge from the Feishu console.
    """
    body = await request.json()
    return await _feishu_handle(body)
