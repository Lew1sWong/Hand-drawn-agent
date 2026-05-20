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

import asyncio
import logging
import os
import shelve
import threading
import time as _time
import uuid
from collections import defaultdict, deque
from enum import Enum
from pathlib import Path
from typing import Optional

import aiofiles
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

load_dotenv()  # load .env before any os.environ reads

from agent import AgentResult, run_agent
from bots.feishu_bot import handle_card_action as _feishu_card, handle_event as _feishu_handle

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
# Job store — shelve-backed for persistence across restarts
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


class _ShelveJobStore:
    """Thread-safe shelve-backed job store. Survives process restarts."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()

    def set(self, job: Job) -> None:
        with self._lock, shelve.open(self._path, writeback=False) as db:
            db[job.job_id] = job.model_dump()

    def get(self, job_id: str) -> Job | None:
        with self._lock, shelve.open(self._path, writeback=False) as db:
            raw = db.get(job_id)
        return Job(**raw) if raw else None

    def counts(self) -> dict:
        with self._lock, shelve.open(self._path, writeback=False) as db:
            result = {s.value: 0 for s in JobStatus}
            for raw in db.values():
                result[raw.get("status", JobStatus.failed.value)] += 1
        return result


_job_store = _ShelveJobStore(os.environ.get("JOB_DB_PATH", "jobs.db"))


# ---------------------------------------------------------------------------
# Rate limiting (sliding window, per-IP, no extra dependencies)
# ---------------------------------------------------------------------------

_RATE_WINDOW_S  = int(os.environ.get("RATE_WINDOW_S", "60"))
_RATE_MAX_REQS  = int(os.environ.get("RATE_MAX_REQS", "5"))
_ip_timestamps: defaultdict[str, deque] = defaultdict(deque)


def _check_rate_limit(client_ip: str) -> None:
    now = _time.monotonic()
    dq  = _ip_timestamps[client_ip]
    while dq and dq[0] < now - _RATE_WINDOW_S:
        dq.popleft()
    if len(dq) >= _RATE_MAX_REQS:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {_RATE_MAX_REQS} requests per {_RATE_WINDOW_S}s",
            headers={"Retry-After": str(_RATE_WINDOW_S)},
        )
    dq.append(now)


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
    job = _job_store.get(job_id)
    if job is None:
        logger.error("_run_job: job %s not found in store", job_id)
        return
    job.status = JobStatus.processing
    _job_store.set(job)

    try:
        result: AgentResult = await run_agent(
            user_request=user_request,
            assets=assets,
        )
        job.status          = JobStatus.completed
        job.video_url       = result.video_url
        job.plan            = result.plan
        job.enhanced_prompt = result.enhanced_prompt
        _job_store.set(job)
        logger.info("Job %s completed  video_url=%s", job_id, result.video_url)

    except TimeoutError as exc:
        job.status = JobStatus.failed
        job.error  = f"Timeout: {exc}"
        _job_store.set(job)
        logger.error("Job %s timed out: %s", job_id, exc)

    except RuntimeError as exc:
        job.status = JobStatus.failed
        job.error  = str(exc)
        _job_store.set(job)
        logger.error("Job %s failed: %s", job_id, exc)

    except Exception as exc:
        job.status = JobStatus.failed
        job.error  = f"Unexpected error: {exc}"
        _job_store.set(job)
        logger.exception("Job %s unexpected error", job_id)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class SubmitResponse(BaseModel):
    job_id:  str
    status:  JobStatus
    message: str


@app.on_event("startup")
async def _start_rate_limit_cleanup() -> None:
    async def _cleanup() -> None:
        while True:
            await asyncio.sleep(300)
            cutoff = _time.monotonic() - _RATE_WINDOW_S
            stale = [ip for ip, dq in _ip_timestamps.items() if not dq or dq[-1] < cutoff]
            for ip in stale:
                del _ip_timestamps[ip]
    asyncio.create_task(_cleanup())


@app.post(
    "/animate",
    status_code=202,
    response_model=SubmitResponse,
    summary="Submit an animation job",
)
async def submit_animation(
    request: Request,
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
    _check_rate_limit(request.client.host)

    if not description.strip():
        raise HTTPException(status_code=422, detail="description must not be empty")

    # Resolve image (optional — omit for text-only text_to_video requests)
    resolved_image_url: Optional[str] = None
    if image_url:
        resolved_image_url = image_url
    elif image:
        resolved_image_url = await _store_upload(image, "image")

    # Resolve audio (optional)
    resolved_audio_url: Optional[str] = None
    if audio_url:
        resolved_audio_url = audio_url
    elif audio:
        resolved_audio_url = await _store_upload(audio, "audio")

    # Build assets dict — the planner sees which keys are present and plans accordingly
    assets: dict = {}
    if resolved_image_url:
        assets["image_url"] = resolved_image_url
    if resolved_audio_url:
        assets["audio_url"] = resolved_audio_url

    # Create job record
    job_id = uuid.uuid4().hex
    _job_store.set(Job(job_id=job_id))

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
    job = _job_store.get(job_id)
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


_MEDIA_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif",  ".webp": "image/webp",
    ".mp3": "audio/mpeg", ".mp4": "video/mp4",  ".wav": "audio/wav",
}

@app.get("/media/{filename}", include_in_schema=False)
async def serve_media(filename: str) -> Response:
    """
    Serve locally stored media files.
    Loads the full file into memory before sending so ngrok / proxies
    receive a complete response body (avoids 'unexpected EOF' from Volcengine).
    """
    path = Path(f"/tmp/{filename}")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    suffix = Path(filename).suffix.lower()
    media_type = _MEDIA_TYPES.get(suffix, "application/octet-stream")
    content = path.read_bytes()
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Length": str(len(content)), "Cache-Control": "public, max-age=3600", "bypass-tunnel-reminder": "true"},
    )


_PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://localhost:8000")
_ALLOWED_VIDEO_PREFIXES = (
    "https://tos-cn-",           # Volcengine TOS CDN
    "https://p3-",               # Volcengine media CDN
    "https://lf3-",              # alternative Volcengine CDN
    f"{_PUBLIC_BASE_URL}/media/",# self-hosted /media/ files
)


@app.get("/view", include_in_schema=False, response_class=HTMLResponse)
async def view_video(url: str):
    """Serve a full-screen HTML video player for allowlisted video URLs."""
    if not any(url.startswith(p) for p in _ALLOWED_VIDEO_PREFIXES):
        raise HTTPException(status_code=400, detail="URL origin not allowed")
    from urllib.parse import quote
    safe_url = quote(url, safe=":/?=&%")
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Video</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #000; width: 100vw; height: 100vh; overflow: hidden; display: flex; align-items: center; justify-content: center; }}
video {{ width: 100vw; height: 100vh; object-fit: contain; display: block; }}
</style>
</head>
<body>
<video src="{safe_url}" controls autoplay playsinline></video>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/health")
async def health():
    counts = _job_store.counts()
    return {"status": "ok", **counts}


@app.post("/feishu/event", include_in_schema=False)
async def feishu_event(request: Request):
    """
    Feishu webhook — receives image / audio / text messages and runs the agent.
    Also handles the one-time URL verification challenge from the Feishu console.
    """
    body = await request.json()
    return await _feishu_handle(body)


@app.post("/feishu/card", include_in_schema=False)
async def feishu_card(request: Request):
    """
    Feishu card-action callback — called when users click buttons on interactive cards.
    Register this URL in Feishu console → App Features → Bot → Card Callback URL
    (SEPARATE from the Event Subscription Request URL).

    Supports v1 and v2 (schema: "2.0") Feishu card callback payloads.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    logger.info("Feishu card callback  schema=%s  keys=%s", body.get("schema","v1"), list(body.keys()))
    try:
        result = await _feishu_card(body)
    except Exception as exc:
        logger.exception("Card handler error")
        result = {"toast": {"type": "error", "content": str(exc)[:80]}}
    return result
