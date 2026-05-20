"""Shared Volcengine VisualService helpers — submit/poll, status checks, URL extraction."""
from __future__ import annotations

import asyncio
import json
import logging
import os

from volcengine.visual.VisualService import VisualService

logger = logging.getLogger(__name__)

_STATUS_DONE   = "done"
_STATUS_FAILED = "failed"
_POLL_INTERVAL = 5      # seconds between status checks
_MAX_TRIES     = 90     # 90 × 5 s = 7.5 min ceiling


def _make_svc(socket_timeout: int = 60) -> VisualService:
    svc = VisualService()
    svc.set_ak(os.environ["VOLC_ACCESSKEY"])
    svc.set_sk(os.environ["VOLC_SECRETKEY"])
    svc.set_connection_timeout(15)
    svc.set_socket_timeout(socket_timeout)
    return svc


def _assert_ok(resp, ctx: str) -> None:
    if isinstance(resp, bytes):
        try:
            resp = json.loads(resp.decode("utf-8"))
        except Exception:
            raise RuntimeError(f"[{ctx}] SDK returned unexpected bytes: {resp[:300]}")
    code = resp.get("code")
    if code != 10000:
        raise RuntimeError(
            f"[{ctx}] code={code} msg='{resp.get('message')}' req_id={resp.get('request_id')}"
        )


def _extract_video_url(data: dict) -> str:
    if url := data.get("video_url"):
        return url
    for info in data.get("video_infos") or []:
        if url := info.get("video_url"):
            return url
    raw = data.get("resp_data", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if url := parsed.get("video_url"):
        return url
    raise RuntimeError(f"No video_url found in response data: {data}")


_POLL_TRANSPORT_RETRIES = 2  # 2 retries × 60s socket timeout = at most 120s overhead per attempt


async def _poll_loop(
    task_id: str,
    req_key: str,
    label: str,
    use_legacy: bool,
    max_tries: int = _MAX_TRIES,
    poll_interval: int = _POLL_INTERVAL,
) -> dict:
    loop = asyncio.get_running_loop()
    poll_body = {"req_key": req_key, "task_id": task_id}
    for attempt in range(1, max_tries + 1):
        # Retry transient transport errors (network blips) with exponential backoff.
        # API-level failures (_assert_ok) are not retried here.
        for _t in range(_POLL_TRANSPORT_RETRIES):
            try:
                if use_legacy:
                    poll_resp = await loop.run_in_executor(
                        None, lambda: _make_svc().cv_get_result(poll_body)
                    )
                else:
                    poll_resp = await loop.run_in_executor(
                        None, lambda: _make_svc().cv_sync2async_get_result(poll_body)
                    )
                break
            except Exception as transport_exc:
                if _t == _POLL_TRANSPORT_RETRIES - 1:
                    logger.error("%s poll transport error (giving up): %s", label, transport_exc)
                    raise
                wait = 2 ** _t
                logger.warning(
                    "%s poll transport error (retry %d/%d in %ds): %s",
                    label, _t + 1, _POLL_TRANSPORT_RETRIES, wait, transport_exc,
                )
                await asyncio.sleep(wait)
        _assert_ok(poll_resp, f"{label}/poll")
        data = poll_resp["data"]
        status = data.get("status")
        if status == _STATUS_DONE:
            logger.info("%s done  task_id=%s", label, task_id)
            return {"task_id": task_id, "data": data}
        if status == _STATUS_FAILED:
            reason = data.get("message") or data.get("err_msg", str(status))
            raise RuntimeError(f"{label} failed  task_id={task_id}  reason={reason}")
        logger.debug("%s attempt=%d/%d status=%s", label, attempt, max_tries, status)
        await asyncio.sleep(poll_interval)
    raise TimeoutError(
        f"{label} task_id={task_id} timed out after {max_tries * poll_interval}s"
    )


async def _submit_and_poll(
    body: dict,
    label: str,
    max_tries: int = _MAX_TRIES,
    poll_interval: int = _POLL_INTERVAL,
) -> dict:
    """
    Submit a Volcengine async task and poll until done.
    Tries cv_sync2async_submit_task first; falls back to legacy cv_submit_task on 504.
    Returns {"task_id": str, "data": dict} on success.
    """
    loop = asyncio.get_running_loop()

    try:
        resp = await loop.run_in_executor(
            None, lambda: _make_svc().cv_sync2async_submit_task(body)
        )
        if isinstance(resp, bytes) and b"504" in resp:
            raise RuntimeError("504")
        _assert_ok(resp, f"{label}/submit")
        task_id = resp["data"]["task_id"]
        logger.info("%s submitted (sync2async)  task_id=%s", label, task_id)
        return await _poll_loop(task_id, body["req_key"], label, use_legacy=False,
                                max_tries=max_tries, poll_interval=poll_interval)

    except RuntimeError as exc:
        if "504" not in str(exc):
            raise
        logger.warning(
            "%s: cv_sync2async_submit_task returned 504 — falling back to cv_submit_task",
            label,
        )

    try:
        resp = await loop.run_in_executor(
            None, lambda: _make_svc(socket_timeout=120).cv_submit_task(body)
        )
    except Exception as e:
        raw = e.args[0] if e.args else b""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        raise RuntimeError(f"[{label}/submit-legacy] {raw}") from None

    _assert_ok(resp, f"{label}/submit-legacy")
    task_id = resp["data"]["task_id"]
    logger.info("%s submitted (legacy)  task_id=%s", label, task_id)
    return await _poll_loop(task_id, body["req_key"], label, use_legacy=True,
                            max_tries=max_tries, poll_interval=poll_interval)
