from __future__ import annotations

import asyncio
import logging

from tools.base import BaseTool, ToolContract
from tools._volcengine import (
    _assert_ok, _extract_video_url, _make_svc,
    _POLL_INTERVAL, _MAX_TRIES, _STATUS_DONE, _STATUS_FAILED,
)

logger = logging.getLogger(__name__)


class AudioPortraitTool(BaseTool):
    name = "audio_portrait"
    description = (
        "Creates a talking-head / lip-sync video (OmniHuman 1.5) from a portrait photo and audio. "
        "Requires context keys: image_url (str), audio_url (str). "
        "Optional: tts_text used as style prompt. "
        "Produces: video_url."
    )
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    contract = ToolContract(
        reads          = ["image_url", "audio_url"],
        writes         = ["video_url", "task_id"],
        optional_reads = ["tts_text", "user_description"],
    )

    _REQ_KEY_DETECT = "jimeng_realman_avatar_object_detection"
    _REQ_KEY_VIDEO  = "jimeng_realman_avatar_picture_omni_v15"

    async def run(self, ctx: dict) -> dict:
        loop = asyncio.get_running_loop()

        # Step A: subject detection → mask URLs (best-effort)
        mask_url: list[str] = []
        try:
            detect_body = {
                "req_key":   self._REQ_KEY_DETECT,
                "image_url": ctx["image_url"],
            }
            detect_resp = await loop.run_in_executor(
                None, lambda: _make_svc().cv_process(detect_body)
            )
            _assert_ok(detect_resp, "AudioPortrait/detect")
            detect_data = detect_resp.get("data", {})
            logger.info("AudioPortrait detection response keys: %s", list(detect_data.keys()))
            masks = (
                detect_data.get("masks")
                or detect_data.get("mask_urls")
                or detect_data.get("mask_url")
                or []
            )
            if isinstance(masks, str):
                masks = [masks]
            if masks:
                mask_url = [masks[0]]
            logger.info("AudioPortrait detection done, masks=%d", len(mask_url))
        except Exception as exc:
            logger.warning("AudioPortrait detection step failed (%s) — continuing with empty mask", exc)

        # Step B: submit video generation task
        video_body = {
            "req_key":   self._REQ_KEY_VIDEO,
            "image_url": ctx["image_url"],
            "mask_url":  mask_url,
            "audio_url": ctx["audio_url"],
            "prompt":    ctx.get("tts_text") or ctx.get("user_description", ""),
        }
        try:
            submit_resp = await loop.run_in_executor(
                None, lambda: _make_svc(socket_timeout=60).cv_submit_task(video_body)
            )
        except Exception as e:
            raw = e.args[0] if e.args else b""
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            if "504" in str(raw) or "timed out" in str(raw).lower():
                raise RuntimeError(
                    "OmniHuman video generation timed out (504). "
                    "Please verify the OmniHuman 1.5 service is fully activated in the "
                    "Volcengine console (即梦AI → 数字人 → 服务开通 → 确认已开通状态)."
                ) from None
            raise RuntimeError(f"[AudioPortrait/submit] {raw}") from None

        _assert_ok(submit_resp, "AudioPortrait/submit")
        task_id = submit_resp["data"]["task_id"]
        logger.info("AudioPortrait submitted task_id=%s", task_id)

        # AudioPortrait typically completes in 30–90s; cap at 3 min (36 × 5s)
        _AUDIO_PORTRAIT_MAX_TRIES = 36
        poll_body = {"req_key": self._REQ_KEY_VIDEO, "task_id": task_id}
        for attempt in range(1, _AUDIO_PORTRAIT_MAX_TRIES + 1):
            poll_resp = await loop.run_in_executor(
                None, lambda: _make_svc().cv_get_result(poll_body)
            )
            _assert_ok(poll_resp, "AudioPortrait/poll")
            data = poll_resp["data"]
            status = data.get("status")

            if status == _STATUS_DONE:
                logger.info("AudioPortrait done task_id=%s", task_id)
                return {
                    "video_url": _extract_video_url(data),
                    "task_id":   task_id,
                }
            if status == _STATUS_FAILED:
                reason = data.get("message") or data.get("err_msg", str(status))
                raise RuntimeError(f"AudioPortrait failed task_id={task_id} reason={reason}")

            logger.debug("AudioPortrait attempt=%d/%d status=%s", attempt, _AUDIO_PORTRAIT_MAX_TRIES, status)
            await asyncio.sleep(_POLL_INTERVAL)

        raise TimeoutError(
            f"AudioPortrait task_id={task_id} timed out after {_AUDIO_PORTRAIT_MAX_TRIES * _POLL_INTERVAL}s"
        )
