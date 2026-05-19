"""
Tool registry — one class per Volcengine capability.

Each tool:
  • has a `name`, `description`, and `input_schema` consumed by the planner
  • implements  async run(ctx: dict) -> dict
  • handles its own submit + poll loop internally

Shared context dict keys produced by each tool:
  image_to_video  → video_url, task_id
  audio_portrait  → video_url, task_id
  video_effects   → video_url, task_id

TODO: verify all req_key values against live docs before going to production:
  https://www.volcengine.com/docs/85621/1783678   (image_to_video)
  https://www.volcengine.com/docs/86081/1804513   (audio_portrait)
  https://www.volcengine.com/docs/86081/1804543   (video_effects)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import edge_tts
from volcengine.visual.VisualService import VisualService

PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared Volcengine helpers
# ---------------------------------------------------------------------------

_STATUS_DONE    = "done"
_STATUS_FAILED  = "failed"
_POLL_INTERVAL  = 5     # seconds between status checks
_MAX_TRIES      = 180   # 180 × 5 s = 15 min ceiling


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


async def _submit_and_poll(body: dict, label: str) -> dict:
    """
    Submit a Volcengine async task and poll until done.
    Returns {"task_id": str, "data": dict} on success.
    """
    loop = asyncio.get_running_loop()

    # Submit
    try:
        resp = await loop.run_in_executor(None, lambda: _make_svc().cv_sync2async_submit_task(body))
    except Exception as e:
        raw = e.args[0] if e.args else b""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        raise RuntimeError(f"[{label}/submit] {raw}") from None
    _assert_ok(resp, f"{label}/submit")
    task_id = resp["data"]["task_id"]
    logger.info("%s submitted  task_id=%s", label, task_id)

    # Poll
    poll_body = {"req_key": body["req_key"], "task_id": task_id}
    for attempt in range(1, _MAX_TRIES + 1):
        poll_resp = await loop.run_in_executor(None, lambda: _make_svc().cv_sync2async_get_result(poll_body))
        _assert_ok(poll_resp, f"{label}/poll")
        data = poll_resp["data"]
        status = data.get("status")

        if status == _STATUS_DONE:
            logger.info("%s done  task_id=%s", label, task_id)
            return {"task_id": task_id, "data": data}

        if status == _STATUS_FAILED:
            reason = data.get("message") or data.get("err_msg", str(status))
            raise RuntimeError(f"{label} failed  task_id={task_id}  reason={reason}")

        logger.debug("%s attempt=%d/%d status=%s", label, attempt, _MAX_TRIES, status)
        await asyncio.sleep(_POLL_INTERVAL)

    raise TimeoutError(
        f"{label} task_id={task_id} timed out after {_MAX_TRIES * _POLL_INTERVAL}s"
    )


# ---------------------------------------------------------------------------
# Base tool interface
# ---------------------------------------------------------------------------

class BaseTool(ABC):
    name: str         # key used in plan JSON
    description: str  # injected verbatim into planner system prompt
    input_schema: dict  # JSON Schema for planner-settable overrides only

    @abstractmethod
    async def run(self, ctx: dict) -> dict:
        """
        Execute the tool against the accumulated execution context.
        Returns a dict whose keys are merged back into ctx.
        """


# ---------------------------------------------------------------------------
# Tool 1 — Image-to-Video  (JiMeng 3.0 Pro)
# docs: https://www.volcengine.com/docs/85621/1783678
# ---------------------------------------------------------------------------

class ImageToVideoTool(BaseTool):
    name = "image_to_video"
    description = (
        "Animates a hand-drawn sketch into a short video using JiMeng 3.0 Pro. "
        "Requires context keys: image_url (str), enhanced_prompt (str). "
        "Optional overrides: duration (4 or 8 s), width (int), height (int). "
        "Produces: video_url."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "duration": {"type": "integer", "enum": [4, 8], "default": 4},
            "width":    {"type": "integer", "default": 1280},
            "height":   {"type": "integer", "default": 720},
        },
        "required": [],
    }

    _REQ_KEY = "jimeng_ti2v_v30_pro"

    async def run(self, ctx: dict) -> dict:
        body = {
            "req_key":    self._REQ_KEY,
            "prompt":     ctx["enhanced_prompt"],
            "image_urls": [ctx["image_url"]],
            "duration":   ctx.get("duration", 4),
            "width":      ctx.get("width", 1280),
            "height":     ctx.get("height", 720),
        }
        result = await _submit_and_poll(body, "ImageToVideo")
        return {
            "video_url": _extract_video_url(result["data"]),
            "task_id":   result["task_id"],
        }


# ---------------------------------------------------------------------------
# Tool 2 — Text-to-Video  (JiMeng 3.0)
# ---------------------------------------------------------------------------

class TextToVideoTool(BaseTool):
    name = "text_to_video"
    description = (
        "Generates a video directly from a text description — no image required. "
        "Use this when the user has NO sketch/image and just wants to describe a scene. "
        "Requires context key: enhanced_prompt (str). "
        "Optional overrides: duration (4 or 8 s), width (int), height (int). "
        "Produces: video_url."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "duration": {"type": "integer", "enum": [4, 8], "default": 4},
            "width":    {"type": "integer", "default": 1280},
            "height":   {"type": "integer", "default": 720},
        },
        "required": [],
    }

    _REQ_KEY = "jimeng_ti2v_v30_pro"

    async def run(self, ctx: dict) -> dict:
        body = {
            "req_key":  self._REQ_KEY,
            "prompt":   ctx["enhanced_prompt"],
            "duration": ctx.get("duration", 4),
            "width":    ctx.get("width", 1280),
            "height":   ctx.get("height", 720),
        }
        result = await _submit_and_poll(body, "TextToVideo")
        return {
            "video_url": _extract_video_url(result["data"]),
            "task_id":   result["task_id"],
        }


# ---------------------------------------------------------------------------
# Tool 3 — Audio-driven Portrait Video
# docs: https://www.volcengine.com/docs/86081/1804513
#
# Two-step flow (handled internally as one logical tool):
#   Step A: register the portrait image  → portrait_id
#   Step B: submit video generation with portrait_id + audio_url  → video_url
# ---------------------------------------------------------------------------

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

    _REQ_KEY_DETECT = "jimeng_realman_avatar_object_detection"
    _REQ_KEY_VIDEO  = "jimeng_realman_avatar_picture_omni_v15"

    async def run(self, ctx: dict) -> dict:
        loop = asyncio.get_running_loop()

        # Step A: subject detection → mask URLs (synchronous, best-effort)
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
            # Try common keys where mask URL(s) might live
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

        # Step B: submit video generation task (async)
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

        # Poll for result
        poll_body = {"req_key": self._REQ_KEY_VIDEO, "task_id": task_id}
        for attempt in range(1, _MAX_TRIES + 1):
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

            logger.debug("AudioPortrait attempt=%d/%d status=%s", attempt, _MAX_TRIES, status)
            await asyncio.sleep(_POLL_INTERVAL)

        raise TimeoutError(
            f"AudioPortrait task_id={task_id} timed out after {_MAX_TRIES * _POLL_INTERVAL}s"
        )


# ---------------------------------------------------------------------------
# Tool 3 — Video Special Effects
# docs: https://www.volcengine.com/docs/86081/1804543
# ---------------------------------------------------------------------------

class VideoEffectsTool(BaseTool):
    name = "video_effects"
    description = (
        "Applies a named visual effect to an existing video. "
        "Requires context key: video_url (str) — produced by a previous step. "
        "Optional override: effect (str) — the effect name, e.g. 'glitch', 'vintage', 'cinematic'. "
        "Produces: video_url (the processed video, replaces prior video_url in context)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "effect": {
                "type": "string",
                "description": "Effect name to apply. Ask the user if unspecified.",
            },
        },
        "required": [],
    }

    # TODO: confirm req_key and 'effect' param name with https://www.volcengine.com/docs/86081/1804543
    _REQ_KEY = "jimeng_video_effects_v2"

    async def run(self, ctx: dict) -> dict:
        body = {
            "req_key":   self._REQ_KEY,
            "video_url": ctx["video_url"],
            "effect":    ctx.get("effect", ""),
        }
        result = await _submit_and_poll(body, "VideoEffects")
        return {
            "video_url": _extract_video_url(result["data"]),
            "task_id":   result["task_id"],
        }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Tool 4 — Text-to-Speech  (edge-tts, free, no API key required)
# Produces an audio file served via /media/, which audio_portrait can consume.
# ---------------------------------------------------------------------------

class TTSTool(BaseTool):
    name = "tts"
    description = (
        "Converts text to natural speech and returns an audio_url. "
        "Use this BEFORE audio_portrait when the user has no audio file. "
        "Requires context key: tts_text (str) — the words to speak; "
        "falls back to user_description if tts_text not set. "
        "Optional override: voice (str) — edge-tts voice name, "
        "default 'zh-CN-XiaoxiaoNeural' (female Chinese). "
        "Other good voices: 'zh-CN-YunxiNeural' (male), 'en-US-JennyNeural' (English). "
        "Produces: audio_url."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "tts_text": {
                "type": "string",
                "description": "Exact text for the character to speak. If omitted, uses user_description.",
            },
            "voice": {
                "type": "string",
                "default": "zh-CN-XiaoxiaoNeural",
                "description": "edge-tts voice name.",
            },
        },
        "required": [],
    }

    async def run(self, ctx: dict) -> dict:
        text  = ctx.get("tts_text") or ctx.get("user_description", "")
        voice = ctx.get("voice", "zh-CN-XiaoxiaoNeural")

        fname    = f"tts_{uuid.uuid4().hex}.mp3"
        tmp_path = f"/tmp/{fname}"

        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(tmp_path)

        audio_url = f"{PUBLIC_BASE_URL}/media/{fname}"
        logger.info("TTS saved  voice=%s  path=%s  url=%s", voice, tmp_path, audio_url)
        return {"audio_url": audio_url}


# ---------------------------------------------------------------------------
# Tool 5 — Multi-Shot Video  (generates N clips from a script, concurrently)
# ---------------------------------------------------------------------------

class MultiShotTool(BaseTool):
    name = "multi_shot_video"
    description = (
        "Generates 2–4 independent video clips (shots/scenes) from a multi-scene script. "
        "Use when the user provides a detailed script with multiple scenes, "
        "or asks for a 'multi-shot', 'multi-scene', or '多镜头' video. "
        "Requires context key: user_description (the full script). "
        "Optional override: n_shots (int 2–4, default 3). "
        "Produces: video_url (first clip), video_urls (JSON list of all clip URLs)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "n_shots": {
                "type": "integer",
                "minimum": 2,
                "maximum": 4,
                "default": 3,
                "description": "Number of video clips to generate.",
            },
        },
        "required": [],
    }

    _REQ_KEY = "jimeng_ti2v_v30_pro"

    async def _parse_scenes(self, script: str, n: int) -> list[str]:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )
        system = f"""\
You are a cinematographer. Split the user's script into exactly {n} short video-clip prompts.
Each prompt must:
1. Start with "Hand-drawn animation style, 2D sketch art,"
2. Include a unique camera move (push-in / pull-back / pan / crane / tracking / static wide)
3. Include cinematic lighting (golden hour / moonlit / lantern glow / dappled / dramatic side-light)
4. Be 40–60 words, present tense, describe only visible motion and atmosphere.
5. End with "consistent line-art aesthetic, fluid animation."
Return a JSON object with key "scenes" containing a list of {n} prompt strings.
Return ONLY the JSON — no markdown, no explanation."""
        resp = await client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=800,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": script},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        scenes = data.get("scenes", [])
        logger.info("MultiShot parsed %d scenes from script", len(scenes))
        return scenes[:n]

    async def _gen_clip(self, prompt: str, ctx: dict, idx: int) -> str:
        body: dict = {
            "req_key":  self._REQ_KEY,
            "prompt":   prompt,
            "duration": ctx.get("duration", 4),
            "width":    ctx.get("width", 1280),
            "height":   ctx.get("height", 720),
        }
        if ctx.get("image_url"):
            body["image_urls"] = [ctx["image_url"]]  # apply to ALL shots
        result = await _submit_and_poll(body, f"MultiShot[{idx+1}]")
        return _extract_video_url(result["data"])

    async def run(self, ctx: dict) -> dict:
        n = min(max(int(ctx.get("n_shots", 3)), 2), 4)
        script = ctx.get("user_description", "")
        scenes = await self._parse_scenes(script, n)
        if not scenes:
            raise RuntimeError("MultiShot: DeepSeek returned no scenes from script")

        clip_tasks = [self._gen_clip(scene, ctx, i) for i, scene in enumerate(scenes)]
        results = await asyncio.gather(*clip_tasks, return_exceptions=True)

        video_urls = [r for r in results if isinstance(r, str)]
        failures   = [r for r in results if isinstance(r, Exception)]
        if failures:
            logger.warning("MultiShot: %d clip(s) failed: %s", len(failures), failures)
        if not video_urls:
            raise RuntimeError("MultiShot: all clip generations failed")

        logger.info("MultiShot produced %d/%d clips", len(video_urls), n)
        return {
            "video_url":    video_urls[0],
            "video_urls":   json.dumps(video_urls, ensure_ascii=False),
            "scene_prompts": json.dumps(scenes, ensure_ascii=False),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_TOOLS: list[BaseTool] = [
    ImageToVideoTool(),
    TextToVideoTool(),
    MultiShotTool(),
    TTSTool(),
    AudioPortraitTool(),
    # VideoEffectsTool — req_key unverified, disabled until confirmed
]

TOOL_MAP: dict[str, BaseTool] = {t.name: t for t in ALL_TOOLS}
