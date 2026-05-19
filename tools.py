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


async def _poll_loop(task_id: str, req_key: str, label: str, use_legacy: bool) -> dict:
    """Shared poll loop for both API paths."""
    loop = asyncio.get_running_loop()
    poll_body = {"req_key": req_key, "task_id": task_id}
    for attempt in range(1, _MAX_TRIES + 1):
        if use_legacy:
            poll_resp = await loop.run_in_executor(
                None, lambda: _make_svc().cv_get_result(poll_body)
            )
        else:
            poll_resp = await loop.run_in_executor(
                None, lambda: _make_svc().cv_sync2async_get_result(poll_body)
            )
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


async def _submit_and_poll(body: dict, label: str) -> dict:
    """
    Submit a Volcengine async task and poll until done.

    Tries cv_sync2async_submit_task first (newer path).
    Falls back to cv_submit_task (older path) if the newer endpoint
    returns a 504 — which happens when the service uses the legacy API.

    Returns {"task_id": str, "data": dict} on success.
    """
    loop = asyncio.get_running_loop()

    # ── Try newer path: cv_sync2async_submit_task ─────────────────────
    try:
        resp = await loop.run_in_executor(
            None, lambda: _make_svc().cv_sync2async_submit_task(body)
        )
        # If the gateway returned raw HTML (504), resp will be bytes
        if isinstance(resp, bytes) and b"504" in resp:
            raise RuntimeError("504")
        _assert_ok(resp, f"{label}/submit")
        task_id = resp["data"]["task_id"]
        logger.info("%s submitted (sync2async)  task_id=%s", label, task_id)
        return await _poll_loop(task_id, body["req_key"], label, use_legacy=False)

    except RuntimeError as exc:
        if "504" not in str(exc):
            raise
        logger.warning(
            "%s: cv_sync2async_submit_task returned 504 — falling back to cv_submit_task",
            label,
        )

    # ── Fallback: cv_submit_task (legacy path) ────────────────────────
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
    return await _poll_loop(task_id, body["req_key"], label, use_legacy=True)


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
        image_url = ctx.get("image_url")
        if not image_url:
            raise ValueError(
                "image_to_video requires an image. "
                "Send a sketch image first, or use text_to_video for text-only requests."
            )
        body = {
            "req_key":    self._REQ_KEY,
            "prompt":     ctx["enhanced_prompt"],
            "image_urls": [image_url],
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
# FigurineToAnimeCharTool  (Replicate pipeline)
# ---------------------------------------------------------------------------

def _replicate_to_url(output) -> str:
    """Normalize any Replicate output (FileOutput / list / str) to a URL string."""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        if not output:
            raise RuntimeError("Replicate returned an empty output list")
        return _replicate_to_url(output[0])
    if hasattr(output, "url"):
        return str(output.url)
    # generator / iterator
    try:
        items = list(output)
        if items:
            return _replicate_to_url(items[0])
    except Exception:
        pass
    return str(output)


class FigurineToAnimeCharTool(BaseTool):
    """
    Two-step Replicate pipeline:
      1. depth-anything-v2-large — extract a depth map to preserve 3-D structure
      2. img2img SDXL ControlNet (depth) — render in 2-D anime style

    Reads  ctx["image_url"]          (the figurine photo)
    Writes ctx["anime_image_url"]    (the anime render)
           ctx["image_url"]          (overwritten so downstream tools use the anime image)
    """

    name        = "figurine_to_anime"
    description = """\
Convert a Q-version figurine / figure (手办) photo into a 2D anime-style character image.

Pipeline (Replicate):
  Step 1 — depth-anything/depth-anything-v2-large: extract depth map (preserves 3-D chibi volume)
  Step 2 — lucataco/sdxl-controlnet + depth: render in 2-D anime style
             (preserves outfit colours, hair, accessories)

Writes anime_image_url and overwrites image_url so the next tool (image_to_video)
automatically picks up the anime render.

USE THIS TOOL when the user uploads a figurine / figure / 手办 photo and asks
for an animated video — run figurine_to_anime FIRST, then image_to_video."""

    input_schema = {
        "type": "object",
        "properties": {
            "prompt_suffix": {
                "type":        "string",
                "description": "Extra style descriptors appended to the anime prompt, e.g. 'pink twin-tails, sailor uniform'",
            },
            "strength": {
                "type":        "number",
                "description": "img2img denoising strength 0.0–1.0 (default 0.75)",
            },
        },
    }

    _DEPTH_MODEL = "depth-anything/depth-anything-v2-large"
    # SDXL ControlNet — depth conditioning; swap version hash if Replicate updates it
    _ANIME_MODEL = "lucataco/sdxl-controlnet:06775cd262843edbde5abab958abdbb65a0a6b58dcd869086358b1f55a0b2c70"

    async def run(self, ctx: dict) -> dict:
        image_url = ctx.get("image_url")
        if not image_url:
            raise ValueError("[FigurineToAnime] image_url is required in context")

        api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not api_token:
            raise RuntimeError("[FigurineToAnime] REPLICATE_API_TOKEN env var is not set")

        # Set token via env var — the most reliable path for replicate.async_run()
        os.environ["REPLICATE_API_TOKEN"] = api_token
        import replicate

        loop = asyncio.get_running_loop()

        # ── Step 1: depth extraction ──────────────────────────────────────
        logger.info("[FigurineToAnime] extracting depth map  image_url=%s", image_url)
        depth_out = await loop.run_in_executor(
            None,
            lambda: replicate.run(self._DEPTH_MODEL, input={"image": image_url}),
        )
        depth_url = _replicate_to_url(depth_out)
        logger.info("[FigurineToAnime] depth map ready  depth_url=%s", depth_url)

        # ── Step 2: anime style conversion ───────────────────────────────
        base_prompt = (
            "masterpiece, best quality, 2D anime illustration, "
            "chibi Q-version character, vibrant colours, clean line art, "
            "detailed outfit and hair"
        )
        suffix = ctx.get("prompt_suffix", "")
        if suffix:
            base_prompt = f"{base_prompt}, {suffix}"

        logger.info("[FigurineToAnime] converting to anime style  prompt='%s'", base_prompt)
        anime_out = await loop.run_in_executor(
            None,
            lambda: replicate.run(
                self._ANIME_MODEL,
                input={
                    "prompt":          base_prompt,
                    "negative_prompt": "3D render, photorealistic, blurry, watermark, text",
                    "image":           depth_url,
                    "num_inference_steps": 30,
                    "guidance_scale":  7.5,
                    "controlnet_conditioning_scale": 0.8,
                    "scheduler":       "K_EULER_ANCESTRAL",
                },
            ),
        )
        anime_url = _replicate_to_url(anime_out)
        logger.info("[FigurineToAnime] anime image ready  anime_url=%s", anime_url)

        return {
            "anime_image_url": anime_url,
            "image_url":       anime_url,
        }


# ---------------------------------------------------------------------------
# AddBGMTool  — merge narration / ambient audio into a silent video
# ---------------------------------------------------------------------------

class AddBGMTool(BaseTool):
    """
    Post-processing step: adds audio to a silent generated video.

    Pipeline:
      1. Download the video from ctx["video_url"] using requests
      2. Synthesise narration from ctx["user_description"] with edge-tts
      3. Loop the audio to match video duration and merge with ffmpeg
      4. Serve the merged file via /media/ and overwrite video_url

    Requires ffmpeg to be installed on the server (brew install ffmpeg / apt install ffmpeg).
    """

    name        = "add_bgm"
    description = """\
Add narration audio to a silent video (image_to_video / text_to_video / multi_shot_video).
Uses edge-tts to synthesise a voice-over from the scene description, then merges
audio and video with ffmpeg. Overwrites video_url with the audio+video file.

Add this as the LAST step of any plan that produces a silent video when the user
asks for sound / 声音 / 配音 / 旁白 / narration / audio / music.
Do NOT add this after audio_portrait — that tool already produces audio.

Reads:  video_url, user_description, lang (optional, default zh)
Writes: video_url (the merged file served via /media/)"""

    input_schema = {
        "type": "object",
        "properties": {
            "voice": {
                "type": "string",
                "description": (
                    "edge-tts voice name override. "
                    "Chinese: zh-CN-XiaoxiaoNeural (F, default), zh-CN-YunxiNeural (M). "
                    "English: en-US-JennyNeural (F), en-US-GuyNeural (M)."
                ),
            },
        },
    }

    async def run(self, ctx: dict) -> dict:
        import subprocess
        import requests as _req

        video_url = ctx.get("video_url")
        if not video_url:
            raise ValueError("[AddBGM] video_url is required in context")

        text = ctx.get("user_description", "")
        if not text:
            raise ValueError("[AddBGM] user_description is required for TTS narration")

        lang  = ctx.get("lang", "zh")
        voice = ctx.get("voice") or (
            "zh-CN-XiaoxiaoNeural" if lang == "zh" else "en-US-JennyNeural"
        )

        uid       = uuid.uuid4().hex
        audio_in  = Path(f"/tmp/bgm_audio_{uid}.mp3")
        video_in  = Path(f"/tmp/bgm_vid_in_{uid}.mp4")
        video_out = Path(f"/tmp/bgm_{uid}.mp4")

        try:
            # ── Step 1: download silent video ─────────────────────────────
            logger.info("[AddBGM] downloading video  url=%s", video_url)
            r = _req.get(video_url, timeout=120)
            r.raise_for_status()
            video_in.write_bytes(r.content)
            logger.info("[AddBGM] video saved  size=%d bytes", len(r.content))

            # ── Step 2: synthesise narration ──────────────────────────────
            narration = text[:400]          # edge-tts is reliable up to ~400 chars
            logger.info("[AddBGM] TTS  voice=%s  text=%s", voice, narration[:60])
            communicate = edge_tts.Communicate(narration, voice)
            await communicate.save(str(audio_in))
            logger.info("[AddBGM] TTS saved  size=%d bytes", audio_in.stat().st_size)

            # ── Step 3: ffmpeg merge ──────────────────────────────────────
            # -stream_loop -1 : loop audio indefinitely
            # -shortest       : stop when the video ends
            # -c:v copy       : don't re-encode video (fast, lossless)
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_in),
                "-stream_loop", "-1", "-i", str(audio_in),
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "128k",
                "-shortest",
                str(video_out),
            ]
            logger.info("[AddBGM] running ffmpeg")
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if proc.returncode != 0:
                raise RuntimeError(f"[AddBGM] ffmpeg error:\n{proc.stderr[-800:]}")
            logger.info("[AddBGM] merge done  size=%d bytes", video_out.stat().st_size)

            # ── Step 4: serve merged file ─────────────────────────────────
            out_name = f"bgm_{uid}.mp4"
            video_out.rename(Path(f"/tmp/{out_name}"))
            new_url = f"{PUBLIC_BASE_URL}/media/{out_name}"
            logger.info("[AddBGM] ready  url=%s", new_url)
            return {"video_url": new_url, "has_audio": True}

        finally:
            for p in (audio_in, video_in, video_out):
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_TOOLS: list[BaseTool] = [
    FigurineToAnimeCharTool(),
    ImageToVideoTool(),
    TextToVideoTool(),
    MultiShotTool(),
    TTSTool(),
    AudioPortraitTool(),
    AddBGMTool(),
    # VideoEffectsTool — req_key unverified, disabled until confirmed
]

TOOL_MAP: dict[str, BaseTool] = {t.name: t for t in ALL_TOOLS}
