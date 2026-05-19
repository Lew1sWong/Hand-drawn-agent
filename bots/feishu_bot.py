"""
Feishu (Lark) Bot — Hand-drawn Animation Agent
================================================
Webhook endpoint: POST /feishu/event  (mounted in api.py)

Setup (one time):
  1. open.feishu.cn → Create App (Enterprise Internal App)
  2. Permissions & Scopes:  im:message:send_as_bot,  im:resource
  3. Event Subscriptions → im.message.receive_v1
     Set Request URL: https://your-server.com/feishu/event
  4. Add to .env:
       FEISHU_APP_ID=cli_xxx
       FEISHU_APP_SECRET=xxx
       FEISHU_VERIFICATION_TOKEN=xxx   (Event Subscriptions page)
       PUBLIC_BASE_URL=https://your-server.com

Conversation flow (mirrors Telegram bot):
  any msg      → ask for image   (if state is waiting_image)
  📷 image     → store, ask for description (audio optional)
  🎵 audio     → store, ask for description
  💬 text desc → run agent → reply with video URL
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    GetMessageResourceRequest,
)

from agent import run_agent

logger = logging.getLogger(__name__)

FEISHU_APP_ID             = os.environ.get("FEISHU_APP_ID", "")
FEISHU_APP_SECRET         = os.environ.get("FEISHU_APP_SECRET", "")
FEISHU_VERIFICATION_TOKEN = os.environ.get("FEISHU_VERIFICATION_TOKEN", "")
PUBLIC_BASE_URL           = os.environ.get("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")

_executor = ThreadPoolExecutor(max_workers=4)

lark_client = (
    lark.Client.builder()
    .app_id(FEISHU_APP_ID)
    .app_secret(FEISHU_APP_SECRET)
    .build()
)


# ---------------------------------------------------------------------------
# In-memory FSM  { open_id: {"state": str, ...data} }
# ---------------------------------------------------------------------------

_S_WAIT_IMAGE  = "waiting_image"
_S_WAIT_AUDIO  = "waiting_audio_or_desc"
_S_WAIT_DESC   = "waiting_desc"
_S_WAIT_RATING = "waiting_rating"   # after video delivered, awaiting 1-5 score

_STOP_WORDS = {"停止", "取消", "stop", "cancel", "停", "不要了", "算了"}

_user_state: dict[str, dict[str, Any]] = {}
_running_tasks: dict[str, asyncio.Task] = {}   # open_id → active agent task


def _get_state(open_id: str) -> dict:
    return _user_state.setdefault(open_id, {"state": _S_WAIT_IMAGE})


def _set_state(open_id: str, state: str, **kwargs) -> None:
    _user_state[open_id] = {"state": state, **kwargs}


def _pop_state(open_id: str) -> dict:
    return _user_state.pop(open_id, {})


# ---------------------------------------------------------------------------
# Feishu API helpers  (SDK calls are sync — run via executor)
# ---------------------------------------------------------------------------

def _send_text_sync(chat_id: str, text: str) -> None:
    req = (
        CreateMessageRequest.builder()
        .receive_id_type("chat_id")
        .request_body(
            CreateMessageRequestBody.builder()
            .receive_id(chat_id)
            .msg_type("text")
            .content(json.dumps({"text": text}))
            .build()
        )
        .build()
    )
    resp = lark_client.im.v1.message.create(req)
    if not resp.success():
        logger.error("send_text failed code=%s msg=%s", resp.code, resp.msg)


def _extract_post_text(content: dict) -> str:
    """Extract plain text from a Feishu 'post' (rich-text) message."""
    lines: list[str] = []
    # content may have language keys like "zh_cn" or "en_us"; take the first one
    body = next(iter(content.values()), {}) if content else {}
    for paragraph in body.get("content", []):
        parts: list[str] = []
        for elem in paragraph:
            if elem.get("tag") in ("text", "a"):
                parts.append(elem.get("text", ""))
        lines.append("".join(parts))
    return "\n".join(lines)


async def _send(chat_id: str, text: str) -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, _send_text_sync, chat_id, text)


def _download_sync(message_id: str, file_key: str, rtype: str) -> bytes:
    req = (
        GetMessageResourceRequest.builder()
        .message_id(message_id)
        .file_key(file_key)
        .type(rtype)
        .build()
    )
    resp = lark_client.im.v1.message_resource.get(req)
    if not resp.success():
        raise RuntimeError(f"Feishu download failed: {resp.msg}")
    return resp.raw.content


async def _save_and_url(message_id: str, file_key: str, rtype: str, suffix: str) -> str:
    """Download a Feishu resource, save to /tmp, return a public URL via /media/."""
    loop = asyncio.get_event_loop()
    data: bytes = await loop.run_in_executor(
        _executor, _download_sync, message_id, file_key, rtype
    )
    fname = f"feishu_{uuid.uuid4().hex}{suffix}"
    Path(f"/tmp/{fname}").write_bytes(data)
    return f"{PUBLIC_BASE_URL}/media/{fname}"


# ---------------------------------------------------------------------------
# Progress-aware agent runner  (registers task so user can cancel it)
# ---------------------------------------------------------------------------

async def _run_agent_with_progress(open_id: str, chat_id: str, **kwargs) -> object:
    """Run run_agent, register the task for cancellation, ping every 30 s."""
    agent_task = asyncio.create_task(run_agent(**kwargs))
    _running_tasks[open_id] = agent_task
    elapsed = 0
    try:
        while not agent_task.done():
            await asyncio.sleep(30)
            elapsed += 30
            if not agent_task.done():
                await _send(
                    chat_id,
                    f"⏳ Still generating... ({elapsed}s elapsed, usually 1–3 min)\n"
                    f"发送「取消」可以中止生成。",
                )
        return await agent_task  # re-raise any exception
    finally:
        _running_tasks.pop(open_id, None)


def _format_video_result(result) -> str:
    """Build the success message, handling single or multi-shot results."""
    plan_str = " → ".join(step["tool"] for step in result.plan)
    if result.video_urls and len(result.video_urls) > 1:
        clips = "\n".join(
            f"  🎞️ 第{i+1}镜：{url}" for i, url in enumerate(result.video_urls)
        )
        return (
            f"🎬 多镜头视频生成完成！({len(result.video_urls)} 个片段)\n\n"
            f"{clips}\n\n"
            f"Pipeline: {plan_str}"
        )
    return (
        f"🎬 Your video is ready!\n\n"
        f"🔗 {result.video_url}\n\n"
        f"Pipeline: {plan_str}"
    )


# ---------------------------------------------------------------------------
# Per-message-type handlers
# ---------------------------------------------------------------------------

async def _on_image(open_id: str, chat_id: str, message_id: str, content: dict) -> None:
    image_key = content.get("image_key", "")
    _set_state(open_id, _S_WAIT_AUDIO, image_key=image_key, image_msg_id=message_id)
    await _send(
        chat_id,
        "✅ Got your sketch!\n\n"
        "Now send me:\n"
        "• A text description of the animation you want\n"
        "• Or send a 🎵 audio / voice file first for a lip-sync effect, then the description\n\n"
        "💡 Tip: Send a detailed multi-scene script for multi-shot (多镜头) generation!",
    )


async def _on_audio(open_id: str, chat_id: str, message_id: str, content: dict) -> None:
    state = _get_state(open_id)
    if state["state"] != _S_WAIT_AUDIO:
        await _send(chat_id, "Please send a sketch image first.")
        return

    audio_key = content.get("file_key", "")
    _set_state(
        open_id, _S_WAIT_DESC,
        image_key=state["image_key"],
        image_msg_id=state["image_msg_id"],
        audio_key=audio_key,
        audio_msg_id=message_id,
    )
    await _send(chat_id, "🎵 Got the audio! Now send me a description of the animation.")


async def _on_text(open_id: str, chat_id: str, text: str) -> None:
    text = text.strip()
    state = _get_state(open_id)
    s = state["state"]

    # ── Cancel running generation ──────────────────────────────────────
    if any(w in text for w in _STOP_WORDS):
        task = _running_tasks.get(open_id)
        if task and not task.done():
            task.cancel()
            _running_tasks.pop(open_id, None)
            _set_state(open_id, _S_WAIT_IMAGE)
            await _send(chat_id, "⛔ 已取消生成。发送新的描述或图片开始新的创作！")
        else:
            await _send(chat_id, "没有正在进行的生成任务。")
        return

    # ── Handle rating reply ────────────────────────────────────────────
    if s == _S_WAIT_RATING:
        try:
            rating = int(text.strip())
            if 1 <= rating <= 5:
                from memory import UserMemory
                mem = UserMemory("user_memory.json").load()
                prompt = state.get("last_prompt", "")
                mem.record_rating(prompt, rating)
                stars = "⭐" * rating
                await _send(chat_id, f"{stars} 谢谢反馈！({rating}/5) 这将帮助我了解你的风格偏好。")
                _set_state(open_id, _S_WAIT_IMAGE)
                await _send(chat_id, "发送新描述或图片，继续创作！🎨")
                return
        except ValueError:
            pass
        # Non-numeric input while waiting for rating — skip rating, treat as new request
        _set_state(open_id, _S_WAIT_IMAGE)
        s = _S_WAIT_IMAGE

    # ── No image yet — text-to-video ────────────────────────────────────
    if s == _S_WAIT_IMAGE:
        if not text:
            await _send(chat_id, "Please send a description or a sketch image.")
            return
        _pop_state(open_id)
        await _send(chat_id, "⏳ Generating your video from text — this usually takes 1–3 minutes.")
        try:
            result = await _run_agent_with_progress(open_id, chat_id, user_request=text, assets={})
            await _send(chat_id, _format_video_result(result))
            _set_state(open_id, _S_WAIT_RATING, last_prompt=result.last_prompt or text)
            await _send(chat_id, "⭐ 请给这个视频打分（回复 1–5）\n1=很差  3=还可以  5=非常满意")
        except asyncio.CancelledError:
            await _send(chat_id, "⛔ 生成已取消。")
            _set_state(open_id, _S_WAIT_IMAGE)
        except Exception as exc:
            logger.exception("Text-to-video error for user %s", open_id)
            await _send(chat_id, f"❌ Something went wrong: {exc}")
            _set_state(open_id, _S_WAIT_IMAGE)
            await _send(chat_id, "Send another description or sketch to create a new animation! 🎨")
        return

    if s not in (_S_WAIT_AUDIO, _S_WAIT_DESC):
        return

    if not text:
        await _send(chat_id, "Please send a non-empty description.")
        return

    data = _pop_state(open_id)
    await _send(
        chat_id,
        "⏳ Got it! Generating your video — this usually takes 1–3 minutes.\n"
        "I'll send the result here when it's ready. 发送「取消」可中止。",
    )

    assets: dict = {}
    try:
        assets["image_url"] = await _save_and_url(
            data["image_msg_id"], data["image_key"], "image", ".jpg"
        )
        if data.get("audio_key"):
            assets["audio_url"] = await _save_and_url(
                data["audio_msg_id"], data["audio_key"], "file", ".mp3"
            )

        result = await _run_agent_with_progress(open_id, chat_id, user_request=text, assets=assets)
        await _send(chat_id, _format_video_result(result))
        _set_state(open_id, _S_WAIT_RATING, last_prompt=result.last_prompt or text)
        await _send(chat_id, "⭐ 请给这个视频打分（回复 1–5）\n1=很差  3=还可以  5=非常满意")

    except asyncio.CancelledError:
        await _send(chat_id, "⛔ 生成已取消。")
        _set_state(open_id, _S_WAIT_IMAGE)
    except Exception as exc:
        logger.exception("Agent error for user %s", open_id)
        await _send(chat_id, f"❌ Something went wrong: {exc}")
        _set_state(open_id, _S_WAIT_IMAGE)
        await _send(chat_id, "Send another sketch to create a new animation! 🎨")


# ---------------------------------------------------------------------------
# Main entry point — called from api.py
# ---------------------------------------------------------------------------

async def handle_event(body: dict) -> dict:
    """
    Parse a raw Feishu webhook payload and dispatch.
    Returns the response dict (FastAPI serialises it to JSON).

    Feishu requires a response within 3 seconds; heavy work runs as a
    background task via asyncio.create_task so we return immediately.
    """
    # URL verification challenge (Feishu calls this once when you save the URL)
    if body.get("type") == "url_verification":
        if FEISHU_VERIFICATION_TOKEN:
            token = body.get("token", "")
            if token != FEISHU_VERIFICATION_TOKEN:
                logger.warning("url_verification: token mismatch")
        return {"challenge": body.get("challenge", "")}

    header     = body.get("header", {})
    event_type = header.get("event_type", "")

    if event_type != "im.message.receive_v1":
        return {}

    event   = body.get("event", {})
    sender  = event.get("sender", {})
    message = event.get("message", {})

    if sender.get("sender_type") == "app":
        return {}  # ignore bot's own messages

    open_id    = sender.get("sender_id", {}).get("open_id", "")
    chat_id    = message.get("chat_id", "")
    message_id = message.get("message_id", "")
    msg_type   = message.get("message_type", "")

    if not (open_id and chat_id and message_id):
        return {}

    try:
        content = json.loads(message.get("content", "{}"))
    except json.JSONDecodeError:
        content = {}

    if msg_type == "image":
        asyncio.create_task(_on_image(open_id, chat_id, message_id, content))
    elif msg_type in ("audio", "file"):
        asyncio.create_task(_on_audio(open_id, chat_id, message_id, content))
    elif msg_type == "text":
        asyncio.create_task(_on_text(open_id, chat_id, content.get("text", "")))
    elif msg_type == "post":
        asyncio.create_task(_on_text(open_id, chat_id, _extract_post_text(content)))
    else:
        asyncio.create_task(
            _send(chat_id, f"Sorry, I only handle image, audio, and text (got: {msg_type}).")
        )

    # Return immediately — Feishu needs a response within 3 s
    return {}
