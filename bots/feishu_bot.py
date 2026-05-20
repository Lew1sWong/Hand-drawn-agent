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
import re
import sys
import uuid
from collections import OrderedDict
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
from conversation import ConversationSession, ConvDecision, PostGenDecision
from memory import memory_path_for, UserMemory

logger = logging.getLogger(__name__)

FEISHU_APP_ID             = os.environ.get("FEISHU_APP_ID", "")
FEISHU_APP_SECRET         = os.environ.get("FEISHU_APP_SECRET", "")
FEISHU_VERIFICATION_TOKEN = os.environ.get("FEISHU_VERIFICATION_TOKEN", "")
PUBLIC_BASE_URL           = os.environ.get("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")

_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="feishu-")


def _guarded_task(coro) -> asyncio.Task:
    """Create an asyncio task that logs exceptions instead of swallowing them."""
    t = asyncio.create_task(coro)

    def _log_exc(task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception():
            logger.error("Background task failed: %s", task.exception(), exc_info=task.exception())

    t.add_done_callback(_log_exc)
    return t

lark_client = (
    lark.Client.builder()
    .app_id(FEISHU_APP_ID)
    .app_secret(FEISHU_APP_SECRET)
    .build()
)


# ---------------------------------------------------------------------------
# In-memory FSM  { open_id: {"state": str, ...data} }
# ---------------------------------------------------------------------------

_S_WAIT_LANG   = "waiting_language"  # onboarding: pick zh or en
_S_WAIT_IMAGE  = "waiting_image"
_S_WAIT_AUDIO  = "waiting_audio_or_desc"
_S_WAIT_DESC   = "waiting_desc"
_S_WAIT_RATING = "waiting_rating"    # after video delivered, awaiting 1-5 score

_HELLO_WORDS = {
    "hello", "hi", "hey", "你好", "嗨", "哈喽", "哈啰",
    "开始", "start", "wake", "唤醒", "启动", "help", "帮助",
}
_EXIT_WORDS   = {"exit", "quit", "退出", "重置", "reset", "/exit", "/reset", "/start"}
_STOP_WORDS   = {"停止", "取消", "stop", "cancel", "停", "不要了", "算了"}
_MEMORY_WORDS = {"记忆", "我的记忆", "memory", "my memory", "/memory", "/记忆"}

# Post-production commands (matched after video delivery)
_BGM_WORDS   = {"加配音", "加声音", "加音乐", "加旁白", "配音", "旁白",
                "add audio", "add sound", "add music", "add narration", "narrate"}
_REGEN_WORDS = {"重新生成", "重做", "再来一次", "再生成", "重试",
                "regenerate", "redo", "retry", "again"}

_user_state: dict[str, dict[str, Any]] = {}
_running_tasks: dict[str, asyncio.Task] = {}        # open_id → active agent task
_conv_sessions: dict[str, ConversationSession] = {} # open_id → conversation session
_user_locks: dict[str, asyncio.Lock] = {}           # per-user lock to prevent concurrent state mutation


def _get_user_lock(open_id: str) -> asyncio.Lock:
    if open_id not in _user_locks:
        _user_locks[open_id] = asyncio.Lock()
    return _user_locks[open_id]


def _get_conv_session(open_id: str) -> ConversationSession:
    if open_id not in _conv_sessions:
        _conv_sessions[open_id] = ConversationSession(open_id)
    return _conv_sessions[open_id]

# Deduplication: Feishu retries failed deliveries, so the same message_id
# can arrive twice. Track the last 500 processed IDs (LRU eviction).
_seen_msg_ids: OrderedDict[str, None] = OrderedDict()
_SEEN_MAX = 500

def _is_duplicate(message_id: str) -> bool:
    if message_id in _seen_msg_ids:
        return True
    _seen_msg_ids[message_id] = None
    if len(_seen_msg_ids) > _SEEN_MAX:
        _seen_msg_ids.popitem(last=False)
    return False


# ---------------------------------------------------------------------------
# Language-aware message helpers
# ---------------------------------------------------------------------------

def _lang(open_id: str) -> str:
    return _user_state.get(open_id, {}).get("lang", "zh")


def _t(open_id: str, zh: str, en: str) -> str:
    return zh if _lang(open_id) == "zh" else en


def _welcome_card(open_id: str) -> str:
    """Interactive Feishu card: bilingual intro + language-select buttons."""
    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {
                "tag": "plain_text",
                "content": "🎨 手绘动画 AI · Hand-Drawn Agent",
            },
            "template": "blue",
        },
        "elements": [
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": (
                        "**欢迎！我可以帮你：**\n"
                        "🖼️  草图 → 动画视频\n"
                        "🎵  图片 + 音频 → 口播视频\n"
                        "🎬  多镜头电影风格生成\n"
                        "⭐  记忆你的风格偏好\n\n"
                        "**Welcome! I can help you:**\n"
                        "🖼️  Sketch → animated video\n"
                        "🎵  Image + audio → talking portrait\n"
                        "🎬  Multi-shot cinematic generation\n"
                        "⭐  Remember your style preferences"
                    ),
                },
            },
            {"tag": "hr"},
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": "**请选择语言 / Choose your language:**",
                },
            },
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "🇨🇳 中文"},
                        "type": "primary",
                        "value": {
                            "action": "set_lang",
                            "lang":    "zh",
                            "open_id": open_id,
                        },
                    },
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "🇬🇧 English"},
                        "type": "default",
                        "value": {
                            "action": "set_lang",
                            "lang":    "en",
                            "open_id": open_id,
                        },
                    },
                ],
            },
        ],
    }
    return json.dumps(card)


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
    try:
        resp = lark_client.im.v1.message.create(req)
    except Exception as exc:
        logger.error("send_text SDK exception chat_id=%s: %s", chat_id, exc, exc_info=True)
        return
    if not resp.success():
        logger.error("send_text failed chat_id=%s code=%s msg=%s", chat_id, resp.code, resp.msg)


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
    loop = asyncio.get_running_loop()
    try:
        await asyncio.wait_for(
            loop.run_in_executor(_executor, _send_text_sync, chat_id, text),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        logger.error("_send timed out after 30s  chat_id=%s  text=%r", chat_id, text[:80])


def _send_card_sync(chat_id: str, card_json: str) -> None:
    req = (
        CreateMessageRequest.builder()
        .receive_id_type("chat_id")
        .request_body(
            CreateMessageRequestBody.builder()
            .receive_id(chat_id)
            .msg_type("interactive")
            .content(card_json)
            .build()
        )
        .build()
    )
    resp = lark_client.im.v1.message.create(req)
    if not resp.success():
        logger.error("send_card failed code=%s msg=%s", resp.code, resp.msg)


async def _send_card(chat_id: str, card_json: str) -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_executor, _send_card_sync, chat_id, card_json)


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


def _compress_image(data: bytes, max_px: int = 1280, quality: int = 85) -> bytes:
    """Resize + JPEG-compress raw image bytes to keep file small for ngrok serving."""
    from PIL import Image as PILImage
    import io as _io
    img = PILImage.open(_io.BytesIO(data))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if max(img.size) > max_px:
        ratio = max_px / max(img.size)
        img = img.resize(
            (int(img.width * ratio), int(img.height * ratio)),
            PILImage.LANCZOS,
        )
    buf = _io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


async def _save_and_url(message_id: str, file_key: str, rtype: str, suffix: str) -> str:
    """Download a Feishu resource, save to /tmp, return a public URL via /media/."""
    loop = asyncio.get_running_loop()
    data: bytes = await loop.run_in_executor(
        _executor, _download_sync, message_id, file_key, rtype
    )
    if not data:
        raise RuntimeError("Feishu returned empty file — download may have failed silently")

    uid = uuid.uuid4().hex
    # Compress images so they reliably transfer through ngrok to Volcengine
    if suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
        try:
            orig_size = len(data)
            data  = await loop.run_in_executor(_executor, _compress_image, data)
            fname = f"feishu_{uid}.jpg"
            logger.info("Image compressed %d → %d bytes", orig_size, len(data))
        except Exception as exc:
            logger.warning("Image compression failed (%s) — using original", exc)
            fname = f"feishu_{uid}{suffix}"
    else:
        fname = f"feishu_{uid}{suffix}"

    path = Path(f"/tmp/{fname}")
    path.write_bytes(data)
    logger.info("Saved  size=%d bytes  url=%s/media/%s", len(data), PUBLIC_BASE_URL, fname)
    return f"{PUBLIC_BASE_URL}/media/{fname}"


# ---------------------------------------------------------------------------
# Progress-aware agent runner  (registers task so user can cancel it)
# ---------------------------------------------------------------------------

_AGENT_TIMEOUT_S = 600  # 10 min hard cap on the entire agent run


async def _distill_exit_summary(
    open_id: str,
    history: list[dict],
    state: dict,
) -> str | None:
    """Call DeepSeek to produce a 1-2 sentence session recap shown on exit."""
    if not history:
        return None

    lang = state.get("lang", "zh")
    last_prompt = state.get("last_prompt", "")

    history_text = "\n".join(
        f"{'用户' if m['role'] == 'user' else 'AI'}: {m['content'][:200]}"
        for m in history[-12:]
    )

    if lang == "zh":
        system = (
            "你是手绘动画AI的会话总结助手。"
            "根据以下对话历史，用1-2句话总结本次创作成果，语气友好自然。"
            "如有视频生成成功，简要提及主题内容；如有评分，提及评分。"
            "如本次什么都没完成，简短说明。直接给出总结，无需前缀。"
        )
        user_msg = f"对话历史：\n{history_text}"
        if last_prompt:
            user_msg += f"\n\n最终生成的视频描述：{last_prompt}"
    else:
        system = (
            "You are a session recap assistant for a hand-drawn animation AI. "
            "Summarize this session in 1-2 friendly sentences. "
            "Mention what was created and the rating if given. "
            "If nothing was completed, say so briefly. Output only the summary."
        )
        user_msg = f"Conversation:\n{history_text}"
        if last_prompt:
            user_msg += f"\n\nFinal video prompt: {last_prompt}"

    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com",
    )
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model="deepseek-chat",
                max_tokens=150,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_msg},
                ],
            ),
            timeout=15.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("Exit distillation failed: %s", exc)
        return None


async def _run_agent_with_progress(open_id: str, chat_id: str, **kwargs) -> object:
    """Run run_agent with cancellation support and a 10-min hard cap."""
    kwargs.setdefault("memory_path", memory_path_for(open_id))
    agent_task = asyncio.create_task(run_agent(**kwargs))
    _running_tasks[open_id] = agent_task
    elapsed = 0
    _multishot_tip_sent = False
    try:
        while not agent_task.done():
            await asyncio.sleep(30)
            elapsed += 30
            if elapsed >= _AGENT_TIMEOUT_S:
                agent_task.cancel()
                raise TimeoutError(
                    f"Agent timed out after {_AGENT_TIMEOUT_S}s — Volcengine API may be overloaded."
                )
            if not agent_task.done():
                # After 90s, append a one-time tip about storyboard format & task splitting
                tip = ""
                if elapsed >= 90 and not _multishot_tip_sent:
                    _multishot_tip_sent = True
                    tip = _t(open_id,
                        "\n\n💡 小技巧：若您有多个场景，下次试试分镜格式：\n"
                        "「镜头1：… 镜头2：… 镜头3：…」\n"
                        "生成更快、效果更稳定！任务量大时建议分多次提交。",
                        "\n\n💡 Tip: For multi-scene requests, try the storyboard format next time:\n"
                        "\"Shot 1: … Shot 2: … Shot 3: …\"\n"
                        "Faster and more reliable! For large tasks, submit each scene separately.",
                    )
                await _send(
                    chat_id,
                    _t(open_id,
                        f"⏳ 生成中… ({elapsed}s，通常需 2–5 分钟)\n发送「取消」可中止。",
                        f"⏳ Still generating... ({elapsed}s, usually 2–5 min)\nSend 'cancel' to stop.",
                    ) + tip,
                )
        return await agent_task  # re-raise any exception
    finally:
        _running_tasks.pop(open_id, None)


def _viewer_url(video_url: str) -> str:
    from urllib.parse import quote
    return f"{PUBLIC_BASE_URL}/view?url={quote(video_url, safe='')}"


def _format_video_result(open_id: str, result) -> str:
    plan_str = " → ".join(step["tool"] for step in result.plan)
    if result.video_urls and len(result.video_urls) > 1:
        clips = "\n".join(
            f"  🎞️ Shot {i+1}: {_viewer_url(url)}" for i, url in enumerate(result.video_urls)
        ) if _lang(open_id) == "en" else "\n".join(
            f"  🎞️ 第{i+1}镜：{_viewer_url(url)}" for i, url in enumerate(result.video_urls)
        )
        return _t(open_id,
            f"🎬 多镜头视频生成完成！({len(result.video_urls)} 个片段)\n\n{clips}\n\nPipeline: {plan_str}",
            f"🎬 Multi-shot done! ({len(result.video_urls)} clips)\n\n{clips}\n\nPipeline: {plan_str}",
        )
    view_url = _viewer_url(result.video_url)
    return _t(open_id,
        f"🎬 视频已生成！\n\n🔗 {view_url}\n\nPipeline: {plan_str}",
        f"🎬 Your video is ready!\n\n🔗 {view_url}\n\nPipeline: {plan_str}",
    )


def _apply_rating(open_id: str, prompt: str, rating: int) -> None:
    """Persist a 1–5 rating to the user's memory file."""
    mem = UserMemory(memory_path_for(open_id)).load()
    mem.record_rating(prompt, rating)


def _format_memory_display(open_id: str) -> str:
    """Return a human-readable summary of the user's memory state."""
    mem = UserMemory(memory_path_for(open_id)).load()
    lang = _lang(open_id)

    if lang == "zh":
        lines = ["🧠 **你的创作记忆**\n"]
        lines.append(f"📊 累计生成：{mem.conversation_count} 次")
        if mem.distilled_style:
            lines.append(f"\n✨ **风格偏好（AI提炼）：**\n{mem.distilled_style}")
        top = mem.top_working(5)
        if top:
            lines.append(f"\n📝 **最近高质量记忆（{len(top)} 条）：**")
            stars_map = {1.0: "⭐⭐⭐⭐⭐", 0.8: "⭐⭐⭐⭐", 0.6: "⭐⭐⭐", 0.5: "⭐⭐⭐", 0.2: "⭐", 0.1: "⭐"}
            for e in top:
                rating_stars = next((v for k, v in stars_map.items() if e.quality >= k), "⭐")
                lines.append(f"  {rating_stars} {e.content[:80]}{'…' if len(e.content) > 80 else ''}")
        if mem.archive:
            lines.append(f"\n📦 归档记忆：{len(mem.archive)} 条（将在适当时自动压缩提炼）")
        if not mem.working and not mem.distilled_style:
            lines.append("\n还没有记忆哦！生成几个视频之后，AI 会开始记住你的偏好。")
    else:
        lines = ["🧠 **Your Creative Memory**\n"]
        lines.append(f"📊 Total generations: {mem.conversation_count}")
        if mem.distilled_style:
            lines.append(f"\n✨ **Style preferences (AI distilled):**\n{mem.distilled_style}")
        top = mem.top_working(5)
        if top:
            lines.append(f"\n📝 **Top memories ({len(top)}):**")
            for e in top:
                stars = round(e.quality * 5)
                lines.append(f"  {'⭐' * stars} {e.content[:80]}{'…' if len(e.content) > 80 else ''}")
        if mem.archive:
            lines.append(f"\n📦 Archived memories: {len(mem.archive)} (will be distilled automatically)")
        if not mem.working and not mem.distilled_style:
            lines.append("\nNo memories yet! Generate a few videos and the AI will start learning your style.")

    return "\n".join(lines)


def _rating_prompt(open_id: str) -> str:
    return _t(open_id,
        "⭐ 请给这个视频打分（回复 1–5）\n"
        "1=很差  3=还可以  5=非常满意\n\n"
        "💡 也可以回复：加配音 · 重新生成 · 发新图/描述继续创作",
        "⭐ Please rate this video (reply 1–5)\n"
        "1=Poor  3=OK  5=Excellent\n\n"
        "💡 Or reply: add audio · regenerate · send new image/description",
    )


# ---------------------------------------------------------------------------
# Per-message-type handlers
# ---------------------------------------------------------------------------

async def _show_welcome(open_id: str, chat_id: str) -> None:
    """Send welcome text and ask user to choose language by typing."""
    _set_state(open_id, _S_WAIT_LANG)
    await _send(chat_id,
        "🎨 手绘动画 AI · Hand-Drawn Agent\n\n"
        "欢迎！我可以帮你：\n"
        "🖼️  草图 → 动画视频\n"
        "🎵  图片 + 音频 → 口播视频\n"
        "🎬  多镜头电影风格生成\n\n"
        "Welcome! I can help you:\n"
        "🖼️  Sketch → animated video\n"
        "🎵  Image + audio → talking portrait\n"
        "🎬  Multi-shot cinematic generation\n\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "请选择语言 / Choose language:\n"
        "输入 1 → 中文\n"
        "输入 2 → English"
    )


def _ensure_state(open_id: str) -> None:
    """Init state with default language (zh) for users who skipped onboarding."""
    if open_id not in _user_state:
        _set_state(open_id, _S_WAIT_IMAGE, lang="zh")


async def _on_image(open_id: str, chat_id: str, message_id: str, content: dict) -> None:
    async with _get_user_lock(open_id):
        _ensure_state(open_id)
        _get_conv_session(open_id).clear()   # new image = new conversation context
        image_key = content.get("image_key", "")
        lang      = _lang(open_id)
        _set_state(open_id, _S_WAIT_AUDIO, image_key=image_key, image_msg_id=message_id,
                   lang=lang)
    await _send(chat_id, _t(open_id,
        "✅ 收到草图！\n\n接下来发送：\n"
        "• 动画描述文字\n"
        "• 或先发 🎵 音频实现口播，再发描述\n\n"
        "💡 发送多场景脚本可生成多镜头视频！",
        "✅ Got your sketch!\n\n"
        "Now send me:\n"
        "• A text description of the animation\n"
        "• Or send a 🎵 audio file first for lip-sync, then description\n\n"
        "💡 Send a multi-scene script for multi-shot generation!",
    ))


async def _on_audio(open_id: str, chat_id: str, message_id: str, content: dict) -> None:
    async with _get_user_lock(open_id):
        state = _get_state(open_id)
        if state["state"] != _S_WAIT_AUDIO:
            msg = _t(open_id, "请先发送草图图片。", "Please send a sketch image first.")
            wrong_state = True
        else:
            audio_key = content.get("file_key", "")
            _set_state(open_id, _S_WAIT_DESC,
                       image_key=state["image_key"], image_msg_id=state["image_msg_id"],
                       audio_key=audio_key, audio_msg_id=message_id,
                       lang=_lang(open_id))
            msg = _t(open_id, "🎵 收到音频！请发送动画描述。",
                     "🎵 Got the audio! Now send me a description of the animation.")
            wrong_state = False
    await _send(chat_id, msg)


async def _on_text(open_id: str, chat_id: str, text: str) -> None:
    text = text.strip()

    # ── Hello / wake trigger — always shows the welcome card ──────────
    if text.lower() in _HELLO_WORDS:
        await _show_welcome(open_id, chat_id)
        return

    # ── Memory display — works from any state ─────────────────────────
    if text.lower() in _MEMORY_WORDS:
        await _send(chat_id, _format_memory_display(open_id))
        return

    # ── Serialize per-user state mutations under a lock to prevent concurrent
    #    state reads from both triggering generation. ────────────────────────
    cancel_had_task = False  # initialise here so it's always bound
    async with _get_user_lock(open_id):
        # ── Exit / reset ──────────────────────────────────────────────
        if text.lower() in _EXIT_WORDS:
            task = _running_tasks.get(open_id)
            if task and not task.done():
                task.cancel()
                _running_tasks.pop(open_id, None)
            # Capture before clearing so we can distill a session recap
            exit_history = list(_get_conv_session(open_id)._history)
            exit_state   = dict(_user_state.get(open_id, {}))
            _user_state.pop(open_id, None)
            _get_conv_session(open_id).clear()
            do_reset = True
            do_cancel = False
        # ── Cancel running generation ─────────────────────────────────
        elif any(w in text for w in _STOP_WORDS):
            do_reset = False
            task = _running_tasks.get(open_id)
            if task and not task.done():
                task.cancel()
                _running_tasks.pop(open_id, None)
                _set_state(open_id, _S_WAIT_IMAGE, lang=_lang(open_id))
                do_cancel = True
                cancel_had_task = True
            else:
                do_cancel = True
                cancel_had_task = False
        else:
            do_reset = False
            do_cancel = False
            exit_history = []
            exit_state   = {}

        # ── Guard: prevent a second message from racing into generation ──
        if not do_reset and not do_cancel:
            _ensure_state(open_id)
            existing_task = _running_tasks.get(open_id)
            if existing_task and not existing_task.done():
                # Another generation is already running — tell the user
                queued_msg = _t(open_id,
                    "⏳ 正在生成中，请稍候…发送「取消」停止当前任务。",
                    "⏳ Already generating, please wait… Send 'cancel' to stop.",
                )
                busy = True
            else:
                busy = False

        state = _get_state(open_id)
        s     = state["state"]

    if do_reset:
        lang = exit_state.get("lang", "zh")
        summary = await _distill_exit_summary(open_id, exit_history, exit_state)
        if summary:
            prefix = "📋 本次创作回顾：" if lang == "zh" else "📋 Session recap:"
            await _send(chat_id, f"{prefix}\n{summary}")
        # Restore state to waiting_image, keeping the user's language preference.
        # Don't re-show the welcome/onboarding card — the user already knows the bot.
        _set_state(open_id, _S_WAIT_IMAGE, lang=lang)
        await _send(chat_id,
            "✅ 已退出，创作记录已清空。\n发送图片或描述开始新创作 🎨"
            if lang == "zh" else
            "✅ Session ended. Send an image or description to start a new creation. 🎨"
        )
        return
    if do_cancel:
        if cancel_had_task:
            await _send(chat_id, _t(open_id,
                "⛔ 已取消生成。发送新描述或图片开始新创作！",
                "⛔ Generation cancelled. Send a new description or image to start!",
            ))
        else:
            await _send(chat_id, _t(open_id,
                "没有正在进行的生成任务。",
                "No active generation task.",
            ))
        return
    if busy:
        await _send(chat_id, queued_msg)
        return

    # ── Waiting for language selection ────────────────────────────────
    if s == _S_WAIT_LANG:
        if text.lower() in ("2", "english", "en"):
            lang = "en"
        else:
            lang = "zh"
        _set_state(open_id, _S_WAIT_IMAGE, lang=lang)
        await _send(chat_id, _t(open_id,
            "✅ 已选择中文！发送图片或描述开始创作。🎨",
            "✅ English selected! Send an image or description to start. 🎨",
        ))
        UserMemory(memory_path_for(open_id)).load().update(language=lang)
        return

    # ── Post-production: LLM understands modifications, audio, rating ─────
    if s == _S_WAIT_RATING:
        lang        = _lang(open_id)
        last_prompt = state.get("last_prompt", "")
        last_video  = state.get("last_video_url")
        last_assets = state.get("last_assets", {})

        # Fast path: unambiguous numeric rating (no LLM needed)
        try:
            rating = int(text)
            if 1 <= rating <= 5:
                _apply_rating(open_id, last_prompt, rating)
                stars = "⭐" * rating
                if rating <= 2:
                    ack = _t(open_id,
                        f"{stars} 谢谢反馈！({rating}/5)\n"
                        "能说说哪里不满意吗？下次我会做得更好~\n"
                        "（也可以直接发新图/描述继续创作 🎨）",
                        f"{stars} Thanks for your feedback! ({rating}/5)\n"
                        "Mind sharing what didn't work? I'll improve next time.\n"
                        "(Or just send a new image/description to continue 🎨)",
                    )
                else:
                    ack = _t(open_id,
                        f"{stars} 谢谢反馈！({rating}/5) 我会记住你的喜好。",
                        f"{stars} Thanks for the rating! ({rating}/5) I'll remember your preferences.",
                    )
                await _send(chat_id, ack)
                # Re-acquire lock before mutating state to avoid racing with a concurrent
                # exit task that may have already reset state via _show_welcome.
                async with _get_user_lock(open_id):
                    if _user_state.get(open_id, {}).get("state") == _S_WAIT_RATING:
                        _get_conv_session(open_id).clear()
                        _set_state(open_id, _S_WAIT_IMAGE, lang=lang)
                        should_continue = True
                    else:
                        should_continue = False
                if should_continue and rating > 2:
                    await _send(chat_id, _t(open_id,
                        "发送新描述或图片，继续创作！🎨",
                        "Send another description or image to continue! 🎨",
                    ))
                return
        except ValueError:
            pass

        # LLM path: understand natural language (modify, add audio, semantic rating, etc.)
        session  = _get_conv_session(open_id)
        pg: PostGenDecision = await session.process_post_gen(text, last_prompt, last_assets)

        # ── rate ──────────────────────────────────────────────────────────
        if pg.action == "rate" and pg.rating is not None:
            _apply_rating(open_id, last_prompt, pg.rating)
            stars = "⭐" * pg.rating
            if pg.rating <= 2 and not pg.reply:
                rate_msg = _t(open_id,
                    f"{stars} 谢谢反馈！({pg.rating}/5)\n"
                    "能说说哪里不满意吗？下次我会做得更好~\n"
                    "（也可以直接发新图/描述继续创作 🎨）",
                    f"{stars} Thanks for your feedback! ({pg.rating}/5)\n"
                    "Mind sharing what didn't work? I'll improve next time.\n"
                    "(Or just send a new image/description to continue 🎨)",
                )
            else:
                rate_msg = pg.reply or _t(open_id,
                    f"{stars} 谢谢反馈！({pg.rating}/5)",
                    f"{stars} Thanks for the feedback! ({pg.rating}/5)",
                )
            await _send(chat_id, rate_msg)
            async with _get_user_lock(open_id):
                if _user_state.get(open_id, {}).get("state") == _S_WAIT_RATING:
                    session.clear()
                    _set_state(open_id, _S_WAIT_IMAGE, lang=lang)
                    should_continue = True
                else:
                    should_continue = False
            if should_continue and pg.rating > 2:
                await _send(chat_id, _t(open_id,
                    "发送新描述或图片，继续创作！🎨",
                    "Send another description or image to continue! 🎨",
                ))
            return

        # ── add_bgm ───────────────────────────────────────────────────────
        if pg.action == "add_bgm":
            if not last_video:
                await _send(chat_id, _t(open_id,
                    "找不到上一个视频，请重新生成。",
                    "No previous video found — please generate one first.",
                ))
                return
            await _send(chat_id, pg.reply)
            try:
                from tools import AddBGMTool
                bgm_ctx = {
                    "video_url":       last_video,
                    "user_description": pg.bgm_description or text,
                    "lang":            lang,
                    "mode":            pg.bgm_mode,
                }
                bgm_out = await AddBGMTool().run(bgm_ctx)
                new_url = bgm_out["video_url"]
                await _send(chat_id, _t(open_id,
                    f"🎬 已完成！\n\n🔗 {_viewer_url(new_url)}",
                    f"🎬 Done!\n\n🔗 {_viewer_url(new_url)}",
                ))
                session.clear()
                _set_state(open_id, _S_WAIT_RATING,
                           last_prompt=last_prompt, last_video_url=new_url,
                           last_assets=last_assets, lang=lang)
                await _send(chat_id, _rating_prompt(open_id))
            except Exception as exc:
                logger.exception("AddBGM post-process error for %s", open_id)
                await _send(chat_id, f"❌ {exc}")
            return

        # ── regenerate ────────────────────────────────────────────────────
        if pg.action == "regenerate":
            regen_assets = {**last_assets, **pg.param_overrides}
            _pop_state(open_id)
            await _send(chat_id, pg.reply)
            session.clear()
            try:
                result = await _run_agent_with_progress(
                    open_id, chat_id,
                    user_request=pg.refined_request or last_prompt,
                    assets=regen_assets,
                )
                await _send(chat_id, _format_video_result(open_id, result))
                _set_state(open_id, _S_WAIT_RATING,
                           last_prompt=result.last_prompt or pg.refined_request,
                           last_video_url=result.video_url,
                           last_assets=regen_assets, lang=lang)
                await _send(chat_id, _rating_prompt(open_id))
            except asyncio.CancelledError:
                await _send(chat_id, _t(open_id, "⛔ 已取消。", "⛔ Cancelled."))
                _set_state(open_id, _S_WAIT_IMAGE, lang=lang)
            except Exception as exc:
                logger.exception("Regen error for %s", open_id)
                await _send(chat_id, f"❌ {exc}")
                _set_state(open_id, _S_WAIT_IMAGE, lang=lang)
            return

        # ── new_request ───────────────────────────────────────────────────
        if pg.action == "new_request":
            session.clear()
            _set_state(open_id, _S_WAIT_IMAGE, lang=lang)
            await _send(chat_id, pg.reply or _t(open_id,
                "好的，开始新的创作！请发送描述或图片。🎨",
                "Got it, starting fresh! Send a description or image. 🎨",
            ))
            return

        # ── chat (including fallback) ─────────────────────────────────────
        await _send(chat_id, pg.reply)
        return

    # ── Text input: route through conversation layer ───────────────────
    # Covers both text-only (no image) and image-uploaded (waiting for desc)
    if s == _S_WAIT_IMAGE:
        if not text:
            await _send(chat_id, _t(open_id,
                "请发送描述文字或图片。",
                "Please send a description or an image.",
            ))
            return
        lang    = _lang(open_id)
        session = _get_conv_session(open_id)
        decision: ConvDecision = await session.process(text, available_assets={})

        if decision.action == "chat":
            # Ask clarifying question; stay in _S_WAIT_IMAGE
            await _send(chat_id, decision.reply)
            return

        # Execute — send confirmation then generate
        _pop_state(open_id)
        await _send(chat_id, decision.reply)
        session.clear()
        try:
            conv_hints: dict = {}
            if decision.audio_mode:
                conv_hints["audio_mode_hint"] = decision.audio_mode
            if decision.tts_text:
                conv_hints["tts_text"] = decision.tts_text
            result = await _run_agent_with_progress(
                open_id, chat_id, user_request=decision.user_request, assets=conv_hints
            )
            await _send(chat_id, _format_video_result(open_id, result))
            _set_state(open_id, _S_WAIT_RATING,
                       last_prompt=result.last_prompt or decision.user_request,
                       last_video_url=result.video_url,
                       last_assets={},
                       lang=lang)
            await _send(chat_id, _rating_prompt(open_id))
        except asyncio.CancelledError:
            await _send(chat_id, _t(open_id, "⛔ 生成已取消。", "⛔ Generation cancelled."))
            _set_state(open_id, _S_WAIT_IMAGE, lang=lang)
        except Exception as exc:
            logger.exception("Text-to-video error for user %s", open_id)
            await _send(chat_id, f"❌ {exc}")
            _set_state(open_id, _S_WAIT_IMAGE, lang=lang)
            await _send(chat_id, _t(open_id,
                "发送新描述或图片重试。🎨",
                "Send a new description or image to retry. 🎨",
            ))
        return

    if s not in (_S_WAIT_AUDIO, _S_WAIT_DESC):
        return

    if not text:
        await _send(chat_id, _t(open_id, "请发送非空描述。", "Please send a non-empty description."))
        return

    lang = _lang(open_id)

    # Build asset indicators so conversation knows what's already uploaded
    pending_assets: dict = {"image_url": "uploaded"}
    if state.get("audio_key"):
        pending_assets["audio_url"] = "uploaded"

    session  = _get_conv_session(open_id)
    decision = await session.process(text, available_assets=pending_assets)

    if decision.action == "chat":
        # Answer the user's question or ask for clarification; keep image in state
        await _send(chat_id, decision.reply)
        return

    # Execute — resolve URLs then generate
    data = _pop_state(open_id)
    await _send(chat_id, decision.reply)
    session.clear()

    assets: dict = {}
    try:
        assets["image_url"] = await _save_and_url(
            data["image_msg_id"], data["image_key"], "image", ".jpg"
        )
        if data.get("audio_key"):
            assets["audio_url"] = await _save_and_url(
                data["audio_msg_id"], data["audio_key"], "file", ".mp3"
            )
        conv_hints_img: dict = {}
        if decision.audio_mode:
            conv_hints_img["audio_mode_hint"] = decision.audio_mode
        if decision.tts_text:
            conv_hints_img["tts_text"] = decision.tts_text
        result = await _run_agent_with_progress(
            open_id, chat_id, user_request=decision.user_request,
            assets={**assets, **conv_hints_img}
        )
        await _send(chat_id, _format_video_result(open_id, result))
        _set_state(open_id, _S_WAIT_RATING,
                   last_prompt=result.last_prompt or decision.user_request,
                   last_video_url=result.video_url,
                   last_assets=assets,
                   lang=lang)
        await _send(chat_id, _rating_prompt(open_id))

    except asyncio.CancelledError:
        await _send(chat_id, _t(open_id, "⛔ 生成已取消。", "⛔ Generation cancelled."))
        _set_state(open_id, _S_WAIT_IMAGE, lang=lang)
    except Exception as exc:
        logger.exception("Agent error for user %s", open_id)
        await _send(chat_id, f"❌ {exc}")
        _set_state(open_id, _S_WAIT_IMAGE, lang=lang)
        await _send(chat_id, _t(open_id,
            "发送新草图继续创作。🎨",
            "Send a new sketch to continue. 🎨",
        ))


# ---------------------------------------------------------------------------
# Card action callback — called from api.py POST /feishu/card
# ---------------------------------------------------------------------------

async def handle_card_action(body: dict) -> dict:
    """
    Feishu calls this URL when a user clicks a button on an interactive card.
    We use it exclusively for the welcome-card language-selection buttons.

    Must be registered in Feishu console → App Features → Bot → Card Callback URL.
    This is SEPARATE from the event subscription URL.

    Supports both Feishu card payload formats:
      v1: body["action"]["value"], body["open_id"], body["open_chat_id"]
      v2: body["schema"]=="2.0", body["event"]["action"]["value"],
          body["event"]["operator"]["open_id"], body["event"]["context"]["open_chat_id"]
    """
    logger.info("handle_card_action body: %s", body)

    # ── handle URL verification challenge (may fire when registering the card URL) ──
    if body.get("type") == "url_verification":
        return {"challenge": body.get("challenge", "")}

    schema = body.get("schema", "1.0")

    if schema == "2.0":
        # Card kit 2.0 format
        event   = body.get("event", {})
        action  = event.get("action", {})
        value   = action.get("value") or {}
        open_id = event.get("operator", {}).get("open_id", "")
        chat_id = event.get("context", {}).get("open_chat_id", "")
    else:
        # Legacy v1 format
        action  = body.get("action") or {}
        value   = action.get("value") or action.get("form_value") or {}
        open_id = body.get("open_id", "")
        chat_id = body.get("open_chat_id", "")

    if isinstance(value, str):
        try:
            import json as _json
            value = _json.loads(value)
        except Exception:
            value = {}

    if value.get("action") != "set_lang":
        logger.info("Card action ignored (not set_lang): schema=%s value=%s", schema, value)
        return {}

    lang = value.get("lang", "zh")
    # open_id may also be embedded in value (we bake it in when building the card)
    open_id = open_id or value.get("open_id", "")

    logger.info("Card set_lang: lang=%s open_id=%s chat_id=%s", lang, open_id, chat_id)

    if not open_id:
        return {"toast": {"type": "error", "content": "Missing open_id"}}

    _set_state(open_id, _S_WAIT_IMAGE, lang=lang)

    UserMemory(memory_path_for(open_id)).load().update(language=lang)

    if chat_id:
        _guarded_task(_send(chat_id, _t(open_id,
            "✅ 已选择中文！\n\n发送图片或文字描述开始创作。🎨\n"
            "💡 发送多场景脚本可生成多镜头视频！",
            "✅ English selected!\n\nSend an image or text description to start. 🎨\n"
            "💡 Send a multi-scene script for multi-shot generation!",
        )))

    return {"toast": {"type": "success", "content": "✓ 语言已设置 / Language set"}}


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
    if body.get("type") == "url_verification":
        if FEISHU_VERIFICATION_TOKEN:
            token = body.get("token", "")
            if token != FEISHU_VERIFICATION_TOKEN:
                logger.warning("url_verification: token mismatch")
        return {"challenge": body.get("challenge", "")}

    # Enforce token on all non-verification events
    if FEISHU_VERIFICATION_TOKEN:
        incoming = body.get("header", {}).get("token", "") or body.get("token", "")
        if incoming != FEISHU_VERIFICATION_TOKEN:
            logger.warning("handle_event: token mismatch — rejected")
            return {}
    else:
        logger.warning("FEISHU_VERIFICATION_TOKEN not set — webhook UNPROTECTED")

    header     = body.get("header", {})
    event_type = header.get("event_type", "")

    if event_type != "im.message.receive_v1":
        return {}

    event   = body.get("event", {})
    sender  = event.get("sender", {})
    message = event.get("message", {})

    if sender.get("sender_type") == "app":
        return {}

    open_id    = sender.get("sender_id", {}).get("open_id", "")
    chat_id    = message.get("chat_id", "")
    message_id = message.get("message_id", "")
    msg_type   = message.get("message_type", "")

    if not (open_id and chat_id and message_id):
        return {}

    # ── Deduplicate (Feishu retries on timeout) ────────────────────────
    if _is_duplicate(message_id):
        logger.debug("Duplicate message_id=%s ignored", message_id)
        return {}

    try:
        content = json.loads(message.get("content", "{}"))
    except json.JSONDecodeError:
        content = {}

    if msg_type == "image":
        logger.info("Received image  open_id=%s  chat_id=%s", open_id, chat_id)
        _guarded_task(_on_image(open_id, chat_id, message_id, content))
    elif msg_type in ("audio", "file"):
        logger.info("Received audio/file  open_id=%s  chat_id=%s", open_id, chat_id)
        _guarded_task(_on_audio(open_id, chat_id, message_id, content))
    elif msg_type in ("text", "post"):
        text = (content.get("text", "") if msg_type == "text"
                else _extract_post_text(content))
        # Strip Feishu @-mention placeholders (group chat: "@_user_xxx bot_name 你好" → "你好")
        text = re.sub(r'@\S+', '', text).strip()
        logger.info("Received text  open_id=%s  chat_id=%s  text=%r", open_id, chat_id, text[:80])
        _guarded_task(_on_text(open_id, chat_id, text))
    else:
        logger.info("Received unsupported type=%s  open_id=%s", msg_type, open_id)
        _guarded_task(_send(chat_id, _t(open_id,
            f"暂不支持该消息类型：{msg_type}",
            f"Unsupported message type: {msg_type}",
        )))

    # Return immediately — Feishu needs a response within 3 s
    return {}
