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

_S_WAIT_LANG   = "waiting_language"  # onboarding: pick zh or en
_S_WAIT_IMAGE  = "waiting_image"
_S_WAIT_AUDIO  = "waiting_audio_or_desc"
_S_WAIT_DESC   = "waiting_desc"
_S_WAIT_RATING = "waiting_rating"    # after video delivered, awaiting 1-5 score

_HELLO_WORDS = {
    "hello", "hi", "hey", "你好", "嗨", "哈喽", "哈啰",
    "开始", "start", "wake", "唤醒", "启动", "help", "帮助",
}
_EXIT_WORDS  = {"exit", "quit", "退出", "重置", "reset", "/exit", "/reset", "/start"}
_STOP_WORDS  = {"停止", "取消", "stop", "cancel", "停", "不要了", "算了"}

# Post-production commands (matched after video delivery)
_BGM_WORDS   = {"加配音", "加声音", "加音乐", "加旁白", "配音", "旁白",
                "add audio", "add sound", "add music", "add narration", "narrate"}
_REGEN_WORDS = {"重新生成", "重做", "再来一次", "再生成", "重试",
                "regenerate", "redo", "retry", "again"}

_user_state: dict[str, dict[str, Any]] = {}
_running_tasks: dict[str, asyncio.Task] = {}   # open_id → active agent task

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
    loop = asyncio.get_event_loop()
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


async def _save_and_url(message_id: str, file_key: str, rtype: str, suffix: str) -> str:
    """Download a Feishu resource, save to /tmp, return a public URL via /media/."""
    loop = asyncio.get_event_loop()
    data: bytes = await loop.run_in_executor(
        _executor, _download_sync, message_id, file_key, rtype
    )
    if not data:
        raise RuntimeError("Feishu returned empty file — download may have failed silently")
    fname = f"feishu_{uuid.uuid4().hex}{suffix}"
    path  = Path(f"/tmp/{fname}")
    path.write_bytes(data)
    logger.info("Saved %s  size=%d bytes  url=%s/media/%s", suffix, len(data), PUBLIC_BASE_URL, fname)
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


def _format_video_result(open_id: str, result) -> str:
    plan_str = " → ".join(step["tool"] for step in result.plan)
    if result.video_urls and len(result.video_urls) > 1:
        clips = "\n".join(
            f"  🎞️ Shot {i+1}: {url}" for i, url in enumerate(result.video_urls)
        ) if _lang(open_id) == "en" else "\n".join(
            f"  🎞️ 第{i+1}镜：{url}" for i, url in enumerate(result.video_urls)
        )
        return _t(open_id,
            f"🎬 多镜头视频生成完成！({len(result.video_urls)} 个片段)\n\n{clips}\n\nPipeline: {plan_str}",
            f"🎬 Multi-shot done! ({len(result.video_urls)} clips)\n\n{clips}\n\nPipeline: {plan_str}",
        )
    return _t(open_id,
        f"🎬 视频已生成！\n\n🔗 {result.video_url}\n\nPipeline: {plan_str}",
        f"🎬 Your video is ready!\n\n🔗 {result.video_url}\n\nPipeline: {plan_str}",
    )


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
    """Send the interactive welcome card and put user in WAIT_LANG."""
    _set_state(open_id, _S_WAIT_LANG)
    await _send_card(chat_id, _welcome_card(open_id))


def _ensure_state(open_id: str) -> None:
    """Init state with default language (zh) for users who skipped onboarding."""
    if open_id not in _user_state:
        _set_state(open_id, _S_WAIT_IMAGE, lang="zh")


async def _on_image(open_id: str, chat_id: str, message_id: str, content: dict) -> None:
    _ensure_state(open_id)

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
    state = _get_state(open_id)
    if state["state"] != _S_WAIT_AUDIO:
        await _send(chat_id, _t(open_id,
            "请先发送草图图片。", "Please send a sketch image first."))
        return

    audio_key = content.get("file_key", "")
    _set_state(open_id, _S_WAIT_DESC,
               image_key=state["image_key"], image_msg_id=state["image_msg_id"],
               audio_key=audio_key, audio_msg_id=message_id,
               lang=_lang(open_id))
    await _send(chat_id, _t(open_id,
        "🎵 收到音频！请发送动画描述。",
        "🎵 Got the audio! Now send me a description of the animation.",
    ))


async def _on_text(open_id: str, chat_id: str, text: str) -> None:
    text = text.strip()

    # ── Hello / wake trigger — always shows the welcome card ──────────
    if text.lower() in _HELLO_WORDS:
        await _show_welcome(open_id, chat_id)
        return

    # ── Exit / reset — clear state, re-show welcome card ──────────────
    if text.lower() in _EXIT_WORDS:
        task = _running_tasks.get(open_id)
        if task and not task.done():
            task.cancel()
            _running_tasks.pop(open_id, None)
        _user_state.pop(open_id, None)   # full reset
        await _show_welcome(open_id, chat_id)
        return

    # ── Brand-new user: init with default lang (zh), don't block ────────
    _ensure_state(open_id)

    state = _get_state(open_id)
    s     = state["state"]

    # ── Waiting for language (card not clicked yet — fallback text) ────
    if s == _S_WAIT_LANG:
        # Accept typed "1"/"2" as a fallback if card buttons don't work
        if text in ("2", "english", "en", "English"):
            lang = "en"
        else:
            lang = "zh"
        _set_state(open_id, _S_WAIT_IMAGE, lang=lang)
        await _send(chat_id, _t(open_id,
            "✅ 已选择中文！发送图片或描述开始创作。🎨",
            "✅ English selected! Send an image or description to start. 🎨",
        ))
        from memory import UserMemory
        UserMemory("user_memory.json").load().update(language=lang)
        return

    # ── Cancel running generation ──────────────────────────────────────
    if any(w in text for w in _STOP_WORDS):
        task = _running_tasks.get(open_id)
        if task and not task.done():
            task.cancel()
            _running_tasks.pop(open_id, None)
            _set_state(open_id, _S_WAIT_IMAGE, lang=_lang(open_id))
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

    # ── Post-production & rating ───────────────────────────────────────
    if s == _S_WAIT_RATING:
        lang = _lang(open_id)

        # ① Numeric rating
        try:
            rating = int(text)
            if 1 <= rating <= 5:
                from memory import UserMemory
                mem = UserMemory("user_memory.json").load()
                mem.record_rating(state.get("last_prompt", ""), rating)
                stars = "⭐" * rating
                await _send(chat_id, _t(open_id,
                    f"{stars} 谢谢反馈！({rating}/5) 我会记住你的喜好。",
                    f"{stars} Thanks for the rating! ({rating}/5) I'll remember your preferences.",
                ))
                _set_state(open_id, _S_WAIT_IMAGE, lang=lang)
                await _send(chat_id, _t(open_id,
                    "发送新描述或图片，继续创作！🎨",
                    "Send another description or image to continue! 🎨",
                ))
                return
        except ValueError:
            pass

        # ② Add audio to the existing video
        if any(w in text for w in _BGM_WORDS):
            video_url = state.get("last_video_url")
            prompt    = state.get("last_prompt", "")
            if not video_url:
                await _send(chat_id, _t(open_id,
                    "找不到上一个视频，请重新生成。",
                    "No previous video found, please generate a new one.",
                ))
                return
            await _send(chat_id, _t(open_id,
                "🎵 正在添加配音，请稍候……",
                "🎵 Adding narration, please wait…",
            ))
            try:
                from tools import AddBGMTool
                bgm_ctx = {"video_url": video_url, "user_description": prompt, "lang": lang}
                bgm_out = await AddBGMTool().run(bgm_ctx)
                new_url = bgm_out["video_url"]
                await _send(chat_id, _t(open_id,
                    f"🎬 配音完成！\n\n🔗 {new_url}",
                    f"🎬 Narration added!\n\n🔗 {new_url}",
                ))
                _set_state(open_id, _S_WAIT_RATING,
                           last_prompt=prompt, last_video_url=new_url,
                           last_assets=state.get("last_assets", {}), lang=lang)
                await _send(chat_id, _rating_prompt(open_id))
            except Exception as exc:
                logger.exception("AddBGM post-process error for %s", open_id)
                await _send(chat_id, f"❌ {exc}")
            return

        # ③ Regenerate with same image + same prompt
        if any(w in text for w in _REGEN_WORDS):
            prompt    = state.get("last_prompt", "")
            last_assets = state.get("last_assets", {})
            if not prompt:
                await _send(chat_id, _t(open_id,
                    "找不到上一次的请求，请重新描述。",
                    "No previous request found, please describe again.",
                ))
                return
            _pop_state(open_id)
            await _send(chat_id, _t(open_id,
                "🔄 重新生成中……",
                "🔄 Regenerating…",
            ))
            try:
                result = await _run_agent_with_progress(
                    open_id, chat_id, user_request=prompt, assets=last_assets
                )
                await _send(chat_id, _format_video_result(open_id, result))
                _set_state(open_id, _S_WAIT_RATING,
                           last_prompt=prompt, last_video_url=result.video_url,
                           last_assets=last_assets, lang=lang)
                await _send(chat_id, _rating_prompt(open_id))
            except asyncio.CancelledError:
                await _send(chat_id, _t(open_id, "⛔ 已取消。", "⛔ Cancelled."))
                _set_state(open_id, _S_WAIT_IMAGE, lang=lang)
            except Exception as exc:
                logger.exception("Regen error for %s", open_id)
                await _send(chat_id, f"❌ {exc}")
                _set_state(open_id, _S_WAIT_IMAGE, lang=lang)
            return

        # ④ Any other text → skip rating, treat as new request
        _set_state(open_id, _S_WAIT_IMAGE, lang=lang)
        s = _S_WAIT_IMAGE

    # ── Text-to-video (no image) ───────────────────────────────────────
    if s == _S_WAIT_IMAGE:
        if not text:
            await _send(chat_id, _t(open_id,
                "请发送描述文字或图片。",
                "Please send a description or an image.",
            ))
            return
        lang = _lang(open_id)
        _pop_state(open_id)
        await _send(chat_id, _t(open_id,
            "⏳ 正在生成视频，通常需要 1–3 分钟……",
            "⏳ Generating your video — this usually takes 1–3 minutes…",
        ))
        try:
            result = await _run_agent_with_progress(open_id, chat_id, user_request=text, assets={})
            await _send(chat_id, _format_video_result(open_id, result))
            _set_state(open_id, _S_WAIT_RATING,
                       last_prompt=result.last_prompt or text,
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
    data = _pop_state(open_id)
    await _send(chat_id, _t(open_id,
        "⏳ 收到！正在生成视频，通常 1–3 分钟。发送「取消」可中止。",
        "⏳ Got it! Generating your video — usually 1–3 min. Send 'cancel' to stop.",
    ))

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
        await _send(chat_id, _format_video_result(open_id, result))
        _set_state(open_id, _S_WAIT_RATING,
                   last_prompt=result.last_prompt or text,
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

    Feishu card callback body (v1/v2 both handled):
      {
        "open_id": "ou_xxx",
        "open_chat_id": "oc_xxx",
        "action": { "value": { "action": "set_lang", "lang": "zh", "open_id": "..." } },
        ...
      }
    """
    logger.debug("handle_card_action body keys: %s", list(body.keys()))

    # ── pull action value — handle both legacy and card-kit formats ──────
    action = body.get("action") or {}
    # card kit v2 nests value differently in some SDK versions
    value  = action.get("value") or action.get("form_value") or {}
    if isinstance(value, str):
        try:
            import json as _json
            value = _json.loads(value)
        except Exception:
            value = {}

    if value.get("action") != "set_lang":
        logger.info("Card action ignored (not set_lang): value=%s", value)
        return {"toast": {"type": "info", "content": "OK"}}

    lang    = value.get("lang", "zh")
    # open_id can be in value (we embed it) or top-level body
    open_id = value.get("open_id") or body.get("open_id", "")
    chat_id = body.get("open_chat_id", "")

    logger.info("Card set_lang: lang=%s open_id=%s chat_id=%s", lang, open_id, chat_id)

    if not open_id:
        return {"toast": {"type": "error", "content": "Missing open_id"}}

    _set_state(open_id, _S_WAIT_IMAGE, lang=lang)

    from memory import UserMemory
    UserMemory("user_memory.json").load().update(language=lang)

    if chat_id:
        asyncio.create_task(_send(chat_id, _t(open_id,
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
        asyncio.create_task(_on_image(open_id, chat_id, message_id, content))
    elif msg_type in ("audio", "file"):
        asyncio.create_task(_on_audio(open_id, chat_id, message_id, content))
    elif msg_type in ("text", "post"):
        text = (content.get("text", "") if msg_type == "text"
                else _extract_post_text(content))
        asyncio.create_task(_on_text(open_id, chat_id, text))
    else:
        asyncio.create_task(_send(chat_id, _t(open_id,
            f"暂不支持该消息类型：{msg_type}",
            f"Unsupported message type: {msg_type}",
        )))

    # Return immediately — Feishu needs a response within 3 s
    return {}
