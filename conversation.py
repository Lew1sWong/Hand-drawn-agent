"""
Conversational gateway — two LLM-driven decision points:

PRE-GENERATION  (ConversationSession.process)
  "chat"    — ask a clarifying question or answer a general inquiry
  "execute" — call the animation pipeline with a refined description

POST-GENERATION  (ConversationSession.process_post_gen)
  "regenerate"  — re-generate with visual/style/param modifications
  "add_bgm"     — add narration or sound effects to the existing video
  "rate"        — record user's satisfaction score (1–5)
  "new_request" — completely different subject; start fresh
  "chat"        — answer questions or handle unclear input

Both paths share the same session history so the LLM has full context.
The session is cleared after each execute / regenerate / add_bgm so each
generation cycle starts with a clean slate.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are the conversational front-end of a hand-drawn animation video AI.
Your sole purpose: understand what animated video the user wants, then decide
whether you need one more clarifying question or have enough to start generating.

=== What the pipeline can do ===
• text_to_video   — generate animation from a scene description (no image needed)
• image_to_video  — animate an uploaded sketch or photo
• audio_portrait  — lip-sync talking portrait (needs image + audio)
• multi_shot      — multi-scene cinematic video (needs a detailed script)
• add_bgm         — add narration or ambient sound to any video

=== Assets already uploaded by the user ===
{assets}

=== Decision ===
Return ONLY a JSON object (no markdown, no extra text):
{{
  "action":       "execute" | "chat",
  "reply":        "<message to send to user — 1–3 sentences, SAME language as user>",
  "user_request": "<full refined animation description — ONLY when action=execute, else empty string>",
  "audio_mode":   "sfx" | "narration" | "mixed" | null,
  "tts_text":     "<verbatim character dialogue to TTS, or null>"
}}

=== Audio intent extraction (for action=execute) ===
Set audio_mode based on the FULL semantic understanding of the request — NOT keyword matching:
- null       → no audio requested
- "sfx"      → only ambient/background/environmental sound (风声, 雨声, engine, animals…)
- "narration"→ ANY form of character speech or recitation, including but not limited to:
               吟诗、低声道、喃喃、叹道、大喊、高呼、说道、念道、唱道、哽咽、歌唱
               say, whisper, exclaim, cry out, recite, chant, sing, speak, murmur, declare…
- "mixed"    → both ambient sound AND character speech are requested

Set tts_text to the EXACT verbatim words the character should say/recite/sing,
taken directly from the user's text (e.g., text inside quotes "..." 「...」 or 『...』).
Do NOT rewrite or paraphrase. If multiple quotes exist, concatenate with a space.
If no explicit dialogue text is given (character speaks but words not specified), set null.

▸ Use action=EXECUTE when ANY of these is true:
  • User described a specific scene, mood, character or story (8+ meaningful words)
  • User confirmed or approved a previous suggestion
  • ≥ 3 user turns have already happened on the same request — stop asking, just do it
  • Request mentions a concrete visual ("樱花飘落", "sunset beach", "cat chasing butterfly")

▸ Use action=CHAT when ALL of these apply:
  • Request is completely vague with zero visual content ("做个视频", "make a video", "帮我做")
  • User is asking a general capability question ("你能做什么?", "what can you do?")
  • User clearly wants to refine/change their idea (first ≤2 turns only)

=== Rules ===
• Reply language MUST match the user's language (Chinese ↔ English; never mix)
• For CHAT: ask exactly ONE specific question (scene content, style, or mood)
• For EXECUTE: reply MUST include a brief intent summary so the user can verify the agent
  understood correctly, followed by a generation confirmation. Format (adapt language):

  我理解您想要：
  • 画面：<1 sentence describing visual content including any characters>
  • 音效/旁白：<what audio, and the exact text to be read aloud if the user quoted any>
  ⚠ 提示：<only include if there is a relevant capability limitation, e.g. 嘴部动画需上传人物肖像图片>
  马上生成！

  If no capability limitation applies, omit the ⚠ line entirely.
  Capability limitations to mention when relevant:
  - Mouth/lip animation (嘴部动画/lip sync) requires an uploaded portrait image — without it,
    the character will appear but won't have animated lips. Mention this when user requests
    mouth animation but no image_url is in assets.
  - Character speech from text_to_video is visual only — the character will be shown in a
    reciting pose but audio is a separate TTS overlay. This is normal behavior.

• user_request for execute must be a vivid, complete scene description ready for the AI
  — expand abbreviations, add atmosphere/lighting if vague, keep ≤120 words
• When assets include image_url: the video WILL animate that image; acknowledge it
• When assets include audio_url: the video will be a lip-sync portrait; acknowledge it
• NEVER discuss anything unrelated to animation video creation
• Do NOT ask multiple questions at once

=== Multi-shot detection & large-task splitting ===
Evaluate the user's request for complexity. Apply the rules below IN ORDER:

1. MULTI-SCENE STORYBOARD — if the description mentions 2+ distinct scenes, locations,
   time periods, or camera angles, add this tip to the reply (after the intent summary):
   zh: "💡 小技巧：您的内容有多个场景，推荐用分镜格式描述，例如：\n「镜头1：... 镜头2：... 镜头3：...」\n这样可以生成多镜头视频，效果更好！"
   en: "💡 Tip: Your request has multiple scenes — try the storyboard format:\n\"Shot 1: … Shot 2: … Shot 3: …\"\nThis generates a multi-shot video with better results!"

2. OVERLOADED SINGLE REQUEST — if the user packs 4+ unrelated elements (characters,
   objects, actions, effects) into one scene, suggest splitting into separate requests:
   zh: "💡 任务量较大，建议分成多次生成：先发一条描述第一个场景，完成后再发第二个，效果更稳定。"
   en: "💡 This is a large request — consider splitting it: send the first scene now, then the next after it's done. Results are more reliable."
   Then proceed with action=execute for what the user asked (do NOT refuse).

3. SIMPLE REQUEST — no tip needed; proceed normally.
"""


# ---------------------------------------------------------------------------
# Post-generation system prompt
# ---------------------------------------------------------------------------

_POST_GEN_SYSTEM = """\
You are the post-production assistant of a hand-drawn animation AI.
The user just received a generated video and is providing feedback or a follow-up request.

=== Original prompt that produced the video ===
{last_prompt}

=== Assets that were used ===
{assets}

=== Available actions ===
• "regenerate"  — re-generate with visual/style/detail modifications (uses same image if available)
• "add_bgm"     — add audio to the EXISTING video (narration or sound effects)
• "rate"        — user is rating the video quality (1–5 scale)
• "new_request" — completely different scene/subject; start fresh
• "chat"        — answer questions, clarify, or handle unclear input

Return ONLY a JSON object (no markdown):
{{
  "action":          "regenerate" | "add_bgm" | "rate" | "new_request" | "chat",
  "reply":           "<1–3 sentences to send the user, SAME language as user>",
  "rating":          <integer 1–5 or null>,
  "bgm_mode":        "sfx" | "narration",
  "bgm_description": "<what audio to produce — keep in user's language>",
  "refined_request": "<complete modified animation description for regenerate>",
  "param_overrides": {{"duration": 4}} or {{}}
}}

▸ "regenerate" when user wants to tweak the video:
  • Visual details: colours, brightness, mood, lighting, objects ("更暗", "换蓝色", "加樱花")
  • Duration: "改成8秒", "longer", "shorter", "4秒" → param_overrides: {{"duration": 4|8}}
  • Style: "更戏剧化", "更平静", "换个镜头角度"
  • Content tweaks: add/remove elements, change character details
  • refined_request MUST be a complete self-contained description = original prompt + changes

▸ "add_bgm" when user mentions audio for the EXISTING video:
  • 旁白 / 配音 / 吟诗 / 朗诵 / 台词 / voice-over / narration / recite → bgm_mode="narration"
  • 音效 / 背景音 / music / sound effects / ambient → bgm_mode="sfx"
  • bgm_description: describe what audio the user wants

▸ "rate" when user expresses satisfaction level:
  • Numeric: 1–5 directly
  • Semantic: 很棒/perfect/excellent → 5 | 不错/good → 4 | 还行/okay → 3 | 差/bad → 2 | 很差/awful → 1
  • Must set integer rating field

▸ "new_request" when the subject/scene is completely different from the original
  (e.g. original was "beach sunset", user says "make me a dragon in a forest")

▸ "chat" for questions, thanks, unclear input, or anything else

Rules:
  • Reply in the SAME language as the user (never mix zh/en)
  • For regenerate: always produce a complete refined_request, NOT just the diff
  • param_overrides only includes "duration" (4 or 8) when explicitly stated
  • For add_bgm: bgm_description should capture what the user wants in their words
  • Never include fields that are not relevant to the chosen action (set to null / empty)
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ConvDecision:
    action:       str        # "execute" | "chat"
    reply:        str        # message to send to user
    user_request: str = ""   # refined description; only meaningful for execute
    audio_mode:   str | None = None  # "sfx"|"narration"|"mixed"|None — extracted by LLM
    tts_text:     str | None = None  # verbatim character dialogue for TTS


@dataclass
class PostGenDecision:
    action:          str               # "regenerate"|"add_bgm"|"rate"|"new_request"|"chat"
    reply:           str               # message to send to user
    rating:          int | None = None # 1–5; only for "rate"
    bgm_mode:        str        = "sfx"
    bgm_description: str        = ""   # what audio to produce
    refined_request: str        = ""   # modified description; only for "regenerate"
    param_overrides: dict       = field(default_factory=dict)  # e.g. {"duration": 8}


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class ConversationSession:
    """
    Per-user conversation history buffer.

    Usage:
        session = ConversationSession(open_id)
        decision = await session.process(user_message, available_assets)
        # decision.action == "chat"    → send decision.reply, wait for next turn
        # decision.action == "execute" → send decision.reply, run agent with decision.user_request
        if decision.action == "execute":
            session.clear()
    """

    _MAX_HISTORY = 12   # total messages (user + assistant) to keep

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._history: list[dict[str, str]] = []

    # ── history management ────────────────────────────────────────────────

    def add_user(self, content: str) -> None:
        self._history.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, content: str) -> None:
        self._history.append({"role": "assistant", "content": content})
        self._trim()

    def _trim(self) -> None:
        if len(self._history) > self._MAX_HISTORY:
            self._history = self._history[-self._MAX_HISTORY:]

    def clear(self) -> None:
        self._history.clear()

    @property
    def exchange_count(self) -> int:
        """Number of user turns so far."""
        return sum(1 for m in self._history if m["role"] == "user")

    # ── core decision ─────────────────────────────────────────────────────

    async def process(
        self,
        user_message: str,
        available_assets: dict[str, Any],
    ) -> ConvDecision:
        """
        Record user_message, call DeepSeek, return a ConvDecision.

        Falls back to execute (with the raw user message) on any LLM error so
        the pipeline always has a chance to run.
        """
        self.add_user(user_message)

        assets_desc = (
            ", ".join(sorted(available_assets.keys()))
            if available_assets else "none"
        )
        system = _SYSTEM.format(assets=assets_desc)

        client = AsyncOpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model="deepseek-chat",
                    max_tokens=300,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        *self._history,
                    ],
                ),
                timeout=30.0,
            )
            raw = resp.choices[0].message.content.strip()
            logger.debug("[Conv %s] LLM → %s", self.session_id, raw)
            data = json.loads(raw)
        except Exception as exc:
            logger.warning(
                "[Conv %s] LLM failed (%s) — falling back to execute",
                self.session_id, exc,
            )
            data = {
                "action":       "execute",
                "reply":        _fallback_reply(user_message),
                "user_request": user_message,
            }

        action       = data.get("action", "execute")
        reply        = data.get("reply", "")
        user_request = data.get("user_request") or user_message
        audio_mode   = data.get("audio_mode") or None   # "sfx"|"narration"|"mixed"|None
        tts_text     = data.get("tts_text") or None

        self.add_assistant(reply)
        return ConvDecision(
            action=action,
            reply=reply,
            user_request=user_request,
            audio_mode=audio_mode,
            tts_text=tts_text,
        )

    # ── post-generation decision ──────────────────────────────────────────

    async def process_post_gen(
        self,
        user_message: str,
        last_prompt: str,
        available_assets: dict[str, Any],
    ) -> PostGenDecision:
        """
        Called after a video has been delivered.  Decides what to do with
        the user's follow-up (modify, add audio, rate, start fresh, or chat).

        Falls back to "chat" on LLM error so we never silently drop a message.
        """
        self.add_user(user_message)

        assets_desc = (
            ", ".join(sorted(available_assets.keys()))
            if available_assets else "none"
        )
        system = _POST_GEN_SYSTEM.format(
            last_prompt=last_prompt or "(not available)",
            assets=assets_desc,
        )

        client = AsyncOpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model="deepseek-chat",
                    max_tokens=400,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        *self._history,
                    ],
                ),
                timeout=30.0,
            )
            raw = resp.choices[0].message.content.strip()
            logger.debug("[PostGen %s] LLM → %s", self.session_id, raw)
            data = json.loads(raw)
        except Exception as exc:
            logger.warning(
                "[PostGen %s] LLM failed (%s) — falling back to chat",
                self.session_id, exc,
            )
            fallback_reply = (
                "抱歉，我没有理解您的意思，能再说一遍吗？"
                if _is_chinese(user_message) else
                "Sorry, I didn't quite understand — could you rephrase that?"
            )
            reply = fallback_reply
            self.add_assistant(reply)
            return PostGenDecision(action="chat", reply=reply)

        action          = data.get("action", "chat")
        reply           = data.get("reply", "")
        rating          = data.get("rating")
        bgm_mode        = data.get("bgm_mode", "sfx")
        bgm_description = data.get("bgm_description", "") or user_message
        refined_request = data.get("refined_request", "") or last_prompt
        param_overrides = data.get("param_overrides") or {}

        # Clamp rating to valid range
        if rating is not None:
            try:
                rating = max(1, min(5, int(rating)))
            except (TypeError, ValueError):
                rating = None

        # Only allow safe param overrides
        safe_overrides = {}
        if "duration" in param_overrides:
            try:
                d = int(param_overrides["duration"])
                safe_overrides["duration"] = d if d in (4, 8) else (8 if d > 4 else 4)
            except (TypeError, ValueError):
                pass

        self.add_assistant(reply)
        return PostGenDecision(
            action=action,
            reply=reply,
            rating=rating,
            bgm_mode=bgm_mode,
            bgm_description=bgm_description,
            refined_request=refined_request,
            param_overrides=safe_overrides,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fallback_reply(text: str) -> str:
    if _is_chinese(text):
        return "好的，正在生成……"
    return "Got it, generating…"


def _is_chinese(text: str) -> bool:
    return any("一" <= c <= "鿿" for c in text)
