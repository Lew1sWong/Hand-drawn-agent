"""
Executor — walks a Plan step by step, piping outputs into a rolling context.

Context lifecycle:
  initial_ctx  ← user-supplied assets  (image_url, audio_url, user_description, …)
       ↓  step 1 inputs merged in
       ↓  prompt enhancement if needed
       ↓  tool.run(ctx) → output dict merged back
       ↓  step 2 inputs merged in
       ↓  tool.run(ctx) → output dict merged back
       …
  final_ctx   returned to orchestrator

Prompt enhancement is performed lazily — only for tools that require an
`enhanced_prompt` and only once per run (subsequent steps inherit it from ctx).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from openai import AsyncOpenAI

from planner import Plan
from tools import TOOL_MAP
from tools.base import ToolError

logger = logging.getLogger(__name__)

# Tools that need an LLM-enhanced prompt before they can run.
_NEEDS_PROMPT = {"image_to_video", "text_to_video"}  # tts/audio_portrait use raw user text

_ENHANCE_SYSTEM = """\
You are a cinematographer writing prompts for AI hand-drawn animation video generation.
Convert the user's description into a vivid English prompt (60–90 words).

Rules:
1. Always start with: "Hand-drawn animation style, 2D sketch art,"
2. Choose ONE specific camera movement that fits the scene:
   "slow push-in", "gentle pull-back", "smooth pan left", "smooth pan right",
   "overhead crane shot descending", "low-angle tracking shot", "static wide shot",
   "rack focus from foreground to background", "handheld close-up", "360 orbit"
3. Add cinematic lighting: "golden-hour rim light", "soft dappled light through leaves",
   "dramatic side-lighting", "misty volumetric fog", "moonlit silhouette",
   "warm lantern glow", "cool blue-tinted dawn"
4. Describe visible motion, light, and atmosphere. If a character or person is present
   (人物 / 角色 / person / character / figure / rider), include them with a clear visual
   description of their appearance and posture. Omit backstory and dialogue text.
5. Use present tense and concrete visual language.
6. Always end with: "consistent line-art aesthetic, fluid animation."
Return only the prompt text — nothing else."""


async def _enhance_prompt(user_description: str, api_key: str) -> str:
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    resp = await client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=256,
        messages=[
            {"role": "system", "content": _ENHANCE_SYSTEM},
            {"role": "user",   "content": user_description},
        ],
    )
    return resp.choices[0].message.content.strip()


def _extract_quoted_text(text: str) -> str | None:
    """Return the first quoted string found in text (Chinese or ASCII quotes)."""
    m = re.search(r'["“”「」](.*?)["“”「」]', text)
    return m.group(1).strip() if m else None


async def execute_plan(
    plan: Plan,
    initial_ctx: dict[str, Any],
    deepseek_api_key: str,
) -> dict[str, Any]:
    """
    Execute a Plan produced by the planner.

    Args:
        plan:              Ordered list of PlanStep objects.
        initial_ctx:       Starting context — typically user assets + user_description.
        deepseek_api_key:  Used for prompt enhancement when needed.

    Returns:
        The final context dict, which contains all accumulated outputs
        (video_url, task_id, enhanced_prompt, …).

    Raises:
        ValueError      — unknown tool name in plan.
        RuntimeError    — Volcengine API error.
        TimeoutError    — polling exceeded ceiling.
    """
    ctx: dict[str, Any] = dict(initial_ctx)
    total = len(plan)

    # Extract verbatim quoted text from user description so TTS uses the
    # user's exact words instead of an LLM-rewritten narration.
    if "tts_text" not in ctx:
        quoted = _extract_quoted_text(ctx.get("user_description", ""))
        if quoted:
            ctx["tts_text"] = quoted
            logger.info("Extracted quoted tts_text: %s", quoted)

    for i, step in enumerate(plan, 1):
        tool = TOOL_MAP.get(step.tool)
        if tool is None:
            raise ValueError(
                f"Step {i}/{total}: unknown tool '{step.tool}'. "
                f"Valid tools: {list(TOOL_MAP)}"
            )

        # Merge planner overrides into context (e.g. duration, effect name)
        if step.inputs:
            ctx.update(step.inputs)

        # Lazy prompt enhancement — only when needed and not already in ctx
        if step.tool in _NEEDS_PROMPT and "enhanced_prompt" not in ctx:
            desc = ctx.get("user_description", "")
            logger.info("Step %d/%d: enhancing prompt for '%s'", i, total, step.tool)
            ctx["enhanced_prompt"] = await _enhance_prompt(desc, deepseek_api_key)
            logger.info("Enhanced prompt: %s", ctx["enhanced_prompt"])

        # Guard: figurine_to_anime + image_to_video both require image_url.
        # If the planner chose either without an image, fall back to
        # text_to_video and stop — do NOT continue to remaining steps.
        if step.tool in ("figurine_to_anime", "image_to_video") and not ctx.get("image_url"):
            logger.warning(
                "Step %d/%d: '%s' needs image_url (not in context) — "
                "falling back to text_to_video and stopping", i, total, step.tool
            )
            from tools import TextToVideoTool
            if "enhanced_prompt" not in ctx:
                ctx["enhanced_prompt"] = await _enhance_prompt(
                    ctx.get("user_description", ""), deepseek_api_key
                )
            output = await TextToVideoTool().run(ctx)
            ctx.update(output)
            logger.info("Fallback text_to_video done — keys: %s", list(output))
            break  # remaining steps (e.g. image_to_video after figurine_to_anime) are skipped

        try:
            tool.check_ctx(ctx)
        except ToolError as exc:
            raise ValueError(
                f"Step {i}/{total}: pre-flight failed for '{step.tool}': {exc}"
            ) from exc

        logger.info(
            "Step %d/%d: running '%s'  reason='%s'",
            i, total, step.tool, step.reason,
        )
        output = await tool.run(ctx)
        ctx.update(output)
        logger.info("Step %d/%d done — produced keys: %s", i, total, list(output))

    return ctx
