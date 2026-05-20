"""
Three-layer planner:
  1. Deterministic rules  — fast, no LLM, confidence=1.0 cases
  2. LLM call            — DeepSeek, only for ambiguous cases
  3. Validate + patch    — inject rule conclusions, strip hallucinated tools
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from memory import UserMemory
from .rules import AudioMode, RuleResult, VideoMode, classify  # noqa: F401
from .prompt import build_system_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    tool:   str
    inputs: dict[str, Any] = field(default_factory=dict)
    reason: str            = ""


Plan = list[PlanStep]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tool_docs() -> str:
    from tools import ALL_TOOLS
    return "\n\n".join(
        f"### {t.name}\n{t.description}\nPlanner-settable inputs schema: {json.dumps(t.input_schema)}"
        for t in ALL_TOOLS
    )


def _valid_tool_names() -> set[str]:
    from tools import TOOL_MAP
    return set(TOOL_MAP.keys())


def _build_plan_from_rule(rule: RuleResult) -> Plan:
    """Construct a plan directly from deterministic rule results (confidence=1.0)."""
    steps: Plan = []

    if rule.video_mode == VideoMode.AUDIO_PORTRAIT:
        steps = [PlanStep(tool="audio_portrait", reason="image + audio asset detected")]

    elif rule.video_mode == VideoMode.MULTI_SHOT:
        inputs = {"n_shots": rule.n_shots} if rule.n_shots else {}
        steps = [PlanStep(tool="multi_shot_video", inputs=inputs, reason="multi-shot keywords detected")]

    elif rule.video_mode == VideoMode.FIGURINE:
        steps = [
            PlanStep(tool="figurine_to_anime", reason="figurine photo detected"),
            PlanStep(tool="image_to_video",    reason="animate anime render"),
        ]

    elif rule.video_mode == VideoMode.IMAGE_TO_VIDEO:
        steps = [PlanStep(tool="image_to_video", reason="image present in context")]

    elif rule.video_mode == VideoMode.TEXT_TO_VIDEO:
        steps = [PlanStep(tool="text_to_video", reason="text-only request, no image")]

    # Append add_bgm if audio was requested (but not for audio_portrait, which already has audio)
    if rule.wants_audio and rule.video_mode != VideoMode.AUDIO_PORTRAIT:
        if rule.audio_mode == AudioMode.MIXED:
            # Both SFX and narration requested — chain two add_bgm steps
            steps.append(PlanStep(
                tool="add_bgm",
                inputs={"mode": "sfx"},
                reason="ambient sound effects requested",
            ))
            steps.append(PlanStep(
                tool="add_bgm",
                inputs={"mode": "narration"},
                reason="character narration/recitation requested",
            ))
        else:
            mode = (
                "sfx"       if rule.audio_mode == AudioMode.SFX
                else "narration" if rule.audio_mode == AudioMode.NARRATION
                else "sfx"   # ambiguous → safe default
            )
            steps.append(PlanStep(
                tool="add_bgm",
                inputs={"mode": mode},
                reason=f"audio requested ({mode})",
            ))

    return steps


def _patch_plan(plan: Plan, rule: RuleResult) -> Plan:
    """Inject high-confidence rule conclusions into the LLM-produced plan."""
    if rule.audio_mode == AudioMode.UNKNOWN:
        return plan

    bgm_steps = [s for s in plan if s.tool == "add_bgm"]

    if rule.audio_mode == AudioMode.MIXED and len(bgm_steps) == 1:
        # LLM produced only one add_bgm but user wants both — insert a second one
        idx = next(i for i, s in enumerate(plan) if s.tool == "add_bgm")
        plan[idx].inputs["mode"] = "sfx"
        plan.insert(idx + 1, PlanStep(
            tool="add_bgm",
            inputs={"mode": "narration"},
            reason="character narration/recitation (patched by rules)",
        ))
        logger.info("MIXED audio: patched single add_bgm into sfx+narration pair")
    else:
        mode = "sfx" if rule.audio_mode == AudioMode.SFX else "narration"
        for step in bgm_steps:
            step.inputs["mode"] = mode
        if bgm_steps:
            logger.info("Audio mode patched by rules: %s", mode)

    return plan


def _validate_plan(plan: Plan) -> Plan:
    """Remove hallucinated tool names; raise if nothing survives."""
    valid = _valid_tool_names()
    cleaned = [s for s in plan if s.tool in valid]
    if not cleaned:
        raise ValueError(
            f"Planner produced no valid steps (all tools unknown): {[s.tool for s in plan]}"
        )
    return cleaned


async def _call_deepseek(
    user_request: str,
    memory: UserMemory,
    available_assets: set[str],
    api_key: str,
) -> Plan:
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    system = build_system_prompt(_tool_docs(), memory, available_assets, query=user_request)

    resp = await client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=512,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_request},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    logger.debug("Planner LLM output: %s", raw)

    parsed = json.loads(raw)
    steps_raw = parsed.get("steps")
    if not isinstance(steps_raw, list):
        raise ValueError(f"Planner returned unexpected shape (no 'steps' list): {parsed}")

    return [
        PlanStep(
            tool=s["tool"],
            inputs=s.get("inputs") or {},
            reason=s.get("reason", ""),
        )
        for s in steps_raw
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def make_plan(
    user_request: str,
    memory: UserMemory,
    available_assets: set[str],
    deepseek_api_key: str,
    audio_mode_hint: str | None = None,
) -> Plan:
    has_image = "image_url" in available_assets
    has_audio = "audio_url" in available_assets

    # Layer 1: deterministic rules
    rule = classify(user_request, has_image, has_audio)

    # Conversation layer's LLM understanding overrides keyword-based audio detection.
    # This handles any expression of character speech (低声道/喃喃/whisper/exclaim/…)
    # that keyword lists would miss.
    if audio_mode_hint:
        _mode_map = {
            "sfx":       AudioMode.SFX,
            "narration": AudioMode.NARRATION,
            "mixed":     AudioMode.MIXED,
        }
        if audio_mode_hint in _mode_map:
            rule.audio_mode  = _mode_map[audio_mode_hint]
            rule.wants_audio = True
            logger.info("Audio mode from conversation layer: %s", audio_mode_hint)

    if rule.confidence == 1.0:
        plan = _build_plan_from_rule(rule)
        logger.info("Plan from rules (LLM skipped): %s", [(s.tool, s.reason) for s in plan])
        return plan

    # Layer 2: LLM (for ambiguous cases — e.g. text_to_video vs multi_shot)
    plan = await _call_deepseek(user_request, memory, available_assets, deepseek_api_key)

    # Layer 3: validate + patch with rule conclusions
    plan = _patch_plan(plan, rule)
    plan = _validate_plan(plan)

    logger.info("Plan from LLM (patched): %s", [(s.tool, s.reason) for s in plan])
    return plan
