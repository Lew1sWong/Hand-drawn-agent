"""
Planner — asks DeepSeek to produce a typed, structured execution plan.

DeepSeek is called with `response_format=json_object`, so it always returns
a JSON object (not an array). We ask it to wrap the steps list under a
"steps" key and unwrap after parsing.

Plan wire format (what DeepSeek returns):
  {
    "steps": [
      {
        "tool":   "<tool_name>",
        "inputs": { <planner-settable overrides only> },
        "reason": "<one-sentence rationale>"
      },
      ...
    ]
  }

Tool outputs (video_url, task_id, etc.) are piped automatically by the
executor — the planner must NOT try to hard-code them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from memory import UserMemory
from tools import ALL_TOOLS

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
# System prompt builder
# ---------------------------------------------------------------------------

def _build_system_prompt(memory: UserMemory, available_assets: set[str]) -> str:
    tool_docs = "\n\n".join(
        f"### {t.name}\n{t.description}\nPlanner-settable inputs schema: {json.dumps(t.input_schema)}"
        for t in ALL_TOOLS
    )
    assets_str = ", ".join(sorted(available_assets)) if available_assets else "none provided"

    return f"""\
You are the planning module of a hand-drawn animation AI agent.
Produce a JSON execution plan that satisfies the user's request using the available tools.

=== Available context assets (already in scope, do NOT put these in inputs) ===
{assets_str}

=== Available tools ===
{tool_docs}

=== User memory ===
{memory.as_context_str()}

=== Output format ===
Return a single JSON object with one key "steps" containing a list:
{{
  "steps": [
    {{"tool": "<name>", "inputs": {{<planner overrides only>}}, "reason": "<one sentence>"}},
    ...
  ]
}}

=== Planning rules ===
1. Use only the tool names listed above — never invent new ones.
2. "inputs" contains ONLY values the planner must set (e.g. duration, n_shots).
   Do NOT include image_url, audio_url, video_url, or enhanced_prompt — they
   are resolved automatically from context.
3. Tool outputs are piped forward: a video_url produced by step N is available
   as input to step N+1 automatically.
4. MULTI-SHOT RULE (highest priority check):
   If the user provides a detailed script with multiple scenes, OR explicitly asks
   for "多镜头", "多场景", "分镜", "multi-shot", or "multi-scene" → use multi_shot_video
   (single step). Set n_shots to the number of distinct scenes (2–4).
5. If "image_url" is NOT in available assets (and not multi-shot) → use text_to_video.
6. If "image_url" IS available and "audio_url" IS available → use audio_portrait.
7. If "image_url" IS available and "audio_url" is NOT available:
   - User wants character to speak/sing → plan: tts then audio_portrait (set tts_text in tts inputs).
   - User wants silent animation → use image_to_video.
8. tts always runs BEFORE audio_portrait; set tts_text to the words the character should say.
9. Minimum 1 step, maximum 3 steps.
10. Output ONLY the JSON object — no markdown, no extra commentary.
"""


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

async def make_plan(
    user_request: str,
    memory: UserMemory,
    available_assets: set[str],
    deepseek_api_key: str,
) -> Plan:
    """
    Call DeepSeek to produce a Plan for the given user_request.

    Args:
        user_request:      Raw user message (any language).
        memory:            Loaded UserMemory instance.
        available_assets:  Keys present in the initial execution context
                           (e.g. {"image_url", "audio_url", "user_description"}).
        deepseek_api_key:  API key for the DeepSeek service.

    Returns:
        A list of PlanStep objects ready for the executor.
    """
    client = AsyncOpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com",
    )
    system_prompt = _build_system_prompt(memory, available_assets)

    resp = await client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=512,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_request},
        ],
    )

    raw = resp.choices[0].message.content.strip()
    logger.debug("Planner raw output: %s", raw)

    parsed = json.loads(raw)
    steps_raw = parsed.get("steps")
    if not isinstance(steps_raw, list):
        raise ValueError(f"Planner returned unexpected shape (no 'steps' list): {parsed}")

    plan: Plan = [
        PlanStep(
            tool=s["tool"],
            inputs=s.get("inputs") or {},
            reason=s.get("reason", ""),
        )
        for s in steps_raw
    ]

    logger.info(
        "Plan produced: %s",
        [(s.tool, s.reason) for s in plan],
    )
    return plan
