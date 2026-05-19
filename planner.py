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

=== CRITICAL — Image presence override (read this first) ===
If "image_url" IS listed in the Available context assets above, the user has uploaded
a photo or sketch and MUST have it animated. You MUST use image_to_video (or
figurine_to_anime → image_to_video for figurines) as the video-generation step.
NEVER substitute text_to_video when image_url is present — even if the user writes a
long visual description. That description tells you HOW to animate their image, not
what to generate from scratch. Violating this rule throws away the user's upload.

=== Planning rules (apply top-to-bottom, first match wins) ===
1. Use only the tool names listed above — never invent new ones.
2. "inputs" contains ONLY values the planner must set (e.g. duration, n_shots).
   Do NOT include image_url, audio_url, video_url, or enhanced_prompt — the
   executor resolves those automatically from context.
3. Tool outputs are piped forward automatically. figurine_to_anime overwrites
   image_url, so image_to_video after it picks up the anime render.

4. ADD SOUND — append add_bgm as final step when requested:
   If the user mentions 声音 / 配音 / 旁白 / 音效 / 音乐 / sound / audio / narration / music,
   append add_bgm as the LAST step after any video-generating tool
   (image_to_video, text_to_video, multi_shot_video).
   Do NOT add add_bgm after audio_portrait (it already produces audio).

5. NO IMAGE → text_to_video (always, no exceptions):
   If "image_url" is NOT in available assets → use text_to_video (single step).
   This applies even if the user mentions 手办 / figurine / figure — without an
   uploaded photo there is nothing to convert.

6. MULTI-SHOT (image_url NOT required):
   User provides a multi-scene script OR says 多镜头 / 分镜 / multi-shot →
   use multi_shot_video (single step, set n_shots = number of scenes, 2–4).

7. FIGURINE + IMAGE → figurine_to_anime → image_to_video:
   "image_url" IS in available assets AND user mentions 手办 / figurine / figure /
   toy → 2-step plan: figurine_to_anime then image_to_video.

8. IMAGE + AUDIO → audio_portrait (single step).

9. IMAGE only, user wants speech/singing → tts then audio_portrait (2 steps,
   set tts_text to the words the character should say).

10. IMAGE only → image_to_video:
    "image_url" IS in available assets → image_to_video. No exceptions.
    (add_bgm appended per rule 4 if sound is requested.)

11. tts must always run BEFORE audio_portrait.
12. Minimum 1 step, maximum 4 steps (add_bgm may be appended as a final step).
13. Output ONLY the JSON object — no markdown, no extra text.
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
