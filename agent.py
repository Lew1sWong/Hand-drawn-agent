"""
Hand-drawn Animation Agent — Orchestrator
==========================================
Single public entry point: run_agent(user_request, assets) -> AgentResult

Pipeline:
  1. Load persistent user memory (user_memory.json)
  2. Planner  — DeepSeek returns a JSON plan [{tool, inputs, reason}, …]
  3. Executor — walks each step, piping outputs into a rolling context dict
  4. Memory   — persist the enhanced_prompt that produced a successful video
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from openai import AsyncOpenAI

from executor import execute_plan
from memory import UserMemory
from planner import Plan, make_plan

logger = logging.getLogger(__name__)

DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    video_url:       Optional[str]      = None
    video_urls:      list[str]          = field(default_factory=list)  # multi-shot clips
    plan:            list[dict]         = field(default_factory=list)
    enhanced_prompt: Optional[str]      = None
    last_prompt:     Optional[str]      = None   # for rating storage
    final_context:   dict[str, Any]     = field(default_factory=dict)
    error:           Optional[str]      = None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def run_agent(
    user_request: str,
    assets: dict[str, Any],
    memory_path: Path | str = "user_memory.json",
) -> AgentResult:
    """
    Run the full agent pipeline.

    Args:
        user_request:  Natural-language request (any language).
        assets:        Pre-resolved inputs available to the tools, e.g.:
                         image_to_video  → {"image_url": "https://…"}
                         audio_portrait  → {"image_url": "…", "audio_url": "…"}
                         chained         → {"image_url": "…", "audio_url": "…"}
                       Always include "user_description" for prompt enhancement.
        memory_path:   Path for the persistent JSON memory file.

    Returns:
        AgentResult — check `.error` first; `.video_url` populated on success.
    """
    # 1. Memory
    memory = UserMemory(memory_path).load()

    # 2. Plan
    available_assets: set[str] = set(assets.keys())
    plan: Plan = await make_plan(
        user_request=user_request,
        memory=memory,
        available_assets=available_assets,
        deepseek_api_key=DEEPSEEK_API_KEY,
    )
    plan_summary = [
        {"tool": s.tool, "inputs": s.inputs, "reason": s.reason}
        for s in plan
    ]
    logger.info("Executing plan: %s", plan_summary)

    if not plan:
        raise ValueError("Planner returned an empty plan (0 steps). Try rephrasing your request.")

    # 3. Execute
    initial_ctx: dict[str, Any] = {"user_description": user_request, **assets}
    final_ctx = await execute_plan(plan, initial_ctx, DEEPSEEK_API_KEY)

    # 4. Persist memory
    ep = final_ctx.get("enhanced_prompt") or user_request
    memory.record_success(ep)

    video_url = final_ctx.get("video_url")
    if not video_url:
        tools_run = " → ".join(s["tool"] for s in plan_summary)
        raise RuntimeError(
            f"Pipeline completed ({tools_run}) but produced no video URL. "
            f"Context keys: {list(final_ctx.keys())}"
        )

    # 5. Maybe distill memory (background, non-blocking)
    import asyncio as _asyncio
    _asyncio.create_task(_maybe_distill(memory))

    # Parse multi-shot URLs if present
    video_urls: list[str] = []
    if raw_urls := final_ctx.get("video_urls"):
        import json as _json
        try:
            video_urls = _json.loads(raw_urls) if isinstance(raw_urls, str) else raw_urls
        except Exception:
            pass

    return AgentResult(
        video_url=video_url,
        video_urls=video_urls,
        plan=plan_summary,
        enhanced_prompt=final_ctx.get("enhanced_prompt"),
        last_prompt=ep,
        final_context=final_ctx,
    )


async def _maybe_distill(memory: UserMemory) -> None:
    """
    Two distillation paths — both run in background after each generation:

    Path A — Working→L1: when enough high-quality L2 entries exist,
              re-distill the distilled_style summary (L1).

    Path B — Archive→L2: when L3 archive is large, LLM compresses it
              into 3-5 MemoryEntry items and promotes them back to L2.
    """
    client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

    # Path A: re-distill L1 from good L2 entries
    if memory.should_distill_working():
        top = memory.top_working(10)
        if top:
            lines = "\n".join(
                f"- [quality={e.quality:.1f}] {e.content[:150]}" for e in top
            )
            system = (
                "You are a visual style analyst. "
                "Based on the user's top-priority video generation memories, "
                "write a 2-3 sentence summary of their visual style preferences "
                "(lighting, camera moves, animation style, mood). "
                "Be specific and concrete. Return ONLY the summary text."
            )
            try:
                resp = await client.chat.completions.create(
                    model="deepseek-chat",
                    max_tokens=200,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": f"Top memories:\n{lines}"},
                    ],
                )
                memory.update_distilled(resp.choices[0].message.content.strip())
                logger.info("L1 distillation complete")
            except Exception:
                logger.warning("L1 distillation failed", exc_info=True)

    # Path B: compress L3 archive back into L2
    if memory.should_distill_archive():
        archive_text = "\n".join(
            f"{i+1}. [q={e.quality:.1f}] {e.content[:200]}"
            for i, e in enumerate(memory.archive)
        )
        system = (
            "You are a memory compression system. "
            "Given these archived video generation memories, compress them into "
            "3-5 distinct, high-value insight entries. "
            "Each entry should capture a unique user preference or successful pattern. "
            "Return a JSON object: {\"entries\": [{\"content\": str, \"quality\": float}, ...]}"
        )
        try:
            resp = await client.chat.completions.create(
                model="deepseek-chat",
                max_tokens=600,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": f"Archive:\n{archive_text}"},
                ],
            )
            data = json.loads(resp.choices[0].message.content)
            from memory import MemoryEntry
            promoted = [
                MemoryEntry.from_prompt(e["content"], quality=float(e.get("quality", 0.6)))
                for e in data.get("entries", [])
            ]
            if promoted:
                memory.promote_from_archive(promoted)
                logger.info("L3→L2 archive distillation: %d entries promoted", len(promoted))
        except Exception:
            logger.warning("Archive distillation failed", exc_info=True)
