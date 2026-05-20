from __future__ import annotations

import asyncio
import json
import logging
import os

from tools.base import BaseTool, ToolContract
from tools._volcengine import _submit_and_poll, _extract_video_url

logger = logging.getLogger(__name__)


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
    contract = ToolContract(
        reads          = ["user_description"],
        writes         = ["video_url", "video_urls", "scene_prompts"],
        optional_reads = ["n_shots", "duration", "width", "height", "image_url"],
    )

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
            body["image_urls"] = [ctx["image_url"]]
        result = await _submit_and_poll(body, f"MultiShot[{idx+1}]")
        return _extract_video_url(result["data"])

    async def run(self, ctx: dict) -> dict:
        n = min(max(int(ctx.get("n_shots", 3)), 2), 4)
        script = ctx["user_description"]
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
            "video_url":     video_urls[0],
            "video_urls":    json.dumps(video_urls, ensure_ascii=False),
            "scene_prompts": json.dumps(scenes, ensure_ascii=False),
        }
