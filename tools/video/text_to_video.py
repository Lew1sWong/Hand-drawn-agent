from __future__ import annotations

from tools.base import BaseTool, ToolContract
from tools._volcengine import _submit_and_poll, _extract_video_url


class TextToVideoTool(BaseTool):
    name = "text_to_video"
    description = (
        "Generates a video directly from a text description — no image required. "
        "Use this when the user has NO sketch/image and just wants to describe a scene. "
        "Requires context key: enhanced_prompt (str). "
        "Optional overrides: duration (4 or 8 s), width (int), height (int). "
        "Produces: video_url."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "duration": {"type": "integer", "enum": [4, 8], "default": 4},
            "width":    {"type": "integer", "default": 1280},
            "height":   {"type": "integer", "default": 720},
        },
        "required": [],
    }
    contract = ToolContract(
        reads          = ["enhanced_prompt"],
        writes         = ["video_url", "task_id"],
        optional_reads = ["duration", "width", "height"],
    )

    _REQ_KEY = "jimeng_ti2v_v30_pro"

    async def run(self, ctx: dict) -> dict:
        body = {
            "req_key":  self._REQ_KEY,
            "prompt":   ctx["enhanced_prompt"],
            "duration": ctx.get("duration", 4),
            "width":    ctx.get("width", 1280),
            "height":   ctx.get("height", 720),
        }
        result = await _submit_and_poll(body, "TextToVideo")
        return {
            "video_url": _extract_video_url(result["data"]),
            "task_id":   result["task_id"],
        }
