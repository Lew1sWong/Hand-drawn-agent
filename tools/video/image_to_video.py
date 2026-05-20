from __future__ import annotations

from tools.base import BaseTool, ToolContract
from tools._volcengine import _submit_and_poll, _extract_video_url


class ImageToVideoTool(BaseTool):
    name = "image_to_video"
    description = (
        "Animates a hand-drawn sketch into a short video using JiMeng 3.0 Pro. "
        "Requires context keys: image_url (str), enhanced_prompt (str). "
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
        reads          = ["image_url", "enhanced_prompt"],
        writes         = ["video_url", "task_id"],
        optional_reads = ["duration", "width", "height"],
    )

    _REQ_KEY = "jimeng_ti2v_v30_pro"

    async def run(self, ctx: dict) -> dict:
        image_url = ctx.get("image_url")
        if not image_url:
            raise ValueError(
                "image_to_video requires an image. "
                "Send a sketch image first, or use text_to_video for text-only requests."
            )
        body = {
            "req_key":    self._REQ_KEY,
            "prompt":     ctx["enhanced_prompt"],
            "image_urls": [image_url],
            "duration":   ctx.get("duration", 4),
            "width":      ctx.get("width", 1280),
            "height":     ctx.get("height", 720),
        }
        result = await _submit_and_poll(body, "ImageToVideo")
        return {
            "video_url": _extract_video_url(result["data"]),
            "task_id":   result["task_id"],
        }
