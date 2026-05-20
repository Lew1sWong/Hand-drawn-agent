from __future__ import annotations

import logging
import uuid

import edge_tts

from tools.base import BaseTool, ToolContract
from tools._replicate import PUBLIC_BASE_URL

logger = logging.getLogger(__name__)


class TTSTool(BaseTool):
    name = "tts"
    description = (
        "Converts text to natural speech and returns an audio_url. "
        "Use this BEFORE audio_portrait when the user has no audio file. "
        "Requires context key: tts_text (str) — the words to speak; "
        "falls back to user_description if tts_text not set. "
        "Optional override: voice (str) — edge-tts voice name, "
        "default 'zh-CN-XiaoxiaoNeural' (female Chinese). "
        "Other good voices: 'zh-CN-YunxiNeural' (male), 'en-US-JennyNeural' (English). "
        "Produces: audio_url."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "tts_text": {
                "type": "string",
                "description": "Exact text for the character to speak. If omitted, uses user_description.",
            },
            "voice": {
                "type": "string",
                "default": "zh-CN-XiaoxiaoNeural",
                "description": "edge-tts voice name.",
            },
        },
        "required": [],
    }
    contract = ToolContract(
        reads          = [],
        writes         = ["audio_url"],
        optional_reads = ["tts_text", "user_description", "voice"],
    )

    async def run(self, ctx: dict) -> dict:
        text  = ctx.get("tts_text") or ctx.get("user_description", "")
        voice = ctx.get("voice", "zh-CN-XiaoxiaoNeural")

        fname    = f"tts_{uuid.uuid4().hex}.mp3"
        tmp_path = f"/tmp/{fname}"

        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(tmp_path)

        audio_url = f"{PUBLIC_BASE_URL}/media/{fname}"
        logger.info("TTS saved  voice=%s  path=%s  url=%s", voice, tmp_path, audio_url)
        return {"audio_url": audio_url}
