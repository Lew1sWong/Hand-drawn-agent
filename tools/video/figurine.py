from __future__ import annotations

import asyncio
import logging
import os

from tools.base import BaseTool, ToolContract
from tools._replicate import _replicate_to_url

logger = logging.getLogger(__name__)


class FigurineToAnimeCharTool(BaseTool):
    """
    Two-step Replicate pipeline:
      1. depth-anything-v2-large — extract a depth map to preserve 3-D structure
      2. img2img SDXL ControlNet (depth) — render in 2-D anime style

    Reads  ctx["image_url"]          (the figurine photo)
    Writes ctx["anime_image_url"]    (the anime render)
           ctx["image_url"]          (overwritten so downstream tools use the anime image)
    """

    name        = "figurine_to_anime"
    description = """\
Convert a Q-version figurine / figure (手办) photo into a 2D anime-style character image.

Pipeline (Replicate):
  Step 1 — depth-anything/depth-anything-v2-large: extract depth map (preserves 3-D chibi volume)
  Step 2 — lucataco/sdxl-controlnet + depth: render in 2-D anime style
             (preserves outfit colours, hair, accessories)

Writes anime_image_url and overwrites image_url so the next tool (image_to_video)
automatically picks up the anime render.

USE THIS TOOL when the user uploads a figurine / figure / 手办 photo and asks
for an animated video — run figurine_to_anime FIRST, then image_to_video."""

    input_schema = {
        "type": "object",
        "properties": {
            "prompt_suffix": {
                "type":        "string",
                "description": "Extra style descriptors appended to the anime prompt, e.g. 'pink twin-tails, sailor uniform'",
            },
            "strength": {
                "type":        "number",
                "description": "img2img denoising strength 0.0–1.0 (default 0.75)",
            },
        },
    }
    contract = ToolContract(
        reads          = ["image_url"],
        writes         = ["anime_image_url", "image_url"],
        optional_reads = ["prompt_suffix", "strength"],
    )

    _DEPTH_MODEL = "depth-anything/depth-anything-v2-large"
    _ANIME_MODEL = "lucataco/sdxl-controlnet:06775cd262843edbde5abab958abdbb65a0a6b58dcd869086358b1f55a0b2c70"

    async def run(self, ctx: dict) -> dict:
        image_url = ctx["image_url"]

        api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not api_token:
            raise RuntimeError("[FigurineToAnime] REPLICATE_API_TOKEN env var is not set")

        os.environ["REPLICATE_API_TOKEN"] = api_token
        import replicate

        loop = asyncio.get_running_loop()

        _REPLICATE_TIMEOUT = 180  # 3 min per Replicate call

        logger.info("[FigurineToAnime] extracting depth map  image_url=%s", image_url)
        depth_out = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: replicate.run(self._DEPTH_MODEL, input={"image": image_url}),
            ),
            timeout=_REPLICATE_TIMEOUT,
        )
        depth_url = _replicate_to_url(depth_out)
        logger.info("[FigurineToAnime] depth map ready  depth_url=%s", depth_url)

        base_prompt = (
            "masterpiece, best quality, 2D anime illustration, "
            "chibi Q-version character, vibrant colours, clean line art, "
            "detailed outfit and hair"
        )
        suffix = ctx.get("prompt_suffix", "")
        if suffix:
            base_prompt = f"{base_prompt}, {suffix}"

        logger.info("[FigurineToAnime] converting to anime style  prompt='%s'", base_prompt)
        anime_out = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: replicate.run(
                    self._ANIME_MODEL,
                    input={
                        "prompt":          base_prompt,
                        "negative_prompt": "3D render, photorealistic, blurry, watermark, text",
                        "image":           depth_url,
                        "num_inference_steps": 30,
                        "guidance_scale":  7.5,
                        "controlnet_conditioning_scale": 0.8,
                        "scheduler":       "K_EULER_ANCESTRAL",
                    },
                ),
            ),
            timeout=_REPLICATE_TIMEOUT,
        )
        anime_url = _replicate_to_url(anime_out)
        logger.info("[FigurineToAnime] anime image ready  anime_url=%s", anime_url)

        return {
            "anime_image_url": anime_url,
            "image_url":       anime_url,
        }
