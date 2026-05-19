"""
Quick smoke-test for FigurineToAnimeCharTool.

Usage:
    python test_figurine.py <path-to-figurine-image>

The image is served via the local /media/ endpoint, so the server must be
running (uvicorn api:app --reload --port 8000) OR you can pass a public URL
directly with --url:
    python test_figurine.py --url https://example.com/figurine.jpg
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


async def main(image_source: str, is_url: bool) -> None:
    from tools import FigurineToAnimeCharTool

    if is_url:
        image_url = image_source
        logger.info("Using remote URL: %s", image_url)
    else:
        # Copy to /tmp and build a local URL so Replicate can fetch it
        src = Path(image_source)
        if not src.exists():
            raise FileNotFoundError(f"Image not found: {src}")
        fname = f"figurine_test_{uuid.uuid4().hex}{src.suffix}"
        dest  = Path("/tmp") / fname
        shutil.copy2(src, dest)
        base  = os.environ.get("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")
        image_url = f"{base}/media/{fname}"
        logger.info("Copied image to %s  →  public URL: %s", dest, image_url)

    tool = FigurineToAnimeCharTool()
    ctx  = {"image_url": image_url, "user_description": "Q-version figurine test"}

    logger.info("Running FigurineToAnimeCharTool …")
    result = await tool.run(ctx)

    print("\n=== Result ===")
    print(f"  anime_image_url : {result['anime_image_url']}")
    print(f"  image_url (set) : {result['image_url']}")
    print("=== Done ===\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FigurineToAnimeCharTool")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("image_path", nargs="?",  help="Local path to a figurine image")
    group.add_argument("--url",                  help="Public URL of the figurine image")
    args = parser.parse_args()

    if args.url:
        asyncio.run(main(args.url, is_url=True))
    else:
        asyncio.run(main(args.image_path, is_url=False))
