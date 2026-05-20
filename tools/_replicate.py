"""Shared Replicate helpers + PUBLIC_BASE_URL config."""
from __future__ import annotations

import os

PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")


def _replicate_to_url(output) -> str:
    """Normalize any Replicate output (FileOutput / list / str / generator) to a URL string."""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        if not output:
            raise RuntimeError("Replicate returned an empty output list")
        return _replicate_to_url(output[0])
    if hasattr(output, "url"):
        return str(output.url)
    try:
        items = list(output)
        if items:
            return _replicate_to_url(items[0])
    except Exception:
        pass
    return str(output)
