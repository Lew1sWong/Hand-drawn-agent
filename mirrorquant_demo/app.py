from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"

Mode = Literal["price_dna", "economic_dna", "social_dna"]


def _load_json(filename: str):
    with (DATA_DIR / filename).open("r", encoding="utf-8") as handle:
        return json.load(handle)


app = FastAPI(
    title="MirrorQuant API",
    version="0.1.0",
    description="API for a polished MirrorQuant concept demo.",
)


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/static/{asset_path:path}", include_in_schema=False)
async def static_files(asset_path: str):
    asset = STATIC_DIR / asset_path
    if not asset.exists() or not asset.is_file():
        raise HTTPException(status_code=404, detail="Static asset not found")
    return FileResponse(asset)


@app.get("/health")
async def health():
    return {"status": "ok", "app": "mirrorquant-demo"}


@app.get("/api/heroes")
async def list_heroes():
    return {"heroes": _load_json("heroes.json")}


@app.get("/api/market-watch")
async def get_market_watch():
    return _load_json("market_watch.json")


@app.get("/api/industry-chain/{ticker}")
async def get_industry_chain(ticker: str):
    chain_data = _load_json("industry_chain.json")
    normalized = ticker.upper()
    if normalized not in chain_data:
        raise HTTPException(status_code=404, detail=f"No industry chain data for {normalized}")
    return {"ticker": normalized, "relationships": chain_data[normalized]}


@app.get("/api/mirrors")
async def get_mirrors(
    ticker: str,
    mode: Mode = "price_dna",
):
    matches = _load_json("mirror_matches.json")
    normalized = ticker.upper()
    if normalized not in matches:
        raise HTTPException(status_code=404, detail=f"No mirror data for {normalized}")
    return {
        "ticker": normalized,
        "mode": mode,
        "hero": next(
            hero
            for hero in _load_json("heroes.json")
            if hero["ticker"] == normalized
        ),
        "matches": matches[normalized][mode],
    }
