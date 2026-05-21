# Backend: server
from __future__ import annotations

from mirrorquant_demo.vqvae_search import find_vqvae_mirrors

import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"

Mode = Literal["price_dna", "economic_dna", "social_dna"] # To set the variable to be one of these 3 menu options only


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


@app.get("/health") # Simple health check endpoint to verify the server is running
async def health():
    return {"status": "ok", "app": "mirrorquant-demo"}

@app.get("/api/heroes") # Endpoint to list available hero stocks and their associated metadata (ticker, name, hero window dates)
async def list_heroes():
    return {"heroes": _load_json("heroes.json")}

@app.get("/api/mirrors")
async def get_mirrors(
    ticker: str,
    mode: Mode = "price_dna",
):
    normalized = ticker.upper()

    heroes = _load_json("heroes.json")
    hero = next((item for item in heroes if item["ticker"] == normalized), None)
    if hero is None:
        raise HTTPException(status_code=404, detail=f"No hero data for {normalized}")

    if mode != "price_dna":
        matches = _load_json("mirror_matches.json")
        if normalized not in matches:
            raise HTTPException(status_code=404, detail=f"No mirror data for {normalized}")
        return {
            "ticker": normalized,
            "mode": mode,
            "hero": hero,
            "hero_regime_code": hero[mode]["regime_code"],
            "matches": matches[normalized][mode],
            "search_backend": "mock",
        }

    try:
        results, hero_code, effective_window = find_vqvae_mirrors(
            hero_ticker=normalized,
            start=hero["start_date"],
            end=hero["end_date"],
            top_k=5,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    matches = []
    for row in results.head(5).itertuples(index=False):
        match_start = row.start_date.strftime("%Y-%m-%d")
        match_end = row.end_date.strftime("%Y-%m-%d")
        matches.append(
            {
                "ticker": row.ticker,
                "name": row.ticker,
                "score": float(row.similarity),
                "regime_label": f"VQ-VAE Price DNA match ({match_start} to {match_end})",
                "sector": "Unknown",
                "explanation": (
                    f"{row.ticker} matched a learned latent regime similar to "
                    f"{normalized}'s encoded hero window. "
                    f"Matched window: {match_start} to {match_end}."
                ),
            }
        )

    return {
        "ticker": normalized,
        "mode": mode,
        "hero": hero,
        "hero_regime_code": hero_code,
        "matches": matches,
        "effective_hero_window": {
            "start_date": effective_window["start_date"].strftime("%Y-%m-%d"),
            "end_date": effective_window["end_date"].strftime("%Y-%m-%d"),
            "window_size": effective_window["window_size"],
        },
        "search_backend": "vqvae",
    }


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

