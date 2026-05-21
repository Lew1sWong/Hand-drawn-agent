from __future__ import annotations

from mirrorquant_demo.mirror_search import load_prices, find_mirrors 

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


@app.get("/health")
async def health():
    return {"status": "ok", "app": "mirrorquant-demo"}


@app.get("/api/mirrors")
async def get_mirrors(
    ticker: str,
    mode: Mode = "price_dna",
):
    normalized = ticker.upper()

    if mode != "price_dna":
        raise HTTPException(
            status_code=400,
            detail="Only price_dna is supported in the first real MVP"
        )

    heroes = _load_json("heroes.json")
    hero = next((item for item in heroes if item["ticker"] == normalized), None)
    if hero is None:
        raise HTTPException(status_code=404, detail=f"No hero data for {normalized}")

    df = load_prices(str(DATA_DIR / "prices.csv"))

    try:
        results = find_mirrors(
            df=df,
            hero_ticker=normalized,
            start=hero["start_date"],
            end=hero["end_date"],
            window_size=40,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    matches = []
    for row in results.head(5).itertuples(index=False):
        matches.append(
            {
                "ticker": row.ticker,
                "name": row.ticker,
                "score": float(row.similarity),
                "regime_label": "Price DNA match",
                "sector": "Unknown",
                "explanation": f"{row.ticker} shows similar price and volume behavior to {normalized} during the selected hero window.",
            }
        )

    return {
        "ticker": normalized,
        "mode": mode,
        "hero": hero,
        "matches": matches,
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

