# Backend: server
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from mirrorquant_demo.vqvae_search import find_vqvae_mirrors

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
PRICES_PATH = DATA_DIR / "prices.csv"
MARKET_WATCH_PRICES_PATH = DATA_DIR / "market_watch_prices.csv"

Mode = Literal["price_dna", "economic_dna", "social_dna"]
MARKET_WATCH_SYMBOLS = {
    "SPY": "US Equities",
    "QQQ": "Growth Leadership",
    "IWM": "Small-Cap Breadth",
    "TLT": "Duration / Rates",
    "HYG": "Credit Appetite",
}


def _load_json(filename: str):
    with (DATA_DIR / filename).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_prices(path: Path = PRICES_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["ticker", "date"]).copy()


def _get_price_window(
    df: pd.DataFrame,
    ticker: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    return df[
        (df["ticker"] == ticker.upper())
        & (df["date"] >= pd.to_datetime(start_date))
        & (df["date"] <= pd.to_datetime(end_date))
    ].sort_values("date").copy()


def _serialize_close_series(window_df: pd.DataFrame) -> list[dict[str, str | float]]:
    return [
        {
            "date": row.date.strftime("%Y-%m-%d"),
            "close": float(row.close),
        }
        for row in window_df.itertuples(index=False)
    ]


def _trend_status(change_pct: float) -> str:
    if change_pct >= 0.05:
        return "Uptrend"
    if change_pct <= -0.05:
        return "Drawdown"
    return "Range-bound"


def _market_watch_headline(changes: dict[str, float]) -> str:
    positive_count = sum(change > 0 for change in changes.values())
    if (
        changes.get("SPY", 0) > 0
        and changes.get("QQQ", 0) > 0
        and changes.get("HYG", 0) > 0
    ):
        return "Risk-on with broad support"
    if positive_count <= 1 and changes.get("TLT", 0) < 0:
        return "Risk-off with rate pressure"
    return "Mixed tape with selective leadership"


def _build_live_market_watch() -> dict | None:
    if not MARKET_WATCH_PRICES_PATH.exists():
        return None

    df = _load_prices(MARKET_WATCH_PRICES_PATH)
    indicators = []
    changes: dict[str, float] = {}

    for ticker, label in MARKET_WATCH_SYMBOLS.items():
        window_df = df[df["ticker"] == ticker].sort_values("date").tail(40).copy()
        if len(window_df) < 2:
            continue

        first_close = float(window_df["close"].iloc[0])
        last_close = float(window_df["close"].iloc[-1])
        change_pct = (last_close / first_close) - 1.0
        changes[ticker] = change_pct

        indicators.append(
            {
                "symbol": ticker,
                "name": label,
                "value": f"{last_close:.2f}",
                "status": _trend_status(change_pct),
                "insight": (
                    f"{ticker} moved {change_pct * 100:+.1f}% over the latest "
                    f"{len(window_df)} sessions, giving a live read on {label.lower()}."
                ),
                "series": _serialize_close_series(window_df),
                "change_pct": round(change_pct * 100, 2),
                "as_of": window_df["date"].iloc[-1].strftime("%Y-%m-%d"),
            }
        )

    if not indicators:
        return None

    as_of = max(indicator["as_of"] for indicator in indicators)
    return {
        "as_of": as_of,
        "headline_regime": _market_watch_headline(changes),
        "indicators": indicators,
    }


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


@app.get("/api/price-series")
async def get_price_series(ticker: str, start_date: str, end_date: str):
    df = _load_prices()
    ticker_df = df[df["ticker"] == ticker.upper()].sort_values("date").copy()
    if ticker_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No price series found for {ticker.upper()}",
        )

    window_df = _get_price_window(df, ticker, start_date, end_date)
    if window_df.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No price series found for {ticker.upper()} from "
                f"{start_date} to {end_date}"
            ),
        )

    return {
        "ticker": ticker.upper(),
        "available_start_date": ticker_df["date"].iloc[0].strftime("%Y-%m-%d"),
        "available_end_date": ticker_df["date"].iloc[-1].strftime("%Y-%m-%d"),
        "start_date": start_date,
        "end_date": end_date,
        "series": _serialize_close_series(ticker_df),
        "window_series": _serialize_close_series(window_df),
    }


@app.get("/api/mirrors")
async def get_mirrors(
    ticker: str,
    mode: Mode = "price_dna",
    start_date: str | None = None,
    end_date: str | None = None,
):
    normalized = ticker.upper()

    heroes = _load_json("heroes.json")
    hero = next((item for item in heroes if item["ticker"] == normalized), None)
    if hero is None:
        raise HTTPException(status_code=404, detail=f"No hero data for {normalized}")

    selected_start = start_date or hero["start_date"]
    selected_end = end_date or hero["end_date"]

    if selected_start > selected_end:
        raise HTTPException(
            status_code=400,
            detail="start_date must be on or before end_date",
        )

    if mode != "price_dna":
        matches = _load_json("mirror_matches.json")
        if normalized not in matches:
            raise HTTPException(status_code=404, detail=f"No mirror data for {normalized}")
        return {
            "ticker": normalized,
            "mode": mode,
            "hero": hero,
            "hero_regime_code": hero[mode]["regime_code"],
            "selected_window": {
                "start_date": selected_start,
                "end_date": selected_end,
            },
            "matches": matches[normalized][mode],
            "search_backend": "mock",
        }

    try:
        results, hero_code, selected_window, effective_window = find_vqvae_mirrors(
            hero_ticker=normalized,
            start=selected_start,
            end=selected_end,
            top_k=5,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    df = _load_prices()
    matches = []
    for row in results.head(5).itertuples(index=False):
        match_start = row.start_date.strftime("%Y-%m-%d")
        match_end = row.end_date.strftime("%Y-%m-%d")
        match_window = _get_price_window(df, row.ticker, row.start_date, row.end_date)
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
                "matched_window": {
                    "start_date": match_start,
                    "end_date": match_end,
                },
                "series": _serialize_close_series(match_window),
            }
        )

    return {
        "ticker": normalized,
        "mode": mode,
        "hero": hero,
        "hero_regime_code": hero_code,
        "selected_window": {
            "start_date": selected_window["start_date"].strftime("%Y-%m-%d"),
            "end_date": selected_window["end_date"].strftime("%Y-%m-%d"),
            "row_count": selected_window["row_count"],
        },
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
    live_market_watch = _build_live_market_watch()
    if live_market_watch is not None:
        return live_market_watch
    return _load_json("market_watch.json")


@app.get("/api/industry-chain/{ticker}")
async def get_industry_chain(ticker: str):
    chain_data = _load_json("industry_chain.json")
    normalized = ticker.upper()
    if normalized not in chain_data:
        raise HTTPException(
            status_code=404,
            detail=f"No industry chain data for {normalized}",
        )
    return {"ticker": normalized, "relationships": chain_data[normalized]}
