from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "prices.csv"
API_URL = "https://www.alphavantage.co/query"

DEFAULT_TICKERS = ["MSFT", "NVDA", "LLY", "AAPL", "AMD", "META", "AVGO", "GOOGL"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch daily OHLCV data from Alpha Vantage and save it to prices.csv.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="Ticker symbols to fetch.",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help="Where to write the merged CSV file.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=12.5,
        help="Pause between API calls to respect free-tier rate limits.",
    )
    return parser.parse_args()


def get_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing ALPHAVANTAGE_API_KEY. Add it to your environment or a .env file."
        )
    return api_key


def fetch_daily_series(symbol: str, api_key: str) -> pd.DataFrame:
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "compact",
        "datatype": "json",
        "apikey": api_key,
    }
    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if "Error Message" in payload:
        raise RuntimeError(f"Alpha Vantage rejected {symbol}: {payload['Error Message']}")
    if "Note" in payload:
        raise RuntimeError(
            f"Alpha Vantage rate limit reached while fetching {symbol}: {payload['Note']}"
        )

    series = payload.get("Time Series (Daily)")
    if not series:
        raise RuntimeError(f"No daily time series returned for {symbol}.")

    rows = []
    for date_str, values in series.items():
        rows.append(
            {
                "ticker": symbol,
                "date": date_str,
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": float(values["4. close"]),
                "volume": int(values["5. volume"]),
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values("date").reset_index(drop=True)


def main():
    args = parse_args()
    api_key = get_api_key()

    all_frames = []
    for index, ticker in enumerate(args.tickers):
        print(f"[{index + 1}/{len(args.tickers)}] Fetching {ticker}...")
        frame = fetch_daily_series(ticker.upper(), api_key)
        all_frames.append(frame)
        if index < len(args.tickers) - 1:
            time.sleep(args.pause_seconds)

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    output_path = Path(args.output)
    combined.to_csv(output_path, index=False)

    print(f"Saved {len(combined)} rows to {output_path}")
    print("Next steps:")
    print("  1. py mirrorquant_demo\\build_training_data.py")
    print("  2. py mirrorquant_demo\\train_vqvae.py")
    print("  3. py -m mirrorquant_demo.encode_windows")


if __name__ == "__main__":
    main()
