from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "prices.csv"

DEFAULT_TICKERS = ["MSFT", "NVDA", "LLY", "AAPL", "AMD", "META", "AVGO", "GOOGL"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch daily OHLCV stock bars from Alpaca and save them to prices.csv.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="Ticker symbols to fetch.",
    )
    parser.add_argument(
        "--start",
        default="2023-01-01",
        help="Start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end",
        default=datetime.utcnow().strftime("%Y-%m-%d"),
        help="End date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--feed",
        choices=["iex", "sip"],
        default="iex",
        help="Market data feed to request. IEX is the safest default for basic setups.",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help="Where to write the merged CSV file.",
    )
    return parser.parse_args()


def get_credentials() -> tuple[str, str]:
    load_dotenv()
    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError(
            "Missing Alpaca credentials. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY "
            "in your environment or a .env file."
        )
    return api_key, secret_key


def fetch_daily_bars(
    client: StockHistoricalDataClient,
    symbol: str,
    start: str,
    end: str,
    feed: str,
) -> pd.DataFrame:
    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        start=datetime.fromisoformat(start),
        end=datetime.fromisoformat(end),
        feed=feed,
    )
    bars = client.get_stock_bars(request)
    frame = bars.df.reset_index()

    if frame.empty:
        raise RuntimeError(f"No bars returned for {symbol} between {start} and {end}.")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    normalized = pd.DataFrame(
        {
            "ticker": frame["symbol"].astype(str),
            "date": frame["timestamp"].dt.strftime("%Y-%m-%d"),
            "open": frame["open"].astype(float),
            "high": frame["high"].astype(float),
            "low": frame["low"].astype(float),
            "close": frame["close"].astype(float),
            "volume": frame["volume"].astype(int),
        }
    )
    return normalized.sort_values("date").reset_index(drop=True)


def main():
    args = parse_args()
    api_key, secret_key = get_credentials()

    client = StockHistoricalDataClient(api_key, secret_key)

    frames = []
    for index, ticker in enumerate(args.tickers):
        print(f"[{index + 1}/{len(args.tickers)}] Fetching {ticker} from Alpaca...")
        frames.append(
            fetch_daily_bars(
                client=client,
                symbol=ticker.upper(),
                start=args.start,
                end=args.end,
                feed=args.feed,
            )
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    output_path = Path(args.output)
    combined.to_csv(output_path, index=False)

    print(f"Saved {len(combined)} rows to {output_path}")
    print("Next steps:")
    print("  1. py mirrorquant_demo\\build_training_data.py")
    print("  2. py mirrorquant_demo\\train_vqvae.py")
    print("  3. py -m mirrorquant_demo.encode_windows")
    print("  4. py -m uvicorn mirrorquant_demo.app:app --reload")


if __name__ == "__main__":
    main()
