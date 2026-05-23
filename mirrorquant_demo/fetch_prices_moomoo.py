from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

try:
    from moomoo import AuType, KLType, OpenQuoteContext, RET_OK
except ImportError as exc:
    raise SystemExit(
        "Missing moomoo-api. Install it with `pip install moomoo-api` before "
        "running this script."
    ) from exc

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "prices.csv"

DEFAULT_TICKERS = ["MSFT", "NVDA", "LLY", "AAPL", "AMD", "META", "AVGO", "GOOGL"]
ADJUSTMENT_MAP = {
    "qfq": AuType.QFQ,
    "hfq": AuType.HFQ,
    "none": AuType.NONE,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch daily OHLCV bars from moomoo OpenAPI and save them to prices.csv.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="Ticker symbols to fetch. Symbols without a market prefix use --market-prefix.",
    )
    parser.add_argument(
        "--start",
        default="2023-01-01",
        help="Start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end",
        default=datetime.now(UTC).strftime("%Y-%m-%d"),
        help="End date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--market-prefix",
        default=os.getenv("MOOMOO_MARKET_PREFIX", "US"),
        help="Market prefix used for symbols without one, for example US or SG.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("MOOMOO_HOST", "127.0.0.1"),
        help="OpenD host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MOOMOO_PORT", "11111")),
        help="OpenD port.",
    )
    parser.add_argument(
        "--adjust",
        choices=sorted(ADJUSTMENT_MAP.keys()),
        default="qfq",
        help="Adjustment mode for historical bars.",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help="Where to write the merged CSV file.",
    )
    return parser.parse_args()


def normalize_code(symbol: str, market_prefix: str) -> str:
    symbol = symbol.strip().upper()
    if "." in symbol:
        return symbol
    return f"{market_prefix.upper()}.{symbol}"


def normalize_frame(frame: pd.DataFrame, requested_code: str) -> pd.DataFrame:
    if frame.empty:
        raise RuntimeError(f"No rows returned for {requested_code}.")

    ticker = requested_code.split(".", 1)[1] if "." in requested_code else requested_code
    normalized = pd.DataFrame(
        {
            "ticker": ticker,
            "date": pd.to_datetime(frame["time_key"]).dt.strftime("%Y-%m-%d"),
            "open": frame["open"].astype(float),
            "high": frame["high"].astype(float),
            "low": frame["low"].astype(float),
            "close": frame["close"].astype(float),
            "volume": pd.to_numeric(frame["volume"]).fillna(0).astype(int),
        }
    )
    return normalized.sort_values("date").reset_index(drop=True)


def fetch_daily_bars(
    quote_ctx: OpenQuoteContext,
    symbol: str,
    start: str,
    end: str,
    adjustment: AuType,
) -> pd.DataFrame:
    page_req_key = None
    pages: list[pd.DataFrame] = []

    while True:
        ret, frame, page_req_key = quote_ctx.request_history_kline(
            code=symbol,
            start=start,
            end=end,
            ktype=KLType.K_DAY,
            autype=adjustment,
            max_count=1000,
            page_req_key=page_req_key,
        )
        if ret != RET_OK:
            raise RuntimeError(f"moomoo request failed for {symbol}: {frame}")

        pages.append(frame)
        if page_req_key is None:
            break

    merged = pd.concat(pages, ignore_index=True)
    return normalize_frame(merged, symbol)


def main():
    load_dotenv()
    args = parse_args()

    quote_ctx = OpenQuoteContext(host=args.host, port=args.port)
    try:
        frames = []
        adjustment = ADJUSTMENT_MAP[args.adjust]

        for index, ticker in enumerate(args.tickers):
            code = normalize_code(ticker, args.market_prefix)
            print(
                f"[{index + 1}/{len(args.tickers)}] Fetching {code} from moomoo "
                f"via OpenD at {args.host}:{args.port}..."
            )
            frames.append(
                fetch_daily_bars(
                    quote_ctx=quote_ctx,
                    symbol=code,
                    start=args.start,
                    end=args.end,
                    adjustment=adjustment,
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
    finally:
        quote_ctx.close()


if __name__ == "__main__":
    main()
