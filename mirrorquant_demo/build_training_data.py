from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PRICES_PATH = DATA_DIR / "prices.csv"
WINDOWS_OUT = DATA_DIR / "training_windows.npz"
META_OUT = DATA_DIR / "training_windows_meta.csv"

WINDOW_SIZE = 40
MIN_HISTORY = 60


def load_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["ticker", "date"]).copy()


def make_sequence_features(window_df: pd.DataFrame) -> np.ndarray:
    window_df = window_df.sort_values("date").copy()

    window_df["daily_return"] = window_df["close"].pct_change().fillna(0.0)
    window_df["volume_change"] = (
        window_df["volume"]
        .pct_change()
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )
    window_df["price_rel"] = (window_df["close"] / window_df["close"].iloc[0]) - 1.0

    features = window_df[["daily_return", "volume_change", "price_rel"]].to_numpy(dtype=np.float32)
    return features


def build_windows(df: pd.DataFrame, window_size: int = WINDOW_SIZE):
    windows = []
    meta_rows = []

    for ticker in sorted(df["ticker"].unique()):
        stock_df = df[df["ticker"] == ticker].sort_values("date").copy()

        if len(stock_df) < MIN_HISTORY:
            continue

        for start_idx in range(0, len(stock_df) - window_size + 1):
            end_idx = start_idx + window_size
            window_df = stock_df.iloc[start_idx:end_idx].copy()

            X_window = make_sequence_features(window_df)

            if len(X_window) != window_size:
                continue

            windows.append(X_window)
            meta_rows.append(
                {
                    "ticker": ticker,
                    "start_date": window_df["date"].iloc[0],
                    "end_date": window_df["date"].iloc[-1],
                }
            )

    X = np.stack(windows)
    meta_df = pd.DataFrame(meta_rows)
    return X, meta_df


def main():
    df = load_prices(PRICES_PATH)
    X, meta_df = build_windows(df)

    np.savez_compressed(WINDOWS_OUT, X=X)
    meta_df.to_csv(META_OUT, index=False)

    print("Built training data")
    print("X shape:", X.shape)
    print("Saved:", WINDOWS_OUT)
    print("Saved:", META_OUT)


if __name__ == "__main__":
    main()