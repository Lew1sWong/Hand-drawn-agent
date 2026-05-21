#Test script for mirror search. Run this to find stocks with similar recent behavior to a "hero" stock.
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


def max_drawdown(close_series: pd.Series) -> float:
    running_max = close_series.cummax()
    drawdown = (close_series - running_max) / running_max
    return drawdown.min()


def compute_window_features(df: pd.DataFrame) -> dict:
    df = df.sort_values("date").copy()

    if len(df) < 2:
        return None

    df["daily_return"] = df["close"].pct_change()

    total_return = (df["close"].iloc[-1] / df["close"].iloc[0]) - 1
    mean_daily_return = df["daily_return"].dropna().mean()
    volatility = df["daily_return"].dropna().std()
    if pd.isna(volatility):
        volatility = 0.0
    mdd = max_drawdown(df["close"])
    avg_volume = df["volume"].mean()

    halfway = len(df) // 2
    if halfway == 0:
        volume_trend = 0
    else:
        first_half_vol = df["volume"].iloc[:halfway].mean()
        second_half_vol = df["volume"].iloc[halfway:].mean()

        if pd.isna(first_half_vol) or first_half_vol == 0 or pd.isna(second_half_vol):
            volume_trend = 0
        else:
            volume_trend = (second_half_vol / first_half_vol) - 1

    features = {
        "total_return": total_return,
        "mean_daily_return": mean_daily_return,
        "volatility": volatility,
        "max_drawdown": mdd,
        "avg_volume": avg_volume,
        "volume_trend": volume_trend,
    }

    if any(pd.isna(v) for v in features.values()):
        return None

    return features


def get_window(df: pd.DataFrame, ticker: str, start: str, end: str) -> pd.DataFrame:
    return df[
        (df["ticker"] == ticker) &
        (df["date"] >= pd.to_datetime(start)) &
        (df["date"] <= pd.to_datetime(end))
    ].copy()


def get_latest_n_days(df: pd.DataFrame, ticker: str, n: int = 40) -> pd.DataFrame:
    stock_df = df[df["ticker"] == ticker].sort_values("date").copy()
    return stock_df.tail(n)


FEATURE_COLUMNS = [
    "total_return",
    "mean_daily_return",
    "volatility",
    "max_drawdown",
    "avg_volume",
    "volume_trend",
]


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "prices.csv"

df = pd.read_csv(CSV_PATH, parse_dates=["date"])
df = df.sort_values(["ticker", "date"])

hero_ticker = "MSFT"
hero_start = "2023-01-03"
hero_end = "2023-04-03"

hero_window = get_window(df, hero_ticker, hero_start, hero_end)
hero_features = compute_window_features(hero_window)
if hero_features is None:
    raise ValueError(
        f"Hero window for {hero_ticker} from {hero_start} to {hero_end} "
        f"does not have enough clean data. Found {len(hero_window)} rows."
    )

rows = [{"ticker": hero_ticker, **hero_features}]

for ticker in sorted(df["ticker"].unique()):
    if ticker == hero_ticker:
        continue

    candidate_window = get_latest_n_days(df, ticker, n=40)
    if len(candidate_window) < 10:
        continue

    candidate_features = compute_window_features(candidate_window)
    if candidate_features is None:
        continue
    rows.append({"ticker": ticker, **candidate_features})

if len(rows) == 1:
    raise ValueError(
        "No valid candidate stocks were found. "
        "Your CSV needs more ticker history, ideally at least 10 rows per stock "
        "and at least 2-3 different stocks."
    )

feature_df = pd.DataFrame(rows)

scaler = StandardScaler()
X = scaler.fit_transform(feature_df[FEATURE_COLUMNS])

hero_vector = X[0].reshape(1, -1)
candidate_vectors = X[1:]

similarities = cosine_similarity(hero_vector, candidate_vectors)[0]

results = feature_df.iloc[1:][["ticker"]].copy()
results["similarity"] = similarities
results = results.sort_values("similarity", ascending=False)

print(f"Hero: {hero_ticker} ({hero_start} to {hero_end})")
print()
print("Top 5 matches:")
print(results.head(5).to_string(index=False))
