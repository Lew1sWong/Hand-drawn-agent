import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from features import compute_window_features

FEATURE_COLUMNS = [
    "total_return",
    "mean_daily_return",
    "volatility",
    "max_drawdown",
    "avg_volume",
    "volume_trend",
]

def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["ticker", "date"])

def get_window(df: pd.DataFrame, ticker: str, start: str, end: str) -> pd.DataFrame:
    mask = (
        (df["ticker"] == ticker) &
        (df["date"] >= pd.to_datetime(start)) &
        (df["date"] <= pd.to_datetime(end))
    )
    return df.loc[mask].copy()

def get_latest_n_days(df: pd.DataFrame, ticker: str, n: int = 40) -> pd.DataFrame:
    stock_df = df[df["ticker"] == ticker].sort_values("date").copy()
    return stock_df.tail(n)

def build_feature_row(feature_dict: dict, ticker: str) -> dict:
    row = {"ticker": ticker}
    row.update(feature_dict)
    return row

def find_mirrors(df: pd.DataFrame, hero_ticker: str, start: str, end: str, window_size: int = 40):
    hero_window = get_window(df, hero_ticker, start, end)
    if len(hero_window) < 10:
        raise ValueError("Hero window is too small")

    hero_features = compute_window_features(hero_window)
    rows = [build_feature_row(hero_features, hero_ticker)]

    tickers = sorted(df["ticker"].unique())
    for ticker in tickers:
        if ticker == hero_ticker:
            continue
        candidate_window = get_latest_n_days(df, ticker, window_size)
        if len(candidate_window) < 10:
            continue
        candidate_features = compute_window_features(candidate_window)
        rows.append(build_feature_row(candidate_features, ticker))

    feature_df = pd.DataFrame(rows)

    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df[FEATURE_COLUMNS])

    hero_vector = X[0].reshape(1, -1)
    candidate_vectors = X[1:]

    sims = cosine_similarity(hero_vector, candidate_vectors)[0]

    results = feature_df.iloc[1:][["ticker"]].copy()
    results["similarity"] = sims
    results = results.sort_values("similarity", ascending=False)

    return results