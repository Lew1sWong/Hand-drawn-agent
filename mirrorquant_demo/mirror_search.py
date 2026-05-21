# Mirror Search: Find stocks with similar historical behavior to a "hero" stock
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from .features import compute_window_features

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

def build_feature_row(
    feature_dict: dict,
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict:
    row = {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
    }
    row.update(feature_dict)
    return row

def iter_candidate_windows(stock_df: pd.DataFrame, window_size: int):
    stock_df = stock_df.sort_values("date").copy()
    for start_idx in range(0, len(stock_df) - window_size + 1):
        end_idx = start_idx + window_size
        yield stock_df.iloc[start_idx:end_idx].copy()

def find_mirrors(df: pd.DataFrame, hero_ticker: str, start: str, end: str, window_size: int = 40):
    hero_window = get_window(df, hero_ticker, start, end)
    if len(hero_window) < 10:
        raise ValueError("Hero window is too small")

    # Compare windows with the same length as the hero period so each stock is
    # judged against the same amount of market history.
    window_size = len(hero_window)
    hero_features = compute_window_features(hero_window)
    rows = [
        build_feature_row(
            hero_features,
            hero_ticker,
            hero_window["date"].iloc[0], #iloc: Pandas method to access rows by integer position
            hero_window["date"].iloc[-1],
        )
    ]

    tickers = sorted(df["ticker"].unique())
    for ticker in tickers:
        if ticker == hero_ticker:
            continue
        stock_df = df[df["ticker"] == ticker].sort_values("date").copy()
        if len(stock_df) < window_size:
            continue

        for candidate_window in iter_candidate_windows(stock_df, window_size):
            candidate_features = compute_window_features(candidate_window)
            rows.append(
                build_feature_row(
                    candidate_features,
                    ticker,
                    candidate_window["date"].iloc[0],
                    candidate_window["date"].iloc[-1],
                )
            )

    feature_df = pd.DataFrame(rows)

    scaler = StandardScaler() # Rescales each feature column so they are on a more comparable scale, which is important for cosine similarity to work well.
                              # Turns each column centred around 0, and measure how far from typical behaviour (units) each window is, rather than being dominated by raw magnitude differences between features.
    X = scaler.fit_transform(feature_df[FEATURE_COLUMNS])

    hero_vector = X[0].reshape(1, -1)
    candidate_vectors = X[1:]

    sims = cosine_similarity(hero_vector, candidate_vectors)[0]

    results = feature_df.iloc[1:][["ticker", "start_date", "end_date"]].copy()
    results["similarity"] = sims
    results = (
        results.sort_values("similarity", ascending=False)
        .drop_duplicates(subset=["ticker"], keep="first")
        .sort_values("similarity", ascending=False)
        .reset_index(drop=True)
    )

    return results
