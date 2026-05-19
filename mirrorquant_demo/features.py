#Turns one stock window into one feature vector
import numpy as np
import pandas as pd

def max_drawdown(close_series: pd.Series) -> float: 
    running_max = close_series.cummax()
    drawdown = (close_series - running_max) / running_max
    return drawdown.min()

def compute_window_features(df: pd.DataFrame) -> dict:
    df = df.sort_values("date").copy()
    df["daily_return"] = df["close"].pct_change()

    total_return = (df["close"].iloc[-1] / df["close"].iloc[0]) - 1
    mean_daily_return = df["daily_return"].mean()
    volatility = df["daily_return"].std()
    mdd = max_drawdown(df["close"])
    avg_volume = df["volume"].mean()

    first_half_vol = df["volume"].iloc[: len(df)//2].mean()
    second_half_vol = df["volume"].iloc[len(df)//2 :].mean()
    volume_trend = (second_half_vol / first_half_vol) - 1 if first_half_vol else 0

    return {
        "total_return": total_return,
        "mean_daily_return": mean_daily_return,
        "volatility": volatility,
        "max_drawdown": mdd,
        "avg_volume": avg_volume,
        "volume_trend": volume_trend,
    }