from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

from mirrorquant_demo.train_vqvae import VQVAE


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PRICES_PATH = DATA_DIR / "prices.csv"
MODEL_PATH = DATA_DIR / "vqvae_model.pt"
EMBEDDINGS_PATH = DATA_DIR / "window_embeddings.npz"
META_PATH = DATA_DIR / "window_embedding_meta.csv"


def load_prices(path: Path = PRICES_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["ticker", "date"]).copy()


def get_window(df: pd.DataFrame, ticker: str, start: str, end: str) -> pd.DataFrame:
    return df[
        (df["ticker"] == ticker)
        & (df["date"] >= pd.to_datetime(start))
        & (df["date"] <= pd.to_datetime(end))
    ].copy()


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

    return window_df[["daily_return", "volume_change", "price_rel"]].to_numpy(dtype=np.float32)


def load_model_bundle():
    return torch.load(MODEL_PATH, map_location="cpu", weights_only=False)


def load_embedding_library():
    data = np.load(EMBEDDINGS_PATH)
    embeddings = data["embeddings"]
    codes = data["codes"]
    meta_df = pd.read_csv(META_PATH, parse_dates=["start_date", "end_date"])
    return embeddings, codes, meta_df


def build_model(bundle):
    model = VQVAE(
        input_dim=bundle["input_dim"],
        latent_dim=bundle["latent_dim"],
        num_codes=bundle["num_codes"],
    )
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    return model


def encode_hero_window(model, bundle, window_df: pd.DataFrame):
    X = make_sequence_features(window_df)

    expected_window_size = bundle["window_size"]
    if len(X) < expected_window_size:
        raise ValueError(
            f"Hero window has {len(X)} rows, but model expects at least {expected_window_size}"
        )

    effective_window_df = window_df.sort_values("date").tail(expected_window_size).copy()
    X = X[-expected_window_size:]

    mean = bundle["mean"]
    std = bundle["std"]

    X_norm = (X - mean[0]) / std[0]
    X_flat = X_norm.reshape(1, -1)

    x_tensor = torch.tensor(X_flat, dtype=torch.float32)

    with torch.no_grad():
        z_e = model.encoder(x_tensor)
        z_q, _, _, code_indices = model.quantizer(z_e)

    embedding = z_q.cpu().numpy()[0]
    code = int(code_indices.cpu().numpy()[0])
    effective_window = {
        "start_date": effective_window_df["date"].iloc[0],
        "end_date": effective_window_df["date"].iloc[-1],
        "window_size": expected_window_size,
    }
    return embedding, code, effective_window


def find_vqvae_mirrors(hero_ticker: str, start: str, end: str, top_k: int = 5):
    df = load_prices()
    hero_window = get_window(df, hero_ticker, start, end)

    if len(hero_window) < 10:
        raise ValueError("Hero window is too small")

    bundle = load_model_bundle()
    model = build_model(bundle)

    hero_embedding, hero_code, effective_window = encode_hero_window(model, bundle, hero_window)

    embeddings, codes, meta_df = load_embedding_library()

    similarities = cosine_similarity(
        hero_embedding.reshape(1, -1),
        embeddings
    )[0]

    results = meta_df.copy()
    results["similarity"] = similarities
    results["candidate_code"] = codes

    results = results[results["ticker"] != hero_ticker].copy()
    results = (
        results.sort_values("similarity", ascending=False)
        .drop_duplicates(subset=["ticker"], keep="first")
        .head(top_k)
        .reset_index(drop=True)
    )

    return results, hero_code, effective_window
