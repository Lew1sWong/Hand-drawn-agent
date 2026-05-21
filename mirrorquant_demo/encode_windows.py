from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mirrorquant_demo.train_vqvae import VQVAE


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

WINDOWS_PATH = DATA_DIR / "training_windows.npz"
META_PATH = DATA_DIR / "training_windows_meta.csv"
MODEL_PATH = DATA_DIR / "vqvae_model.pt"

EMBEDDINGS_PATH = DATA_DIR / "window_embeddings.npz"
OUTPUT_META_PATH = DATA_DIR / "window_embedding_meta.csv"


def load_model_bundle():
    # This checkpoint was created locally by train_vqvae.py and includes
    # NumPy arrays for normalization, so we explicitly allow full loading.
    bundle = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    return bundle


def load_windows():
    data = np.load(WINDOWS_PATH)
    X = data["X"].astype(np.float32)
    meta_df = pd.read_csv(META_PATH)
    return X, meta_df


def normalize_windows(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (X - mean) / std


def flatten_windows(X: np.ndarray):
    return X.reshape(X.shape[0], -1)


def encode_all_windows(model: VQVAE, X_flat: np.ndarray):
    model.eval()

    x_tensor = torch.tensor(X_flat, dtype=torch.float32)

    with torch.no_grad():
        z_e = model.encoder(x_tensor)
        z_q, _, _, code_indices = model.quantizer(z_e)

    embeddings = z_q.cpu().numpy()
    codes = code_indices.cpu().numpy()

    return embeddings, codes


def main():
    bundle = load_model_bundle()

    mean = bundle["mean"]
    std = bundle["std"]
    input_dim = bundle["input_dim"]
    latent_dim = bundle["latent_dim"]
    num_codes = bundle["num_codes"]

    X, meta_df = load_windows()
    X_norm = normalize_windows(X, mean, std)
    X_flat = flatten_windows(X_norm)

    model = VQVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_codes=num_codes,
    )
    model.load_state_dict(bundle["model_state_dict"])

    embeddings, codes = encode_all_windows(model, X_flat)

    np.savez_compressed(
        EMBEDDINGS_PATH,
        embeddings=embeddings,
        codes=codes,
    )

    meta_df = meta_df.copy()
    meta_df["code"] = codes
    meta_df.to_csv(OUTPUT_META_PATH, index=False)

    print("Encoded windows successfully")
    print("Embeddings shape:", embeddings.shape)
    print("Codes shape:", codes.shape)
    print("Saved embeddings to:", EMBEDDINGS_PATH)
    print("Saved metadata to:", OUTPUT_META_PATH)


if __name__ == "__main__":
    main()
