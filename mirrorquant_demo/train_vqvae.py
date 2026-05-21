from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

WINDOWS_PATH = DATA_DIR / "training_windows.npz"
MODEL_PATH = DATA_DIR / "vqvae_model.pt"

BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 1e-3
LATENT_DIM = 16
NUM_CODES = 8
BETA = 0.25
VALIDATION_RATIO = 0.2
SEED = 42


class VectorQuantizer(nn.Module):
    def __init__(self, num_codes: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_codes, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z_e: torch.Tensor):
        distances = (
            torch.sum(z_e ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_e, self.embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(encoding_indices)

        codebook_loss = torch.mean((z_q - z_e.detach()) ** 2)
        commitment_loss = torch.mean((z_e - z_q.detach()) ** 2)

        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, codebook_loss, commitment_loss, encoding_indices


class VQVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, num_codes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.quantizer = VectorQuantizer(num_codes, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor):
        z_e = self.encoder(x)
        z_q, codebook_loss, commitment_loss, encoding_indices = self.quantizer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, z_e, z_q, codebook_loss, commitment_loss, encoding_indices


def load_training_data():
    data = np.load(WINDOWS_PATH)
    X = data["X"].astype(np.float32)
    return X


def normalize_windows(X: np.ndarray):
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X_norm = (X - mean) / std
    return X_norm, mean, std


def flatten_windows(X: np.ndarray):
    num_samples = X.shape[0]
    return X.reshape(num_samples, -1)


def build_dataloaders(X_flat: np.ndarray):
    tensor_x = torch.tensor(X_flat, dtype=torch.float32)
    dataset = TensorDataset(tensor_x)

    val_size = int(len(dataset) * VALIDATION_RATIO)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def run_epoch(model, loader, optimizer=None, beta=BETA, device="cpu"):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_codebook = 0.0
    total_commitment = 0.0
    total_samples = 0

    with torch.set_grad_enabled(is_training):
        for (batch,) in loader:
            batch = batch.to(device)

            x_recon, _, _, codebook_loss, commitment_loss, _ = model(batch)
            recon_loss = torch.mean((x_recon - batch) ** 2)
            loss = recon_loss + codebook_loss + beta * commitment_loss

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = batch.size(0)
            total_loss += loss.item() * batch_size
            total_recon += recon_loss.item() * batch_size
            total_codebook += codebook_loss.item() * batch_size
            total_commitment += commitment_loss.item() * batch_size
            total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "recon": total_recon / total_samples,
        "codebook": total_codebook / total_samples,
        "commitment": total_commitment / total_samples,
    }


def inspect_code_usage(model, X_flat: np.ndarray, device="cpu"):
    model.eval()
    x = torch.tensor(X_flat, dtype=torch.float32).to(device)

    with torch.no_grad():
        _, _, _, _, _, indices = model(x)

    unique, counts = torch.unique(indices.cpu(), return_counts=True)
    usage = {int(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())}
    return usage


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    X = load_training_data()
    print("Original X shape:", X.shape)

    X_norm, mean, std = normalize_windows(X)
    X_flat = flatten_windows(X_norm)
    print("Flattened X shape:", X_flat.shape)

    train_loader, val_loader = build_dataloaders(X_flat)

    input_dim = X_flat.shape[1]
    model = VQVAE(input_dim=input_dim, latent_dim=LATENT_DIM, num_codes=NUM_CODES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, optimizer=optimizer, beta=BETA, device=device)
        val_metrics = run_epoch(model, val_loader, optimizer=None, beta=BETA, device=device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.5f} | "
            f"val_loss={val_metrics['loss']:.5f} | "
            f"train_recon={train_metrics['recon']:.5f} | "
            f"val_recon={val_metrics['recon']:.5f}"
        )

    code_usage = inspect_code_usage(model, X_flat, device=device)
    print("Code usage:", code_usage)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "input_dim": input_dim,
            "latent_dim": LATENT_DIM,
            "num_codes": NUM_CODES,
            "window_size": X.shape[1],
            "num_features": X.shape[2],
        },
        MODEL_PATH,
    )

    print("Saved model to:", MODEL_PATH)


if __name__ == "__main__":
    main()