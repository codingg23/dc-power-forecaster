"""
train.py  -  training script for power forecaster models

Usage:
    python train.py --data ./data/power.parquet --model lstm --horizon 96 --epochs 50
    python train.py --data ./data/power.parquet --model tft --horizon 96 --epochs 30 --wandb
"""

import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from data.preprocessing import PowerDataset, PreprocessConfig
from models.lstm_forecaster import LSTMForecaster, LSTMConfig, build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        preds = model(X_batch)
        total_loss += criterion(preds, y_batch).item() * len(X_batch)
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", choices=["lstm", "tft"], default="lstm")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--context", type=int, default=672)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--checkpoint-dir", default="./checkpoints/")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    logger.info(f"Training {args.model} on {args.data}, device={DEVICE}")

    # optionally log to W&B  -  skip if not configured
    if args.wandb:
        try:
            import wandb
            wandb.init(project="veltora-power-forecaster", config=vars(args))
            log_fn = lambda d: wandb.log(d)
        except ImportError:
            logger.warning("wandb not installed, skipping")
            log_fn = lambda d: None
    else:
        log_fn = lambda d: None

    # data
    pp_config = PreprocessConfig(horizon=args.horizon, context_len=args.context)
    dataset = PowerDataset(pp_config)
    df = dataset.load_and_preprocess(args.data)
    train_df, val_df, test_df = dataset.train_val_test_split(df)

    dataset.fit_scaler(train_df)

    train_arr = dataset.transform(train_df)
    val_arr = dataset.transform(val_df)

    X_train, y_train = dataset.make_windows(train_arr)
    X_val, y_val = dataset.make_windows(val_arr)

    logger.info(f"Train windows: {len(X_train)}, Val windows: {len(X_val)}")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # model
    if args.model == "lstm":
        model_config = LSTMConfig(
            input_size=X_train.shape[-1],
            hidden_size=args.hidden,
            horizon=args.horizon,
            context_len=args.context,
        )
        model = build_model(model_config).to(DEVICE)
    else:
        raise NotImplementedError("TFT training not implemented here yet  -  use pytorch-forecasting directly")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    criterion = nn.HuberLoss(delta=1.0)  # more robust to outliers than MSE
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = 10  # early stopping

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        log_fn({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        logger.info(f"Epoch {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "config": model_config,
                "val_loss": val_loss,
            }, checkpoint_dir / f"{args.model}_best.pt")
            logger.info(f"  → saved checkpoint (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    dataset.save_scaler(str(checkpoint_dir / "scaler.pkl"))
    logger.info(f"Training done. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
