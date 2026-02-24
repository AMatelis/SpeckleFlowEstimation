import os
import sys
import time
import logging
import random
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# =========================================================
# Handle imports for models and src
# =========================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from models.bloodflow_cnn import BloodFlowCNN2D
from src.dataloader import create_dataloaders

# =========================================================
# Utilities
# =========================================================
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("bloodflow_regression_train")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
        fh = logging.FileHandler(os.path.join(output_dir, "training.log"))
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger

# =========================================================
# Plotting
# =========================================================
FLOW_RATES = [0, 5, 50, 100, 150, 400]  # ensure 5 uL is included

def plot_loss_curve(train_losses: List[float], val_losses: List[float], out_path: str) -> None:
    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(train_losses)+1), train_losses, marker="o", label="Train Loss")
    plt.plot(range(1, len(val_losses)+1), val_losses, marker="x", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_predictions_vs_truth(y_true: List[float], y_pred: List[float], out_path: str, title="Predictions vs True") -> None:
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, c='blue', alpha=0.6, edgecolors='k')

    min_flow = min(FLOW_RATES)
    max_flow = max(FLOW_RATES)

    plt.plot([min_flow, max_flow], [min_flow, max_flow], 'r--', lw=2, label="Ideal")
    plt.xlim(min_flow-5, max_flow+20)
    plt.ylim(min_flow-5, max_flow+20)
    plt.xticks(FLOW_RATES)
    plt.yticks(FLOW_RATES)

    plt.xlabel("True Flowrate (uL/min)")
    plt.ylabel("Predicted Flowrate (uL/min)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# =========================================================
# Training & Validation
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion, device, amp_scaler, grad_clip: float, logger=None) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    for xb, yb in tqdm(loader, desc="Train", leave=False, dynamic_ncols=True):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=(amp_scaler is not None and device.type=="cuda")):
            preds = model(xb)
            if preds.shape != yb.shape:
                yb = yb.view_as(preds)
            loss = criterion(preds, yb)
        if amp_scaler:
            amp_scaler.scale(loss).backward()
            if grad_clip > 0:
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        running_loss += loss.item() * xb.size(0)
        total += xb.size(0)
    avg_loss = running_loss / total if total > 0 else float("nan")
    if logger:
        logger.info(f"Train Loss: {avg_loss:.6f}")
    return avg_loss

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    y_trues, y_preds = [], []
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="Val", leave=False, dynamic_ncols=True):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            preds = model(xb)
            if preds.shape != yb.shape:
                yb = yb.view_as(preds)
            loss = criterion(preds, yb)
            running_loss += loss.item() * xb.size(0)
            total += xb.size(0)
            y_trues.extend(yb.detach().cpu().numpy().flatten())
            y_preds.extend(preds.detach().cpu().numpy().flatten())
    avg_loss = running_loss / total if total > 0 else float("nan")
    return avg_loss, np.array(y_trues), np.array(y_preds)

def save_epoch_predictions(epoch, y_true, y_pred, output_dir, logger, plot=False):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"epoch_{epoch}_predictions.csv")
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"# Epoch {epoch} Metrics: MSE={mse:.6f}, MAE={mae:.6f}, R2={r2:.6f}"])
        writer.writerow(["True Flowrate", "Predicted Flowrate"])
        for t, p in zip(y_true, y_pred):
            writer.writerow([t, p])
    logger.info(f"Saved epoch {epoch} predictions to {csv_path}")

    if plot:
        plot_path = os.path.join(output_dir, f"epoch_{epoch}_predictions.png")
        plot_predictions_vs_truth(y_true, y_pred, plot_path, title=f"Epoch {epoch} Predictions vs True")

# =========================================================
# Main training loop
# =========================================================
def train(
    data_dir: str,
    output_dir: str,
    batch_size: int = 8,
    stride: int = 1,
    num_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    num_workers: int = 4,
    grad_clip: float = 1.0,
    seed: int = 42,
    target_resolution=(128,128),
    plot_every_n_epochs: int = 10, 
    dataset_fraction: float = 1.0
):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    epoch_preds_dir = os.path.join(output_dir, "epoch_predictions")
    os.makedirs(epoch_preds_dir, exist_ok=True)

    logger = setup_logger(output_dir)
    logger.info("Preparing dataloaders...")

    train_loader, val_loader, *_ = create_dataloaders(
        data_dir,
        batch_size=batch_size,
        stride=stride,
        num_workers=num_workers,
        target_resolution=target_resolution,
        fraction=dataset_fraction
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BloodFlowCNN2D(input_channels=1).to(device)
    logger.info(f"Model created and moved to {device}")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    amp_scaler = GradScaler(enabled=(device.type=="cuda"))

    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    best_ckpt = os.path.join(checkpoints_dir, "best_model.pth")

    logger.info("Starting training loop...")
    for epoch in range(1, num_epochs+1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, amp_scaler, grad_clip, logger)
        val_loss, val_trues, val_preds = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logger.info(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {time.time()-t0:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict()}, best_ckpt)
            logger.info(f"Saved best model to {best_ckpt}")

        # Save per-epoch predictions
        save_epoch_predictions(
            epoch,
            val_trues,
            val_preds,
            epoch_preds_dir,
            logger,
            plot=(epoch % plot_every_n_epochs == 0 or epoch == num_epochs)
        )

        scheduler.step(val_loss)

    # Final evaluation and plotting
    logger.info("Training complete. Loading best model for final evaluation.")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    final_val_loss, final_trues, final_preds = validate(model, val_loader, criterion, device)
    logger.info(f"Final Val MSE Loss: {final_val_loss:.6f}")

    plot_loss_curve(train_losses, val_losses, os.path.join(output_dir, "training_loss.png"))
    plot_predictions_vs_truth(final_trues, final_preds, os.path.join(output_dir, "predictions_vs_true.png"), title="Final Predictions vs True")

    # Save final predictions CSV
    save_epoch_predictions("final", final_trues, final_preds, output_dir, logger, plot=True)

# =========================================================
# CLI
# =========================================================
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Train 2D BloodFlow CNN")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset_fraction", type=float, default=0.8, help="Fraction of dataset to use (0-1)")
    return p.parse_args()

def main():
    args = parse_args()
    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        stride=args.stride,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        grad_clip=args.grad_clip,
        seed=args.seed
    )

if __name__ == "__main__":
    main()