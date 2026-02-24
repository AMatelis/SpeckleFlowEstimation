import os
import sys
import time
import logging
from typing import List, Tuple, Optional
from fastapi import logger
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optima
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
# metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

# -------------------------------
# Fix import paths for models/src
# -------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for p in [MODELS_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.append(p)

from src.dataloader import create_3d_dataloaders
from models.bloodflow_cnn import BloodFlowCNN3D


# =========================================================
# Utilities
# =========================================================
def set_seed(seed: int = 42):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("bloodflow3d_train")
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


def plot_loss_curve(train_losses: List[float], val_losses: List[float], out_path: str):
    """Publication-ready training/validation loss curve."""
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'o-', label="Train Loss", linewidth=1.5)
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'x-', label="Validation Loss", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training vs Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_pred_vs_true(true: np.ndarray, pred: np.ndarray, out_path: str, 
                      title: str = "Predicted vs True", flow_rates: list = [0, 5, 50, 100, 150, 400]):
    """
    Plot predicted vs true flowrates in μL/min with R² score.
    Forces x-axis ticks to specific flow rates.
    """
    true = np.array(true).flatten() * 400.0  # Assuming max_flow=400 as in your code
    pred = np.array(pred).flatten() * 400.0

    r2 = r2_score(true, pred)
    plt.figure(figsize=(6, 6))
    plt.scatter(true, pred, alpha=0.6, s=18)
    
    # Perfect prediction line
    plt.plot([0, 400], [0, 400], 'r--', linewidth=1.5, label="Perfect")

    plt.xlabel("True Flowrate (μL/min)")
    plt.ylabel("Predicted Flowrate (μL/min)")
    plt.title(f"{title} (R²={r2:.3f})")
    plt.xticks(flow_rates)
    plt.yticks(flow_rates)
    plt.xlim(0, 400)
    plt.ylim(0, 400)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# =========================================================
# Helpers to compute metrics & save predictions
# =========================================================
def compute_regression_metrics(trues: np.ndarray, preds: np.ndarray) -> dict:
    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)
    ev = explained_variance_score(trues, preds)
    return {"mse": float(mse), "mae": float(mae), "r2": float(r2), "explained_variance": float(ev)}


def save_epoch_predictions(epoch: int, trues: np.ndarray, preds: np.ndarray, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for i, (t, p) in enumerate(zip(trues.tolist(), preds.tolist())):
        tval = float(np.squeeze(t))
        pval = float(np.squeeze(p))
        rows.append({
            "index": i,
            "true": tval,
            "pred": pval,
            "abs_error": abs(pval - tval),
            "squared_error": (pval - tval) ** 2
        })

    import csv
    fname = os.path.join(out_dir, f"epoch_{epoch:03d}_predictions.csv")
    with open(fname, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return fname


# =========================================================
# Training / Validation Functions
# =========================================================
def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: optima.Optimizer, criterion: nn.Module,
                    device: torch.device, amp_scaler: Optional[GradScaler], grad_clip: float,
                    logger: Optional[logging.Logger] = None) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    use_amp = amp_scaler is not None and device.type == "cuda"

    for xb, yb in tqdm(loader, desc="Train", leave=False, dynamic_ncols=True):
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            preds = model(xb)
            loss = criterion(preds, yb)

        if amp_scaler:
            amp_scaler.scale(loss).backward()
            if grad_clip > 0:
                amp_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running_loss += loss.item() * xb.size(0)
        total += xb.size(0)

    avg_loss = running_loss / total if total > 0 else float("nan")
    if logger:
        logger.info(f"Train Loss: {avg_loss:.6f}")
    return avg_loss


def validate_and_collect(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    losses = 0.0
    n = 0
    all_preds = []
    all_trues = []
    use_amp = False

    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="Val (collect)", leave=False, dynamic_ncols=True):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with autocast(enabled=use_amp):
                preds = model(xb)
                loss = criterion(preds, yb)
            losses += loss.item() * xb.size(0)
            n += xb.size(0)
            all_preds.append(preds.cpu().numpy())
            all_trues.append(yb.cpu().numpy())

    if n == 0:
        return float("nan"), np.array([]), np.array([])

    preds_arr = np.concatenate(all_preds, axis=0).reshape(-1)
    trues_arr = np.concatenate(all_trues, axis=0).reshape(-1)
    avg_loss = losses / n
    return avg_loss, trues_arr, preds_arr


# =========================================================
# Main Training Loop
# =========================================================
def train(data_dir: str, output_dir: str, batch_size: int = 8, stride: int = 1, num_epochs: int = 50,
          lr: float = 1e-3, weight_decay: float = 1e-5, num_workers: int = 4, grad_clip: float = 1.0,
          seed: int = 42, target_resolution: Tuple[int, int] = (128, 128), channel_width_um: float = 100.0):

    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    predictions_dir = os.path.join(output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    logger = setup_logger(output_dir)
    logger.info("Preparing dataloaders...")

    train_loader, val_loader = create_3d_dataloaders(
        data_folder=data_dir,
        batch_size=batch_size,
        stride=stride,
        target_resolution=target_resolution,
        channel_width_um=channel_width_um,
        num_workers=num_workers
    )

    # ===========================
    # FIXED: dataset fraction
    # ===========================
    dataset_fraction = 0.8  # change to 0.2, 0.4, 0.6, 0.8, 1.0 as needed
    train_dataset = train_loader.dataset
    subset_size = int(len(train_dataset) * dataset_fraction)
    from torch.utils.data import Subset
    train_loader = DataLoader(
        Subset(train_dataset, range(subset_size)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    logger.info(f"Using {subset_size}/{len(train_dataset)} samples for training ({dataset_fraction*100:.0f}%)")
    logger.info(f"Using {subset_size}/{len(train_dataset)} samples for training ({dataset_fraction*100:.0f}%)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BloodFlowCNN3D(input_channels=1, output_channels=1).to(device)
    logger.info(f"Model created on {device}")

    criterion = nn.MSELoss()
    optimizer = optima.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    amp_scaler = GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    best_ckpt = os.path.join(checkpoints_dir, "best_model.pth")

    epoch_metrics_csv = os.path.join(predictions_dir, "epoch_metrics.csv")
    if os.path.exists(epoch_metrics_csv):
        os.remove(epoch_metrics_csv)
    with open(epoch_metrics_csv, "w") as f:
        f.write("epoch,mse,mae,r2,explained_variance,train_loss,val_loss\n")

    logger.info("Starting training...")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, amp_scaler, grad_clip, logger)
        val_loss, trues_arr, preds_arr = validate_and_collect(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        metrics = compute_regression_metrics(trues_arr, preds_arr) if trues_arr.size else \
                  {"mse": np.nan, "mae": np.nan, "r2": np.nan, "explained_variance": np.nan}

        if trues_arr.size:
            pred_csv = save_epoch_predictions(epoch, trues_arr, preds_arr, predictions_dir)
            logger.info(f"Saved predictions for epoch {epoch} -> {pred_csv}")

        with open(epoch_metrics_csv, "a") as f:
            f.write(f"{epoch},{metrics['mse']:.6e},{metrics['mae']:.6e},{metrics['r2']:.6f},{metrics['explained_variance']:.6f},{train_loss:.6e},{val_loss:.6e}\n")

        logger.info(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                    f"MSE: {metrics['mse']:.6e} | MAE: {metrics['mae']:.6e} | R2: {metrics['r2']:.4f} | "
                    f"Time: {time.time() - t0:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict()}, best_ckpt)
            logger.info(f"Saved best model to {best_ckpt}")

        scheduler.step(val_loss)

    logger.info("Training complete. Loading best model for final evaluation.")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    final_val_loss, trues_arr, preds_arr = validate_and_collect(model, val_loader, criterion, device)
    final_metrics = compute_regression_metrics(trues_arr, preds_arr) if trues_arr.size else {}

    logger.info(f"Final Validation MSE Loss: {final_val_loss:.6f}")
    logger.info(f"Final metrics: {final_metrics}")

    if trues_arr.size and preds_arr.size:
        save_epoch_predictions(0, trues_arr, preds_arr, predictions_dir)
        plot_pred_vs_true(trues_arr, preds_arr, 
                          os.path.join(output_dir, "final_pred_vs_true.png"),
                          title=f"Final Pred vs True", 
                          flow_rates=[0, 5, 50, 100, 150, 400])
    plot_loss_curve(train_losses, val_losses, os.path.join(output_dir, "training_loss.png"))
    logger.info(f"All outputs saved to {output_dir} and predictions to {predictions_dir}.")

    return {
        "best_ckpt": best_ckpt,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "final_metrics": final_metrics,
        "predictions_dir": predictions_dir
    }


# =========================================================
# CLI
# =========================================================
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="3D Blood Flow CNN Regression Training")
    
    # Existing mode argument
    parser.add_argument("--mode", type=str, required=True, choices=["train", "evaluate"])
    
    # Add your data folder argument
    parser.add_argument("--data_dir", type=str, required=True, help="Path to your dataset folder")
    
    # Existing arguments
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


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