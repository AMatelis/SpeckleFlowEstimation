import os
import sys
import time
import logging
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler
from torch import amp
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp.autocast")

from src.dataloader import create_dataloaders
from models.bloodflow_cnn import BloodFlowCNN2D

# =========================================================
# Utilities
# =========================================================

def set_seed(seed: int = 42) -> None:
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("bloodflow_classification_train")
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

def save_predictions_csv(preds: List[int], targets: List[int], out_path: str) -> None:
    import csv
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Index", "TrueClass", "PredictedClass"])
        for i, (t, p) in enumerate(zip(targets, preds)):
            w.writerow([i, t, p])

def plot_loss_curve(train_losses: List[float], val_losses: List[float], out_path: str) -> None:
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, marker="o", label="Train Loss")
    plt.plot(range(1, len(val_losses)+1), val_losses, marker="x", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropy Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# =========================================================
# NEW: Confusion Matrix Plot
# =========================================================
def plot_confusion_matrix(targets, preds, out_path):
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")

    tick_marks = np.arange(len(np.unique(targets)))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    # write numbers inside cells
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# =========================================================
# NEW: Predictions vs True Plot
# =========================================================
def plot_preds_vs_true(targets, preds, out_path):
    plt.figure(figsize=(7, 4))
    plt.scatter(range(len(preds)), preds, marker="o", label="Predicted")
    plt.scatter(range(len(targets)), targets, marker="x", label="True")
    plt.xlabel("Sample Index")
    plt.ylabel("Class")
    plt.title("Predicted vs True Labels")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# =========================================================
# Training / Validation
# =========================================================

def _fix_input_shape(xb: torch.Tensor) -> torch.Tensor:
    if xb.dim() == 5 and xb.size(2) == 1:
        xb = xb.squeeze(2)
    return xb

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    amp_scaler: Optional[GradScaler],
    grad_clip: float,
    logger: Optional[logging.Logger] = None,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    for xb, yb in tqdm(loader, desc="train", leave=False, dynamic_ncols=True):
        xb = _fix_input_shape(xb).to(device, non_blocking=True)
        yb = yb.long().to(device)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast("cuda", enabled=(amp_scaler is not None and device.type == "cuda")):
            logits = model(xb)
            loss = criterion(logits, yb)

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

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, List[int], List[int]]:
    model.eval()
    running_loss = 0.0
    total = 0
    all_preds: List[int] = []
    all_targets: List[int] = []
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="val", leave=False, dynamic_ncols=True):
            xb = _fix_input_shape(xb).to(device, non_blocking=True)
            yb = yb.long().to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            running_loss += loss.item() * xb.size(0)
            total += xb.size(0)

            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(yb.cpu().tolist())

    avg_loss = running_loss / total if total > 0 else float("nan")
    return avg_loss, all_preds, all_targets

# =========================================================
# Main training loop
# =========================================================

def train(
    data_dir,
    output_dir,
    batch_size,
    sequence_len,
    stride,
    num_epochs,
    lr,
    weight_decay,
    num_workers,
    grad_clip,
    seed,
    num_classes,
    augment_train,
    data_fraction=1.0,
    safe_mode=False,
):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    logger = setup_logger(output_dir)
    logger.info("Preparing dataloaders...")

    train_loader, val_loader, full_dataset = create_dataloaders(
        data_folder=data_dir,
        batch_size=batch_size,
        sequence_len=sequence_len,
        stride=stride,
        num_workers=(0 if safe_mode else num_workers),
        normalize_mode="scale",
        cache_file=None,
        augment_train=augment_train,
        safe_mode=safe_mode,
    )

    if data_fraction < 1.0:
        subset = Subset(full_dataset, np.random.choice(len(full_dataset), int(len(full_dataset) * data_fraction), replace=False))
        train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=(0 if safe_mode else num_workers))

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BloodFlowCNN2D(input_channels=1, num_classes=num_classes).to(device)
    logger.info(f"Model created with {num_classes} classes and moved to {device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    amp_scaler = GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    best_ckpt = os.path.join(checkpoints_dir, "best_model.pth")

    logger.info("Starting training loop...")
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, amp_scaler, grad_clip, logger)
        val_loss, val_preds, val_targets = validate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logger.info(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {time.time()-t0:.1f}s")

        save_predictions_csv(val_preds, val_targets, os.path.join(output_dir, f"predictions_epoch_{epoch}.csv"))

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict()}, best_ckpt)
            logger.info(f"Saved best model to {best_ckpt}")

    # Final evaluation
    ck = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    final_loss, final_preds, final_targets = validate(model, val_loader, criterion, device)
    acc = accuracy_score(final_targets, final_preds)
    logger.info(f"Final metrics | Val Loss: {final_loss:.6f} | Accuracy: {acc:.4f}")
    logger.info("Classification Report:\n" + classification_report(final_targets, final_preds))

    save_predictions_csv(final_preds, final_targets, os.path.join(output_dir, "final_predictions.csv"))
    plot_loss_curve(train_losses, val_losses, os.path.join(output_dir, "training_loss.png"))

    # NEW: Generate CM & preds-vs-true
    plot_confusion_matrix(final_targets, final_preds, os.path.join(output_dir, "confusion_matrix.png"))
    plot_preds_vs_true(final_targets, final_preds, os.path.join(output_dir, "pred_vs_true.png"))

    logger.info("Training complete. Confusion matrix and prediction plots saved.")

# =========================================================
# CLI
# =========================================================

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Train BloodFlow 2D-CNN Classification")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--sequence_len", type=int, default=1)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--augment_train", action="store_true")
    p.add_argument("--data_fraction", type=float, default=0.2)
    return p.parse_args()

def main():
    args = parse_args()
    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        sequence_len=args.sequence_len,
        stride=args.stride,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        grad_clip=args.grad_clip,
        seed=args.seed,
        num_classes=args.num_classes,
        augment_train=args.augment_train,
        data_fraction=args.data_fraction,
    )

if __name__ == "__main__":
    main()