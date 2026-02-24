import os
import sys
import time
import argparse
import logging
from typing import Tuple, List, Optional

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler
from torch import amp
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp.autocast")

from src.dataloader import create_dataloaders
from models.bloodflow_cnn import get_model

# ----------------------
# Flow rate classes
# ----------------------
FLOW_CLASSES = [0, 5, 50, 100, 150, 400]
FLOW_TO_CLASS = {flow: i for i, flow in enumerate(FLOW_CLASSES)}
CLASS_TO_FLOW = {i: flow for flow, i in FLOW_TO_CLASS.items()}

# ----------------------
# Utilities
# ----------------------
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
    logger = logging.getLogger("bloodflow_3d_train")
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
        w.writerow(["Index", "TrueFlow", "PredictedFlow", "Correct"])
        for i, (t, p) in enumerate(zip(targets, preds)):
            correct = int(t == p)
            w.writerow([i, FLOW_CLASSES[t], FLOW_CLASSES[p], correct])

def plot_pred_scatter(targets: List[int], preds: List[int], out_path: str) -> None:
    true_flows = [FLOW_CLASSES[t] for t in targets]
    pred_flows = [FLOW_CLASSES[p] for p in preds]
    plt.figure(figsize=(6, 6))
    plt.scatter(true_flows, pred_flows, alpha=0.7, edgecolors="k")
    mn = min(min(true_flows), min(pred_flows))
    mx = max(max(true_flows), max(pred_flows))
    plt.plot([mn, mx], [mn, mx], "r--", label="Ideal")
    plt.xlabel("True Flow (uL/min)")
    plt.ylabel("Predicted Flow (uL/min)")
    plt.title("Predicted vs True Flow Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

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

def plot_confusion(targets: List[int], preds: List[int], out_path: str) -> None:
    cm = confusion_matrix(targets, preds, labels=list(range(len(FLOW_CLASSES))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=FLOW_CLASSES)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(out_path)
    plt.close()

# ----------------------
# Dataset reduction utility
# ----------------------
def reduce_dataset(dataset, fraction: float, seed: int = 42):
    if fraction >= 1.0:
        return dataset
    np.random.seed(seed)
    total_len = len(dataset)
    reduced_len = int(total_len * fraction)
    indices = np.random.choice(total_len, reduced_len, replace=False)
    return Subset(dataset, indices)

# ----------------------
# Training / Validation
# ----------------------
def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer,
                    criterion: nn.Module, device: torch.device,
                    amp_scaler: Optional[GradScaler], grad_clip: float,
                    logger: Optional[logging.Logger] = None) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    for xb, yb in tqdm(loader, desc="train", leave=False, dynamic_ncols=True):
        xb = xb.to(device, non_blocking=True)
        yb_class = torch.tensor([FLOW_TO_CLASS[int(y.item())] for y in yb], device=device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast("cuda", enabled=(amp_scaler is not None and device.type == "cuda")):
            logits = model(xb)
            loss = criterion(logits, yb_class)

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

def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, List[int], List[int], float]:
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="val", leave=False, dynamic_ncols=True):
            xb = xb.to(device, non_blocking=True)
            yb_class = torch.tensor([FLOW_TO_CLASS[int(y.item())] for y in yb], device=device, dtype=torch.long)

            logits = model(xb)
            loss = criterion(logits, yb_class)
            running_loss += loss.item() * xb.size(0)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb_class).sum().item()
            total += yb_class.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(yb_class.cpu().tolist())

    avg_loss = running_loss / total if total > 0 else float("nan")
    acc = correct / total if total > 0 else 0.0
    return avg_loss, all_preds, all_targets, acc

# ----------------------                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
# Main training
# ----------------------
def train(data_dir, output_dir, batch_size, sequence_len, stride, num_epochs,
          lr, weight_decay, patience, test_split, num_workers, grad_clip,
          seed, augment_train, data_fraction=0.8, safe_mode=False):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    logger = setup_logger(output_dir)

    # Load datasets first
    train_loader_full, val_loader, _ = create_dataloaders(
        data_folder=data_dir,
        batch_size=batch_size,
        sequence_len=sequence_len,
        test_split=test_split,
        stride=stride,
        num_workers=num_workers,
        normalize_mode=None,
        augment_train=augment_train,
        safe_mode=safe_mode,
    )

    # Reduce training dataset BEFORE DataLoader initialization
    reduced_train_dataset = reduce_dataset(train_loader_full.dataset, fraction=data_fraction, seed=seed)
    train_loader = DataLoader(
        reduced_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers if not safe_mode else 0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=len(FLOW_CLASSES))
    model.to(device)
    logger.info(f"Model created and moved to {device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience)
    amp_scaler = GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    best_ckpt = os.path.join(checkpoints_dir, "best_model.pth")
    train_losses, val_losses = [], []

    logger.info("Starting training loop...")
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, amp_scaler, grad_clip, logger)
        val_loss, val_preds, val_targets, val_acc = validate(model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        logger.info(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Acc: {val_acc*100:.2f}% | Time: {time.time()-t0:.1f}s")

        save_predictions_csv(val_preds, val_targets, os.path.join(output_dir, f"predictions_epoch_{epoch}.csv"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict()}, best_ckpt)
            logger.info(f"Saved best model to {best_ckpt}")

    # Final outputs
    plot_loss_curve(train_losses, val_losses, os.path.join(output_dir, "training_loss.png"))
    val_loss, val_preds, val_targets, val_acc = validate(model, val_loader, device)
    save_predictions_csv(val_preds, val_targets, os.path.join(output_dir, "final_predictions.csv"))
    plot_pred_scatter(val_targets, val_preds, os.path.join(output_dir, "final_pred_vs_true.png"))
    plot_confusion(val_targets, val_preds, os.path.join(output_dir, "confusion_matrix.png"))
    logger.info(f"Training complete | Final Val Acc: {val_acc*100:.2f}%")

# ----------------------
# Evaluation
# ----------------------
def evaluate(checkpoint: str, data_dir: str, output_dir: str,
             batch_size: int, sequence_len: int, stride: int, num_workers: int):
    logger = setup_logger(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating on device: {device}")

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    ck = torch.load(checkpoint, map_location=device)
    model = get_model(num_classes=len(FLOW_CLASSES))
    model.to(device)
    model.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
    model.eval()

    _, val_loader, _ = create_dataloaders(
        data_folder=data_dir,
        batch_size=batch_size,
        sequence_len=sequence_len,
        test_split=0.2,
        stride=stride,
        num_workers=num_workers,
        normalize_mode=None,
        augment_train=False,
    )

    val_loss, val_preds, val_targets, val_acc = validate(model, val_loader, device)
    logger.info(f"Eval results | Loss: {val_loss:.6f} | Accuracy: {val_acc*100:.2f}%")
    save_predictions_csv(val_preds, val_targets, os.path.join(output_dir, "eval_predictions.csv"))
    plot_pred_scatter(val_targets, val_preds, os.path.join(output_dir, "eval_pred_vs_true.png"))
    plot_confusion(val_targets, val_preds, os.path.join(output_dir, "eval_confusion_matrix.png"))

# ----------------------
# CLI
# ----------------------
def parse_args():
    p = argparse.ArgumentParser(description="BloodFlow 3D-CNN Class Prediction")
    p.add_argument("--mode", choices=["train","eval"], required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--sequence_len", type=int, default=5)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--augment_train", action="store_true")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--data_fraction", type=float, default=0.4, help="Fraction of training data to use (0-1).")
    return p.parse_args()

def main():
    args = parse_args()
    if args.mode == "train":
        train(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            sequence_len=args.sequence_len,
            stride=args.stride,
            num_epochs=args.num_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=10,
            test_split=0.2,
            num_workers=args.num_workers,
            grad_clip=args.grad_clip,
            seed=args.seed,
            augment_train=args.augment_train,
            data_fraction=args.data_fraction,
        )
    else:
        if not args.checkpoint:
            raise ValueError("Please provide --checkpoint for eval mode")
        evaluate(
            checkpoint=args.checkpoint,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            sequence_len=args.sequence_len,
            stride=args.stride,
            num_workers=args.num_workers,
        )

if __name__ == "__main__":
    main()