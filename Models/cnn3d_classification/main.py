import argparse
import os
import sys
import platform
from typing import Optional, List

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from src.dataloader import create_dataloaders
from models.bloodflow_cnn import get_model
from src.train import FLOW_CLASSES, FLOW_TO_CLASS, CLASS_TO_FLOW

# Ensure root path for imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

def evaluate_model(
    checkpoint_path: str,
    data_dir: str,
    sequence_len: int = 5,
    batch_size: int = 8,
    device: Optional[str] = None,
    output_dir: Optional[str] = None,
    num_workers: int = 0,
    safe_mode: bool = True,
) -> None:
    """Load a trained class-based model and evaluate on validation data."""
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Evaluating on device: {device}")

    _, val_loader, _ = create_dataloaders(
        data_folder=data_dir,
        batch_size=batch_size,
        sequence_len=sequence_len,
        test_split=0.2,
        stride=1,
        num_workers=num_workers,
        normalize_mode=None,
        augment_train=False,
        safe_mode=safe_mode,
    )

    if val_loader is None or len(val_loader) == 0:
        raise RuntimeError("Validation data not found or improperly formatted.")

    model = get_model(num_classes=len(FLOW_CLASSES)).to(device)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise ValueError("Invalid checkpoint format.")

    model.eval()

    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            # Convert true flow values to class indices
            all_targets.extend([FLOW_TO_CLASS[int(y.item())] for y in targets])

    acc = accuracy_score(all_targets, all_preds)
    print(f"\n[RESULTS] Accuracy: {acc*100:.2f}%")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame({
            "TargetFlow": [FLOW_CLASSES[t] for t in all_targets],
            "PredictedFlow": [FLOW_CLASSES[p] for p in all_preds],
            "Correct": [int(t==p) for t,p in zip(all_targets, all_preds)]
        })
        df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

        # Scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(df["TargetFlow"], df["PredictedFlow"], alpha=0.6, edgecolors='k')
        mn = min(df["TargetFlow"].min(), df["PredictedFlow"].min())
        mx = max(df["TargetFlow"].max(), df["PredictedFlow"].max())
        plt.plot([mn, mx], [mn, mx], 'r--', label="Ideal")
        plt.title("Blood Flow Rate Prediction (Class-based)")
        plt.xlabel("True Flow Rate (uL/min)")
        plt.ylabel("Predicted Flow Rate (uL/min)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "flowrate_scatter.png"))
        plt.close()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blood Flow Estimation (Class-based)")
    parser.add_argument("--mode", choices=["train", "evaluate"], required=True)
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint (.pth) for evaluation")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--sequence_len", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Windows safety mode
    safe_mode = False
    if platform.system() == "Windows":
        print("[INFO] Windows detected — enabling DataLoader safe mode (num_workers=0)")
        args.num_workers = 0
        safe_mode = True

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Running on device: {device}")

    outputs_dir = os.path.join(ROOT_DIR, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    try:
        if args.mode == "train":
            from src.train import train
            train(
                data_dir=args.data_dir,
                output_dir=outputs_dir,
                batch_size=args.batch_size,
                sequence_len=args.sequence_len,
                stride=1,
                num_epochs=args.num_epochs,
                lr=1e-3,
                weight_decay=1e-5,
                patience=10,
                test_split=0.2,
                num_workers=args.num_workers,
                grad_clip=1.0,
                seed=args.seed,
                augment_train=True,
                safe_mode=safe_mode,
            )
        elif args.mode == "evaluate":
            if not args.checkpoint:
                raise ValueError("--checkpoint is required for evaluation")
            evaluate_model(
                checkpoint_path=args.checkpoint,
                data_dir=args.data_dir,
                output_dir=outputs_dir,
                batch_size=args.batch_size,
                sequence_len=args.sequence_len,
                num_workers=args.num_workers,
                safe_mode=safe_mode,
            )
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()