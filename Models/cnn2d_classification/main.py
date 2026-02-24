import argparse
import os
import sys
import platform
from typing import Optional, List

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

# Ensure root path for imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.train import train
from src.dataloader import create_dataloaders
from models.bloodflow_cnn import BloodFlowCNN2D


def evaluate_model(
    checkpoint_path: str,
    data_dir: str,
    sequence_len: int = 1,
    batch_size: int = 8,
    device: Optional[str] = None,
    output_dir: Optional[str] = None,
    num_workers: int = 0,
    safe_mode: bool = True,
    num_classes: int = 6,
) -> None:
    """Load a trained model and evaluate on validation data."""
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Evaluating on device: {device}")

    train_loader, val_loader, _ = create_dataloaders(
        data_folder=data_dir,
        batch_size=batch_size,
        sequence_len=sequence_len,
        test_split=0.2,
        stride=1,
        num_workers=num_workers,
        normalize_mode="scale",
        augment_train=False,
        safe_mode=safe_mode,
    )

    if val_loader is None or len(val_loader) == 0:
        raise RuntimeError("Validation data not found or improperly formatted.")

    model = BloodFlowCNN2D(input_channels=1, num_classes=num_classes).to(device)

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
            targets = targets.long().to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    acc = accuracy_score(all_targets, all_preds)
    report = classification_report(all_targets, all_preds)
    print("\n[RESULTS]")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", report)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame({"Target": all_targets, "Prediction": all_preds})
        df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

        # Confusion matrix
        try:
            import seaborn as sns
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(all_targets, all_preds)
            plt.figure(figsize=(6,6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
            plt.close()
        except ImportError:
            print("[WARN] seaborn not installed — skipping confusion matrix plot.")

        # Predictions vs True scatter plot for six flow rates
        try:
            plt.figure(figsize=(6,6))
            plt.scatter(all_targets, all_preds, alpha=0.7)
            plt.plot([0, num_classes-1], [0, num_classes-1], 'r--')  # y=x line
            plt.xlabel("True Flowrate Class")
            plt.ylabel("Predicted Flowrate Class")
            plt.title("Predictions vs True Flowrates")
            plt.xticks(range(num_classes))
            plt.yticks(range(num_classes))
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "predictions_vs_true.png"))
            plt.close()
        except Exception as e:
            print(f"[WARN] Could not create predictions vs true plot: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blood Flow Classification")
    parser.add_argument("--mode", choices=["train", "evaluate"], required=True)
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint (.pth) for evaluation")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--sequence_len", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--augment_train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
            train(
                data_dir=args.data_dir,
                output_dir=outputs_dir,
                batch_size=args.batch_size,
                sequence_len=args.sequence_len,
                stride=1,
                num_epochs=args.num_epochs,
                lr=1e-3,
                weight_decay=1e-5,
                num_workers=args.num_workers,
                grad_clip=1.0,
                seed=args.seed,
                num_classes=args.num_classes,
                augment_train=args.augment_train,
                safe_mode=safe_mode,
            )
        elif args.mode == "evaluate":
            if not args.checkpoint:
                raise ValueError("--checkpoint is required for evaluation")
            evaluate_model(
                checkpoint_path=args.checkpoint,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                sequence_len=args.sequence_len,
                device=args.device,
                output_dir=outputs_dir,
                num_workers=args.num_workers,
                safe_mode=safe_mode,
                num_classes=args.num_classes,
            )
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()