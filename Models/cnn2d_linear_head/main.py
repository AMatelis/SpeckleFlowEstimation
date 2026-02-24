import argparse
import os
import sys
import platform
from typing import Optional, List

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure root path for imports
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.train import train
from src.dataloader import create_dataloaders
from models.bloodflow_cnn import BloodFlowCNN2D, plot_predictions_vs_truth


def evaluate_model(
    checkpoint_path: str,
    data_dir: str,
    batch_size: int = 8,
    device: Optional[str] = None,
    output_dir: Optional[str] = None,
    num_workers: int = 0,
    dataset_fraction: float = 1.0,   # added
    safe_mode: bool = True,
    stride: int = 1,
    target_resolution=(128,128)
) -> None:
    """Load a trained model and evaluate on validation data."""
    
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Evaluating on device: {device}")

    # Load dataloaders with the fraction parameter
    train_loader, val_loader, *_ = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        stride=stride,
        num_workers=num_workers,
        target_resolution=target_resolution,
        fraction=dataset_fraction
    )

    if val_loader is None or len(val_loader) == 0:
        raise RuntimeError("Validation data not found or improperly formatted.")

    # Load model
    model = BloodFlowCNN2D(input_channels=1).to(device)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise ValueError("Invalid checkpoint format.")
    
    model.eval()

    all_preds: List[float] = []
    all_targets: List[float] = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device)
            outputs = model(inputs)
            if outputs.shape != targets.shape:
                targets = targets.view_as(outputs)
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    preds = np.array(all_preds)
    targets = np.array(all_targets)
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)

    print("\n[RESULTS]")
    print(f"MSE: {mse:.6f} | MAE: {mae:.6f} | R²: {r2:.6f}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame({"True Flowrate": targets, "Predicted Flowrate": preds})
        csv_path = os.path.join(output_dir, "predictions.csv")
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Predictions saved to {csv_path}")

        # Use updated plotting with 5 uL/min included
        plot_predictions_vs_truth(
            y_true=targets.tolist(),
            y_pred=preds.tolist(),
            save_path=os.path.join(output_dir, "prediction_scatter.png"),
            flow_rates=[0, 5, 50, 100, 150, 400]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blood Flow Regression")
    parser.add_argument("--mode", choices=["train", "evaluate"], required=True)
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint for evaluation")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--augment_train", action="store_true")
    parser.add_argument(
        "--dataset_fraction",
        type=float,
        default=1.0,
        help="Fraction of the dataset to use (0-1)"
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    safe_mode = False
    if platform.system() == "Windows":
        print("[INFO] Windows detected, enabling safe mode for DataLoader (num_workers=0)")
        args.num_workers = 0
        safe_mode = True

    outputs_dir = os.path.join(ROOT_DIR, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    try:
        if args.mode == "train":
            train(
                data_dir=args.data_dir,
                output_dir=outputs_dir,
                batch_size=args.batch_size,
                stride=1,
                num_epochs=args.num_epochs,
                lr=1e-3,
                weight_decay=1e-5,
                num_workers=args.num_workers,
                grad_clip=args.grad_clip,
                seed=args.seed,
                target_resolution=(128, 128),
                plot_every_n_epochs=10,
                dataset_fraction=args.dataset_fraction
            )
        elif args.mode == "evaluate":
            if not args.checkpoint:
                raise ValueError("--checkpoint is required for evaluation")
            evaluate_model(
                checkpoint_path=args.checkpoint,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                device=args.device,
                output_dir=outputs_dir,
                num_workers=args.num_workers,
                safe_mode=safe_mode
            )
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()