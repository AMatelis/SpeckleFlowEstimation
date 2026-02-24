import argparse
import os
import sys
import platform
from typing import Optional, Tuple

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# ---------------------------------------------------------
# Set root path for imports
# ---------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "models"))
    sys.path.append(os.path.join(ROOT_DIR, "src"))

from src.train import train  # Your updated 3D train function
from src.dataloader import create_3d_dataloaders  # Robust 3D dataloader
from models.bloodflow_cnn import BloodFlowCNN3D  # 3D regression model

# =========================================================
# Evaluation Function
# =========================================================

def evaluate_model(
    checkpoint_path: str,
    batch_size: int = 4,
    device: Optional[str] = None,
    output_dir: Optional[str] = None,
    num_workers: int = 0,
    stride: int = 1,
    frame_depth: int = 16,
    target_resolution: Tuple[int, int] = (128, 128),
    channel_width_um: float = 100.0,
    max_flow: float = 400.0,
) -> None:
    data_dir = r"C:\Users\andre\Downloads\SpeckleModelProject\SpeckleModelCodeVer6\data"
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Evaluating on device: {device}")

    # Load dataloaders
    train_loader, val_loader = create_3d_dataloaders(
        data_folder=data_dir,
        batch_size=batch_size,
        stride=stride,
        frame_depth=frame_depth,
        target_resolution=target_resolution,
        channel_width_um=channel_width_um,
        num_workers=num_workers,
    )

    if len(val_loader) == 0:
        raise RuntimeError("[ERROR] No validation data found or improperly formatted.")

    # Load model
    model = BloodFlowCNN3D(input_channels=1, output_channels=1).to(device)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    all_preds, all_targets = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).squeeze(1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(yb.cpu().tolist())

    preds = np.array(all_preds) * max_flow
    targets = np.array(all_targets) * max_flow
    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))
    r2 = r2_score(targets, preds)

    print("\n[RESULTS]")
    print(f"MSE: {mse:.6f} | MAE: {mae:.6f} | R²: {r2:.6f}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame({"True_Flowrate": targets, "Predicted_Flowrate": preds})
        df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

        # Scatter plot with fixed flow rates
        flow_rates = [0, 5, 50, 100, 150, 400]
        plt.figure(figsize=(6, 6))
        plt.scatter(targets, preds, alpha=0.6, s=18)
        plt.plot([0, max_flow], [0, max_flow], "r--", lw=2, label="Perfect")
        plt.xticks(flow_rates)
        plt.yticks(flow_rates)
        plt.xlim(0, max_flow)
        plt.ylim(0, max_flow)
        plt.xlabel("True Flowrate (μL/min)")
        plt.ylabel("Predicted Flowrate (μL/min)")
        plt.title(f"Predicted vs True Flowrate (R²={r2:.3f})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "prediction_scatter.png"))
        plt.close()


# =========================================================
# CLI Argument Parser
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D Blood Flow Regression")
    parser.add_argument("--mode", choices=["train", "evaluate"], required=True)
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint for evaluation")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--frame_depth", type=int, default=16)
    parser.add_argument("--target_resolution", type=int, nargs=2, default=(128, 128))
    parser.add_argument("--channel_width_um", type=float, default=100.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# =========================================================
# Main Entry Point
# =========================================================

def main() -> None:
    import platform
    args = parse_args()
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Windows safe mode
    if platform.system() == "Windows":
        print("[INFO] Windows detected — enabling safe mode (num_workers=0)")
        args.num_workers = 0

    # Select device
    device = torch.device(args.device if getattr(args, "device", None) else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Running on device: {device}")

    # Ensure outputs directory exists
    outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    # Hard-coded data directory
    data_dir = r"C:\Users\andre\Downloads\Projects\SpeckleModelProject\Models\SpeckleModelCodeVer6\data"

    if args.mode == "train":
        # Run training
        train(
            data_dir=data_dir,
            output_dir=outputs_dir,
            batch_size=args.batch_size,
            stride=args.stride,
            num_epochs=args.num_epochs,
            grad_clip=args.grad_clip,
            num_workers=args.num_workers,
            seed=args.seed,
            target_resolution=tuple(args.target_resolution),
            channel_width_um=args.channel_width_um,
        )

    elif args.mode == "evaluate":
        # Run evaluation if implemented
        if not getattr(args, "checkpoint", None):
            raise ValueError("--checkpoint is required for evaluation")
        evaluate_model(
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
            device=device,
            output_dir=outputs_dir,
            num_workers=args.num_workers,
            stride=args.stride,
            frame_depth=args.frame_depth,
            target_resolution=tuple(args.target_resolution),
            channel_width_um=args.channel_width_um,
        )


if __name__ == "__main__":
    main()