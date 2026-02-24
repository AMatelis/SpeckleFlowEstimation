import os
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt

# =========================================================
# Device configuration
# =========================================================
def get_device() -> torch.device:
    """Return the best available device (GPU or CPU)."""
    if torch.cuda.is_available():
        print("[INFO] Using GPU (CUDA)")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("[INFO] Using GPU (Apple MPS)")
        return torch.device("mps")
    else:
        print("[INFO] Using CPU")
        return torch.device("cpu")


# =========================================================
# Utility functions
# =========================================================
def extract_flowrate_from_filename(filename: str) -> Optional[float]:
    """Extract the first numeric value from a filename as flowrate."""
    match = re.search(r"([\d.]+)", filename.lower())
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def load_video_frames(video_path: str, target_resolution: Tuple[int, int]=(128,128)) -> List[np.ndarray]:
    """Load grayscale frames from video, resize to target resolution."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"[ERROR] Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"[ERROR] Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, target_resolution, interpolation=cv2.INTER_AREA)
        frames.append(frame.astype(np.float32))

    cap.release()
    if not frames:
        raise RuntimeError(f"[ERROR] No frames found in: {video_path}")

    return frames


# =========================================================
# Dataset class
# =========================================================
class SpeckleLinearRegressionDataset(Dataset):
    """PyTorch Dataset for 2D linear regression predicting scalar flowrate."""
    def __init__(
        self,
        frames: List[np.ndarray],
        flowrates: List[float],
        normalize: str = "zscore"
    ):
        assert len(frames) == len(flowrates), "Frames and flowrates length mismatch."
        self.frames = frames
        self.flowrates = flowrates
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int):
        frame = self.frames[idx]
        flow = self.flowrates[idx]

        # Normalize frame
        if self.normalize == "scale":
            frame = frame / 255.0
        elif self.normalize == "zscore":
            mu, sd = frame.mean(), frame.std()
            frame = (frame - mu) / (sd + 1e-8)

        frame_tensor = torch.from_numpy(frame[None,...]).float()  # (1,H,W)
        flow_tensor = torch.tensor([flow], dtype=torch.float32)   # scalar target

        return frame_tensor, flow_tensor

    def plot_sample(self, idx: int):
        """Visualize a single frame with its flowrate."""
        frame = self.frames[idx]
        flow = self.flowrates[idx]
        plt.imshow(frame, cmap="gray")
        plt.title(f"Flowrate: {flow:.2f} µL/min")
        plt.axis("off")
        plt.show()


# =========================================================
# Dataset preparation
# =========================================================
def prepare_dataset(
    folder: str,
    stride: int = 1,
    target_resolution: Tuple[int,int]=(128,128)
) -> Tuple[List[np.ndarray], List[float]]:
    """Load frames and flowrates from videos in folder."""
    all_frames, all_flowrates = [], []
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"[ERROR] Dataset folder not found: {folder}")

    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".avi")])
    if not files:
        raise RuntimeError(f"[ERROR] No .avi files found in {folder}")

    for filename in files:
        flow = extract_flowrate_from_filename(filename)
        if flow is None:
            print(f"[WARN] Skipping file with invalid flowrate: {filename}")
            continue

        video_path = os.path.join(folder, filename)
        try:
            frames = load_video_frames(video_path, target_resolution)
        except Exception as e:
            print(f"[ERROR] Failed to load {filename}: {e}")
            continue

        for i in range(0, len(frames), stride):
            all_frames.append(frames[i])
            all_flowrates.append(flow)

    if not all_frames:
        raise RuntimeError(f"[ERROR] No valid frames extracted from {folder}")

    print(f"[INFO] Prepared {len(all_frames)} frames from {len(files)} videos.")
    return all_frames, all_flowrates


# =========================================================
# DataLoader creation
# =========================================================
def create_dataloaders(
    folder: str,
    batch_size: int = 8,
    test_split: float = 0.2,
    stride: int = 1,
    target_resolution: Tuple[int,int]=(128,128),
    num_workers: int = 0,
    seed: int = 42,
    fraction: float = 1.0
) -> Tuple[DataLoader, DataLoader, torch.device]:
    device = get_device()
    frames, flows = prepare_dataset(folder, stride=stride, target_resolution=target_resolution)

    if fraction < 1.0:
        n_samples = int(len(frames) * fraction)
        np.random.seed(seed)
        indices = np.random.choice(len(frames), n_samples, replace=False)
        frames = [frames[i] for i in indices]
        flows = [flows[i] for i in indices]

    X_train, X_val, y_train, y_val = train_test_split(
        frames, flows, test_size=test_split, shuffle=True, random_state=seed
    )

    train_set = SpeckleLinearRegressionDataset(X_train, y_train)
    val_set = SpeckleLinearRegressionDataset(X_val, y_val)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type=="cuda")
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type=="cuda")
    )

    print(f"[INFO] Train samples: {len(train_set)}, Validation samples: {len(val_set)}")
    return train_loader, val_loader, device


# =========================================================
# Visualization
# =========================================================
def plot_flowrate_distribution(flowrates: List[float], save_path: Optional[str] = None):
    """Plot a histogram of flowrates."""
    plt.figure(figsize=(6,4))
    plt.hist(flowrates, bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("Flowrate (µL/min)")
    plt.ylabel("Count")
    plt.title("Flowrate Distribution")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Saved flowrate histogram to {save_path}")
    plt.show()