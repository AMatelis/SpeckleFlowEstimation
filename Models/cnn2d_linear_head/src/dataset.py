import os
import re
import csv
import math
import shutil
from typing import Optional, List, Tuple, Dict, Union, Iterable
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# Constants and small helpers
# =============================================================================

DEFAULT_TARGET_RESOLUTION = (128, 128)
DEFAULT_OUTPUTS_DIR = "outputs"

def ensure_dir(path: Union[str, Path]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def extract_flowrate_from_filename(fname: str) -> Optional[float]:
    """
    Extract the first numeric token in a filename and return it as float.
    Returns None if no numeric token is found.
    Examples:
        '5ulpermin.avi' -> 5.0
        'flow_12.5_video.avi' -> 12.5
    """
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", os.path.basename(fname))
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None

# =============================================================================
# Video frame reading (with simple in-memory caching)
# =============================================================================

class VideoFrameCache:
    """
    Simple LRU-ish cache for decoded video frames keyed by absolute video path.
    Caches up to `max_videos` videos in memory; each entry stores a numpy array
    (N_frames, H, W). This prevents repeated disk I/O while training.
    """
    def __init__(self, max_videos: int = 8):
        self.max_videos = max_videos
        self._cache: Dict[str, np.ndarray] = {}
        self._access_order: List[str] = []

    def get(self, path: str) -> Optional[np.ndarray]:
        path = os.path.abspath(path)
        if path in self._cache:
            # update access order
            if path in self._access_order:
                self._access_order.remove(path)
            self._access_order.append(path)
            return self._cache[path]
        return None

    def put(self, path: str, frames: np.ndarray) -> None:
        path = os.path.abspath(path)
        if path in self._cache:
            # replace and update order
            self._cache[path] = frames
            if path in self._access_order:
                self._access_order.remove(path)
            self._access_order.append(path)
            return
        # evict if needed
        if len(self._cache) >= self.max_videos:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        self._cache[path] = frames
        self._access_order.append(path)

    def clear(self) -> None:
        self._cache.clear()
        self._access_order.clear()

# single global cache instance used by dataset
_GLOBAL_VIDEO_CACHE = VideoFrameCache(max_videos=8)

def read_video_frames(video_path: str, target_resolution: Tuple[int, int] = DEFAULT_TARGET_RESOLUTION,
                      use_cache: bool = True, force_reload: bool = False) -> List[np.ndarray]:
    """
    Read all frames from a video, convert to grayscale, resize to target_resolution,
    and return a list of float32 numpy arrays (H, W). Uses a global cache by default.
    Raises RuntimeError if the file cannot be opened.
    """
    video_path = os.path.abspath(video_path)
    if use_cache and not force_reload:
        cached = _GLOBAL_VIDEO_CACHE.get(video_path)
        if cached is not None:
            # return a copy to avoid accidental modifications upstream
            return [frame.copy() for frame in cached]

    if not os.path.exists(video_path):
        raise RuntimeError(f"[ERROR] Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"[ERROR] Could not open video: {video_path}")

    frames: List[np.ndarray] = []
    target_h, target_w = target_resolution
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # convert to grayscale if needed
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # resize exactly to the requested resolution
        if (frame.shape[0], frame.shape[1]) != (target_h, target_w):
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        frames.append(frame.astype(np.float32))
    cap.release()

    if use_cache:
        try:
            np_stack = np.stack(frames, axis=0) if frames else np.zeros((0, target_h, target_w), dtype=np.float32)
            _GLOBAL_VIDEO_CACHE.put(video_path, np_stack)
        except Exception:
            # if stack fails for any reason, skip caching
            pass

    return frames

# =============================================================================
# Handcrafted numeric features extracted from a single frame
# =============================================================================

def compute_frame_features(frame: np.ndarray) -> np.ndarray:
    """
    Compute a small set of numeric features from a single grayscale frame.
    Returns a 1D numpy array of floats:
      [mean, std, median, p10, p90, contrast_measure, sobel_mean]
    The features are scale-invariant in the sense that they are safe if input is 0-255.
    """
    if frame.size == 0:
        return np.zeros((7,), dtype=np.float32)

    # safe statistics
    mean = float(np.mean(frame))
    std = float(np.std(frame))
    median = float(np.median(frame))
    p10 = float(np.percentile(frame, 10.0))
    p90 = float(np.percentile(frame, 90.0))

    # simple contrast measure: (p90 - p10) / (mean + eps)
    eps = 1e-8
    contrast = (p90 - p10) / (mean + eps)

    # edge magnitude mean using Sobel
    try:
        sobelx = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx * sobelx + sobely * sobely)
        sobel_mean = float(np.mean(sobel_mag))
    except Exception:
        sobel_mean = 0.0

    return np.asarray([mean, std, median, p10, p90, contrast, sobel_mean], dtype=np.float32)

# =============================================================================
# Dataset
# =============================================================================

class SpeckleLinearRegressionDataset(Dataset):
    """
    PyTorch Dataset for predicting a scalar flowrate from a video frame.

    Parameters
    ----------
    folder : str
        Directory with .avi videos. Filenames should contain a numeric flowrate token.
    stride : int
        Frame sampling stride for creating many samples from long videos.
    normalize : str
        One of 'scale', 'zscore', 'minmax', 'none' controlling frame normalization.
    target_resolution : tuple
        (H, W) resolution frames are resized to.
    feature_mode : str
        'image' returns (image_tensor, target)
        'stats' returns (stats_tensor, target)
        'both' returns ((image_tensor, stats_tensor), target)
    preload : bool
        If True, the dataset will pre-read and cache every video's frames once at init.
        This increases memory usage but speeds up training for small datasets.
    max_cache_videos : int
        Maximum number of videos to keep in the global cache.
    """

    def __init__(
        self,
        folder: str,
        stride: int = 1,
        normalize: str = "zscore",
        target_resolution: Tuple[int, int] = DEFAULT_TARGET_RESOLUTION,
        feature_mode: str = "image",
        preload: bool = False,
        max_cache_videos: int = 8
    ):
        super().__init__()
        self.folder = folder
        self.stride = max(1, int(stride))
        assert normalize in ("scale", "zscore", "minmax", "none"), "normalize must be one of 'scale','zscore','minmax','none'"
        self.normalize = normalize
        self.target_resolution = tuple(map(int, target_resolution))
        assert feature_mode in ("image", "stats", "both"), "feature_mode must be 'image', 'stats', or 'both'"
        self.feature_mode = feature_mode
        self.preload = bool(preload)
        self.video_index: List[Tuple[str, int, float]] = []  # list of (video_path, frame_index, flow)

        if not os.path.isdir(folder):
            raise NotADirectoryError(f"[ERROR] Dataset folder not found: {folder}")

        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".avi")])
        if not files:
            raise RuntimeError(f"[ERROR] No .avi files found in {folder}")

        # adjust cache size
        _GLOBAL_VIDEO_CACHE.max_videos = max(1, int(max_cache_videos))

        # first pass: collect samples and parse flows
        total_frames = 0
        for fn in files:
            path = os.path.join(folder, fn)
            flow = extract_flowrate_from_filename(fn)
            if flow is None:
                print(f"[WARN] Could not parse flow rate from filename: {fn}")
                continue

            try:
                frames = read_video_frames(path, target_resolution=self.target_resolution, use_cache=self.preload)
            except RuntimeError as e:
                print(f"[WARN] Skipping {fn}: {e}")
                continue

            n = len(frames)
            if n == 0:
                print(f"[WARN] No frames in {fn}, skipping.")
                continue

            for i in range(0, n, self.stride):
                self.video_index.append((path, int(i), float(flow)))
            total_frames += n

        if not self.video_index:
            raise RuntimeError(f"[ERROR] No valid samples found in {folder}")

        print(f"[INFO] Prepared {len(self.video_index)} samples from {len(files)} videos. Total frames scanned: {total_frames}")

        # preloading into cache if requested
        if self.preload:
            for path, _, _ in self.video_index:
                if _GLOBAL_VIDEO_CACHE.get(path) is None:
                    frames = read_video_frames(path, target_resolution=self.target_resolution, use_cache=True, force_reload=True)
                    # cached inside function
            print(f"[INFO] Preloaded video frames into cache (max_videos={_GLOBAL_VIDEO_CACHE.max_videos}).")

    def __len__(self) -> int:
        return len(self.video_index)

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize a float32 frame array according to self.normalize.
        Frames are expected in original range 0..255 typically.
        """
        if self.normalize == "none":
            out = frame.astype(np.float32)
        elif self.normalize == "scale":
            out = frame.astype(np.float32) / 255.0
        elif self.normalize == "minmax":
            mn = frame.min()
            mx = frame.max()
            if mx - mn < 1e-6:
                out = np.zeros_like(frame, dtype=np.float32)
            else:
                out = (frame - mn) / (mx - mn)
        elif self.normalize == "zscore":
            mu = frame.mean()
            sd = frame.std()
            if sd < 1e-6:
                out = (frame - mu).astype(np.float32)
            else:
                out = ((frame - mu) / (sd + 1e-8)).astype(np.float32)
        else:
            out = frame.astype(np.float32)
        return out.astype(np.float32)

    def __getitem__(self, idx: int):
        """
        Return depending on feature_mode:
          - 'image' -> (image_tensor, target_tensor)
          - 'stats' -> (stats_tensor, target_tensor)
          - 'both'  -> ((image_tensor, stats_tensor), target_tensor)

        image_tensor shape: (1, H, W) dtype float32
        stats_tensor shape: (F,) dtype float32 where F == 7
        target_tensor shape: (1,) dtype float32
        """
        path, frame_idx, flow = self.video_index[idx]
        frames = None
        try:
            frames = _GLOBAL_VIDEO_CACHE.get(path)
            if frames is None:
                # read and maybe cache
                frames_list = read_video_frames(path, target_resolution=self.target_resolution, use_cache=True)
                if frames_list:
                    frames = np.stack(frames_list, axis=0)
                else:
                    frames = np.zeros((0, self.target_resolution[0], self.target_resolution[1]), dtype=np.float32)
        except Exception as e:
            # on any error reading frames, raise a clear message
            raise RuntimeError(f"[ERROR] Failed reading frames for {path}: {e}")

        if frame_idx >= frames.shape[0]:
            # fallback to last frame of the video
            frame_idx = max(0, frames.shape[0] - 1)

        raw_frame = frames[frame_idx].astype(np.float32) if frames.shape[0] > 0 else np.zeros(self.target_resolution, dtype=np.float32)
        norm_frame = self._normalize_frame(raw_frame)

        # image tensor: (1, H, W)
        image_tensor = torch.from_numpy(norm_frame[None, ...]).float()

        # stats tensor
        stats = compute_frame_features(raw_frame)
        stats_tensor = torch.from_numpy(stats).float()

        target_tensor = torch.tensor([flow], dtype=torch.float32)

        if self.feature_mode == "image":
            return image_tensor, target_tensor
        elif self.feature_mode == "stats":
            return stats_tensor, target_tensor
        else:  # both
            return (image_tensor, stats_tensor), target_tensor

    def plot_sample(self, idx: int, save_path: Optional[str] = None):
        """
        Plot a sample frame and its flowrate. If save_path is provided the plot is saved.
        """
        path, frame_idx, flow = self.video_index[idx]
        frames = read_video_frames(path, target_resolution=self.target_resolution, use_cache=True)
        if frame_idx >= len(frames):
            frame_idx = max(0, len(frames) - 1)
        frame = frames[frame_idx] if frames else np.zeros(self.target_resolution, dtype=np.float32)

        plt.figure(figsize=(4,4))
        plt.imshow(frame, cmap="gray")
        plt.title(f"Flowrate: {flow:.3f}")
        plt.axis("off")
        plt.tight_layout()
        if save_path:
            ensure_dir(os.path.dirname(save_path) or ".")
            plt.savefig(save_path, dpi=200)
            plt.close()
        else:
            plt.show()

# =============================================================================
# DataLoader factory
# =============================================================================

def create_dataloaders(
    folder: str,
    batch_size: int = 8,
    test_split: float = 0.2,
    stride: int = 1,
    target_resolution: Tuple[int,int] = DEFAULT_TARGET_RESOLUTION,
    feature_mode: str = "image",
    normalize: str = "zscore",
    num_workers: int = 0,
    preload: bool = False,
    max_cache_videos: int = 8,
    seed: int = 42,
    outputs_dir: str = DEFAULT_OUTPUTS_DIR
) -> Tuple[DataLoader, DataLoader, SpeckleLinearRegressionDataset]:
    """
    Build train and validation DataLoaders plus return the underlying dataset object.

    Returns (train_loader, val_loader, dataset)

    The returned dataset is the full dataset before splitting. The train/val sets are Subsets.
    """
    ensure_dir(outputs_dir)
    dataset = SpeckleLinearRegressionDataset(
        folder=folder,
        stride=stride,
        normalize=normalize,
        target_resolution=target_resolution,
        feature_mode=feature_mode,
        preload=preload,
        max_cache_videos=max_cache_videos
    )

    # deterministic indices
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=float(test_split), random_state=int(seed), shuffle=True)
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=torch.cuda.is_available())

    print(f"[INFO] create_dataloaders: train_samples={len(train_set)}, val_samples={len(val_set)}, batch_size={batch_size}")

    return train_loader, val_loader, dataset

# =============================================================================
# Utilities to save plots / CSVs for training loops
# =============================================================================

def plot_flowrate_distribution(dataset: SpeckleLinearRegressionDataset, save_path: Optional[str] = None):
    """
    Plot and optionally save the histogram of flowrate distribution in the dataset.
    """
    flows = [dataset.video_index[i][2] for i in range(len(dataset))]
    plt.figure(figsize=(6,4))
    plt.hist(flows, bins=min(40, max(5, int(len(flows) ** 0.5))), edgecolor="black")
    plt.xlabel("Flowrate")
    plt.ylabel("Count")
    plt.title("Flowrate distribution")
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path) or ".")
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()

def save_epoch_predictions(predictions: Iterable[float], targets: Iterable[float], epoch: int,
                           out_dir: str = DEFAULT_OUTPUTS_DIR, prefix: str = "predictions"):
    """
    Save predictions and targets for one epoch to a CSV file.
    CSV columns: epoch, index, target, prediction, residual
    """
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{prefix}_epoch_{epoch:03d}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "index", "target", "prediction", "residual"])
        for i, (t, p) in enumerate(zip(targets, predictions)):
            try:
                targ = float(t)
                pred = float(p)
            except Exception:
                targ = float(np.asarray(t).tolist() if hasattr(t, "tolist") else t)
                pred = float(np.asarray(p).tolist() if hasattr(p, "tolist") else p)
            writer.writerow([epoch, i, targ, pred, pred - targ])
    print(f"[INFO] Saved epoch predictions to: {out_path}")

def save_training_log(log_lines: Iterable[str], out_dir: str = DEFAULT_OUTPUTS_DIR, filename: str = "training_log.txt"):
    """
    Append training log lines to a text file in outputs directory.
    """
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, filename)
    mode = "a"
    with open(out_path, mode, encoding="utf-8") as f:
        for ln in log_lines:
            f.write(str(ln).rstrip() + "\n")
    print(f"[INFO] Appended {len(list(log_lines)) if hasattr(log_lines, '__len__') else 'N'} lines to {out_path}")