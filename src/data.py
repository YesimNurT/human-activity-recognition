"""Raw IMU loading and normalization for the UCI HAR dataset.

Each sample is a 2.56 s window (128 samples @ 50 Hz) of 9 inertial channels:
body acceleration, body angular velocity and total acceleration, each on x/y/z.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "UCI HAR Dataset"
MODELS_DIR = ROOT / "models"

SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]
LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]


def load_split(split, data_dir=DATA_DIR):
    """Return X (N, 9, 128) float32, y (N,) in 0..5, subjects (N,) for 'train' or 'test'."""
    split_dir = Path(data_dir) / split
    X = np.stack(
        [np.loadtxt(split_dir / "Inertial Signals" / f"{s}_{split}.txt") for s in SIGNALS],
        axis=1,
    ).astype(np.float32)
    y = np.loadtxt(split_dir / f"y_{split}.txt", dtype=np.int64) - 1
    subjects = np.loadtxt(split_dir / f"subject_{split}.txt", dtype=np.int64)
    return X, y, subjects


def fit_normalizer(X):
    """Per-channel mean/std, computed on training windows only."""
    mean = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = (X.std(axis=(0, 2), keepdims=True) + 1e-8).astype(np.float32)
    return mean, std


def normalize(X, mean, std):
    return ((X - mean) / std).astype(np.float32)


def save_normalizer(mean, std, path=MODELS_DIR / "norm.npz"):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=mean, std=std)


def load_normalizer(path=MODELS_DIR / "norm.npz"):
    d = np.load(path)
    return d["mean"], d["std"]
