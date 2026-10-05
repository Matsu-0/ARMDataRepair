#!/usr/bin/env python3
"""
CSDI detect-then-repair for ARM comparison.

CSDI is an imputation model. This script does not use ARM's error mask E.
It (1) trains CSDI on the dirty series with random masking, (2) reconstructs
held-out timesteps and flags high residual points, (3) treats flagged points
as missing and imputes them with CSDI.

Official CSDI sources are imported from baseline/CSDI.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _add_csdi_path(csdi_root: str) -> None:
    root = os.path.abspath(csdi_root)
    if root not in sys.path:
        sys.path.insert(0, root)


class WindowDataset(Dataset):
    def __init__(self, series: np.ndarray, cond_mask: np.ndarray, window: int, starts):
        self.series = series.astype(np.float32)
        self.cond_mask = cond_mask.astype(np.float32)
        self.window = window
        self.starts = list(starts)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        e = s + self.window
        obs = self.series[s:e]
        gm = self.cond_mask[s:e]
        om = np.ones_like(obs)
        return {
            "observed_data": obs,
            "observed_mask": om,
            "gt_mask": gm,
            "timepoints": np.arange(self.window, dtype=np.float32),
            "cut_length": np.array(0, dtype=np.int64),
            "hist_mask": om,
        }


def default_config(epochs: int, batch_size: int, lr: float, num_steps: int) -> dict:
    return {
        "train": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "itr_per_epoch": 80,
        },
        "diffusion": {
            "layers": 2,
            "channels": 32,
            "nheads": 4,
            "diffusion_embedding_dim": 64,
            "beta_start": 0.0001,
            "beta_end": 0.5,
            "num_steps": num_steps,
            "schedule": "quad",
            "is_linear": False,
        },
        "model": {
            "is_unconditional": 0,
            "timeemb": 64,
            "featureemb": 16,
            "target_strategy": "random",
        },
    }


def window_starts(n: int, window: int, stride: int):
    if n <= window:
        return [0]
    starts = list(range(0, n - window + 1, stride))
    last = n - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def pick_device(name: str) -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def impute_series(model, series, cond_mask, window, stride, batch_size, nsample, device):
    """Impute positions where cond_mask==0. Average overlapping windows."""
    n, k = series.shape
    starts = window_starts(n, window, stride)
    acc = np.zeros((n, k), dtype=np.float64)
    wgt = np.zeros((n, k), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for i0 in range(0, len(starts), batch_size):
            batch_starts = starts[i0 : i0 + batch_size]
            ds = WindowDataset(series, cond_mask, window, batch_starts)
            loader = DataLoader(ds, batch_size=len(batch_starts), shuffle=False)
            batch = next(iter(loader))
            samples, _, _, _, _ = model.evaluate(batch, nsample)
            median = samples.median(dim=1).values.permute(0, 2, 1).cpu().numpy()
            for i, s in enumerate(batch_starts):
                acc[s : s + window] += median[i]
                wgt[s : s + window] += 1.0
    filled = acc / np.maximum(wgt, 1e-8)
    out = series.copy()
    missing = cond_mask < 0.5
    out[missing] = filled[missing]
    return out


def detect_then_repair(series, args, device):
    from main_model import CSDI_Physio
    from utils import train

    n, k = series.shape
    window = min(args.window, n)
    stride = max(1, window // 2)
    mean = series.mean(axis=0)
    std = series.std(axis=0)
    std[std < 1e-8] = 1.0
    normed = (series - mean) / std

    rng = np.random.RandomState(args.seed)
    all_starts = window_starts(n, window, stride)
    if len(all_starts) > args.max_train_windows:
        idx = rng.choice(len(all_starts), size=args.max_train_windows, replace=False)
        train_starts = [all_starts[i] for i in sorted(idx)]
    else:
        train_starts = all_starts

    cond_all = np.ones_like(normed)
    train_set = WindowDataset(normed, cond_all, window, train_starts)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    config = default_config(args.epochs, args.batch_size, args.lr, args.num_steps)
    model = CSDI_Physio(config, str(device), target_dim=k).to(device)

    folder = args.workdir
    os.makedirs(folder, exist_ok=True)
    print(f"[CSDI] device={device} n={n} k={k} window={window} train_windows={len(train_starts)}")
    train(model, config["train"], train_loader, valid_loader=None, foldername=folder)

    even_mask = np.ones_like(normed)
    even_mask[0::2] = 0.0
    odd_mask = np.ones_like(normed)
    odd_mask[1::2] = 0.0
    print("[CSDI] detection pass (even/odd reconstruction)")
    hat_even = impute_series(
        model, normed, even_mask, window, stride, args.batch_size, args.nsample, device
    )
    hat_odd = impute_series(
        model, normed, odd_mask, window, stride, args.batch_size, args.nsample, device
    )
    hat = normed.copy()
    hat[0::2] = hat_even[0::2]
    hat[1::2] = hat_odd[1::2]
    residual = np.linalg.norm(hat - normed, axis=1)
    med = float(np.median(residual))
    mad = float(np.median(np.abs(residual - med))) + 1e-8
    tau = med + args.detect_k * 1.4826 * mad
    detected = residual > tau
    n_det = int(detected.sum())
    print(f"[CSDI] residual median={med:.4f} mad={mad:.4f} tau={tau:.4f} detected={n_det}/{n}")

    repaired_norm = normed.copy()
    if n_det > 0:
        repair_mask = np.ones_like(normed)
        repair_mask[detected] = 0.0
        print("[CSDI] repair pass (impute self-detected points)")
        repaired_norm = impute_series(
            model, normed, repair_mask, window, stride, args.batch_size, args.nsample, device
        )
    repaired = repaired_norm * std + mean
    np.savetxt(os.path.join(folder, "detect_mask.csv"), detected.astype(int), fmt="%d")
    return repaired, detected


def load_series(path: str) -> np.ndarray:
    raw = np.loadtxt(path, delimiter=",")
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    return raw.astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description="CSDI self-detect then repair")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csdi_root", default="./baseline/CSDI")
    parser.add_argument("--workdir", default="./models/csdi")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--nsample", type=int, default=3)
    parser.add_argument("--detect_k", type=float, default=3.0)
    parser.add_argument("--max_train_windows", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=665)
    args = parser.parse_args()

    _add_csdi_path(args.csdi_root)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    series = load_series(args.input)
    device = pick_device(args.device)
    repaired, _ = detect_then_repair(series, args, device)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    np.savetxt(args.output, repaired, delimiter=",", fmt="%.10f")
    print(f"[CSDI] wrote {args.output}")


if __name__ == "__main__":
    main()
