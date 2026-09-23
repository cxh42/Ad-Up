"""Measure low-level degradation statistics of videos, to compare real-world LQ against synthetic LQ.

Usage (from the repo root): .venv-iqa/bin/python -m adup.analysis.degradation_stats <group> <out.csv> <videos...>
"""

import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
from skimage.restoration import estimate_sigma

N_FRAMES = 8


def stream_stats(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "frame=pict_type,key_frame,pkt_size", "-of", "json", path],
                         capture_output=True, text=True).stdout
    frames = json.loads(out).get("frames", [])
    types = [f.get("pict_type") for f in frames]
    keys = [i for i, f in enumerate(frames) if f.get("key_frame") == 1]
    gop = float(np.median(np.diff(keys))) if len(keys) > 1 else float(len(frames))
    return {"n_frames": len(frames), "gop": gop, "b_ratio": types.count("B") / max(len(types), 1)}


def read_frames(path, n=N_FRAMES):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(total * 0.1, total * 0.9, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            frames.append(f)
    # consecutive pairs for duplicate-frame detection
    pairs = []
    for i in np.linspace(total * 0.1, total * 0.9, 30).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok1, a = cap.read()
        ok2, b = cap.read()
        if ok1 and ok2:
            pairs.append((a, b))
    cap.release()
    return frames, pairs


def blockiness(y):
    """Mean |horizontal gradient| on 8-px block boundaries divided by the mean elsewhere (1.0 = no blocking)."""
    g = np.abs(np.diff(y.astype(np.float32), axis=1))
    cols = np.arange(g.shape[1])
    on = g[:, cols % 8 == 7].mean()
    off = g[:, cols % 8 != 7].mean()
    gv = np.abs(np.diff(y.astype(np.float32), axis=0))
    rows = np.arange(gv.shape[0])
    on_v, off_v = gv[rows % 8 == 7].mean(), gv[rows % 8 != 7].mean()
    return float((on / (off + 1e-6) + on_v / (off_v + 1e-6)) / 2)


def overshoot(y):
    """Sharpening-halo proxy: how far pixels next to strong edges exceed the local min/max envelope of a smoothed image."""
    y = y.astype(np.float32)
    edges = cv2.Canny(y.astype(np.uint8), 100, 200) > 0
    if edges.sum() < 100:
        return np.nan
    near = cv2.dilate(edges.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    smooth = cv2.GaussianBlur(y, (0, 0), 2)
    lo = cv2.erode(smooth, np.ones((7, 7), np.uint8))
    hi = cv2.dilate(smooth, np.ones((7, 7), np.uint8))
    excess = np.maximum(y - hi, 0) + np.maximum(lo - y, 0)
    return float(excess[near].mean())


def downup_psnr(rgb, factor):
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (int(w / factor), int(h / factor)), interpolation=cv2.INTER_AREA)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    mse = np.mean((rgb.astype(np.float64) - up) ** 2)
    return 10 * np.log10(255 ** 2 / mse) if mse > 0 else 99.0


def black_bars(f):
    y = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    rows = (y.mean(1) < 16) & (y.std(1) < 4)
    cols = (y.mean(0) < 16) & (y.std(0) < 4)
    return float(rows.mean()), float(cols.mean())


def analyze(path):
    row = stream_stats(path)
    frames, pairs = read_frames(path)
    ys = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    row["height"], row["width"] = frames[0].shape[:2]
    row["blockiness"] = float(np.mean([blockiness(y) for y in ys]))
    row["noise_sigma"] = float(np.mean([estimate_sigma(y.astype(np.float32)) for y in ys]))
    row["overshoot"] = float(np.nanmean([overshoot(y) for y in ys]))
    for fac in (1.5, 2, 3):
        row[f"downup_x{fac}"] = float(np.mean([downup_psnr(f, fac) for f in frames]))
    diffs = [np.abs(a.astype(np.int16) - b).mean() for a, b in pairs]
    row["dup_frame_ratio"] = float(np.mean([d < 0.3 for d in diffs])) if diffs else np.nan
    bars = [black_bars(f) for f in frames]
    row["bar_rows"], row["bar_cols"] = float(np.mean([b[0] for b in bars])), float(np.mean([b[1] for b in bars]))
    return row


def main(group, out_csv, paths):
    rows = []
    for i, p in enumerate(paths):
        try:
            rows.append({"group": group, "file": p, **analyze(p)})
            print(f"[{group} {i + 1}/{len(paths)}] {os.path.basename(p)}")
        except Exception as e:
            print(f"  failed {p}: {type(e).__name__}: {e}")
    pd.DataFrame(rows).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
