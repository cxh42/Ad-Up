"""Compare metric distributions of a synthetic LQ group against the real LQ group (calibration report).

Usage (from the repo root): .venv-iqa/bin/python -m adup.analysis.compare <real_group> <synth_group> [csv ...]
Reads outputs/analysis/degradation.csv and outputs/analysis/quality.csv by default and prints, per metric, the real vs synthetic
quartiles and a normalized 1-D Wasserstein distance (in units of the real group's IQR; < ~0.3 is a close match).
"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from adup.paths import ANALYSIS

METRICS = ["bpp", "bitrate_kbps", "blockiness", "noise_sigma", "overshoot", "downup_x1.5", "downup_x2", "downup_x3",
           "dover", "dover_tech", "clipiqa", "musiq"]


def main(real, synth, csvs):
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    # one row per (group, file) with the columns from both reports merged
    df = df.groupby(["group", "file"], as_index=False).first()
    r, s = df[df.group == real], df[df.group == synth]
    print(f"real={real} (n={len(r)})  synth={synth} (n={len(s)})")
    print(f"{'metric':14s} {'real q25/q50/q75':>26s} {'synth q25/q50/q75':>26s} {'W/IQR':>7s}")
    for m in METRICS:
        if m not in df or r[m].dropna().empty or s[m].dropna().empty:
            continue
        a, b = r[m].dropna(), s[m].dropna()
        iqr = max(a.quantile(0.75) - a.quantile(0.25), 1e-6)
        w = wasserstein_distance(a, b) / iqr
        fmt = lambda x: "/".join(f"{v:.3g}" for v in x.quantile([0.25, 0.5, 0.75]))
        flag = "  <-- off" if w > 0.5 else ""
        print(f"{m:14s} {fmt(a):>26s} {fmt(b):>26s} {w:7.2f}{flag}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:] or [str(ANALYSIS / "degradation.csv"), str(ANALYSIS / "quality.csv")])
