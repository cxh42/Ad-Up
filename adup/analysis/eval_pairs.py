"""Full-reference evaluation of restored videos against GT, per frame, split into frames near shot cuts and the rest.

Pairs come from adup.make_pairs (meta.json gives the shot boundaries). A model that mixes shots across a cut
shows up as a PSNR / LPIPS drop in the frames next to the cut.

Usage (from the repo root):
  .venv-iqa/bin/python -m adup.analysis.eval_pairs --pairs data/pairs/cuttest_2k_x2 --pred outputs/runs/dove/x \
      --lq lq_0 [--window 4] [--csv out.csv]
Prediction files are <pred>/<pair_id>_<lq>.mp4.
"""

import argparse
import glob
import json
import os
import subprocess

import numpy as np
import pandas as pd
import pyiqa
import torch


def frames(path, w, h):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 10 * np.log10(255 ** 2 / max(mse, 1e-10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--lq", default="lq_0")
    ap.add_argument("--window", type=int, default=4, help="frames on each side of a cut counted as 'near cut'")
    ap.add_argument("--csv")
    args = ap.parse_args()
    lpips = pyiqa.create_metric("lpips", device="cuda")
    rows = []
    for m in sorted(glob.glob(os.path.join(args.pairs, "*", "meta.json"))):
        meta = json.load(open(m))
        pred = os.path.join(args.pred, f"{meta['id']}_{args.lq}.mp4")
        if not os.path.exists(pred):
            continue
        w, h = meta["gt_size"]
        gt, pr = frames(os.path.join(os.path.dirname(m), "gt.mp4"), w, h), frames(pred, w, h)
        cuts = [s["start_frame"] for s in meta.get("shots", [])[1:]]
        for i in range(min(len(gt), len(pr))):
            d = min([abs(i - c) if i >= c else abs(i - c) - 1 for c in cuts], default=10 ** 6)  # distance to nearest cut
            t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].float().cuda() / 255
            rows.append({"pair": meta["id"], "frame": i, "near_cut": d < args.window, "dist_to_cut": d,
                         "psnr": psnr(gt[i], pr[i]), "lpips": float(lpips(t(pr[i]), t(gt[i])))})
    df = pd.DataFrame(rows)
    if args.csv:
        df.to_csv(args.csv, index=False)
    print(df.groupby("near_cut")[["psnr", "lpips"]].mean().round(4).assign(frames=df.groupby("near_cut").size()))
    print("overall", df[["psnr", "lpips"]].mean().round(4).to_dict())


if __name__ == "__main__":
    main()
