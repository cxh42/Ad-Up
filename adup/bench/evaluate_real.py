"""Score benchmark outputs on real ads: no-reference quality (track C) and the Meta SD -> HD "semi-pair" (track D).

  C  every output of a method on a real-ad list: DOVER (technical / aesthetic / overall, as in adup.analysis.quality)
     and MUSIQ / CLIP-IQA on 8 sampled frames. No reference exists; these scores only rank methods and are checked
     against the human study later.
  D  for Meta ads with both renditions: the method's 1080p output of the 360p SD file is area-downsampled to the HD
     size and compared with Meta's own 720p HD file of the same ad (luma PSNR / SSIM, LPIPS on sampled frames). HD is
     compressed too, so it is a near reference: it comes from the same upload at twice the resolution and shows whether
     the method moves towards the real higher-resolution content or invents its own.

Usage (from the repo root):
  .venv-iqa/bin/python -m adup.bench.evaluate_real --runs outputs/bench/real_test --methods bicubic dove \\
      --split test --csv outputs/bench/real_test/results.csv
Outputs are expected as <runs>/<method>/<input file stem>.mp4 (adup.bench.run --list).
"""

import argparse
import os

import numpy as np
import pandas as pd
import pyiqa
import torch

from adup.analysis.quality import dover_scores, load_dover, sample_frames
from adup.bench.metrics import luma, psnr, ssim
from adup.media import probe, stream_frames
from adup.paths import REAL_STATS, ROOT


def semi_pair(out_path, hd_path, lpips, n_max=90, every=6):
    """Output (1080p of the SD input) area-downsampled to the HD size vs the real HD file."""
    hw, hh, _, hn = probe(hd_path)
    ow, oh, _, on = probe(out_path)
    n = min(hn or n_max, on or n_max, n_max)
    ps, ss, lp = [], [], []
    for i, (o, h) in enumerate(zip(stream_frames(out_path, f"scale={hw}:{hh}:flags=area", hw, hh, n),
                                   stream_frames(hd_path, "null", hw, hh, n))):
        yo, yh = luma(o), luma(h)
        ps.append(psnr(yo, yh))
        ss.append(ssim(yo, yh))
        if i % every == 0:
            t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].float().cuda() / 255
            lp.append(float(lpips(t(o), t(h))))
    return {"semi_psnr": np.mean(ps), "semi_ssim": np.mean(ss), "semi_lpips": np.mean(lp)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--split", default="test", help="real-ad split whose SD / HD files define the semi-pairs")
    ap.add_argument("--no-dover", action="store_true")
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    sp = pd.read_csv(REAL_STATS / "splits.csv", dtype={"ad_id": str})
    sp = sp[sp.split == args.split]
    stem = lambda p: os.path.splitext(os.path.basename(p))[0]
    by_stem = {}
    for r in sp.itertuples():
        by_stem[stem(r.video_file)] = ("hd", r)
        if isinstance(r.video_sd_file, str):
            by_stem[stem(r.video_sd_file)] = ("sd", r)
    dover = None if args.no_dover else load_dover()
    musiq, clipiqa = pyiqa.create_metric("musiq", device="cuda"), pyiqa.create_metric("clipiqa", device="cuda")
    lpips = pyiqa.create_metric("lpips", device="cuda")
    rows = []
    for method in args.methods:
        od = os.path.join(args.runs, method)
        for f in sorted(os.listdir(od)) if os.path.isdir(od) else []:
            if not f.endswith(".mp4") or f[:-4] not in by_stem:
                continue
            kind, r = by_stem[f[:-4]]
            path = os.path.join(od, f)
            row = {"method": method, "ad_id": r.ad_id, "source": r.source, "input": kind}
            if dover:
                row["dover_tech"], row["dover_aes"], row["dover"] = dover_scores(path, *dover)
            ts = [torch.from_numpy(x).permute(2, 0, 1).float().div(255)[None].cuda() for x in sample_frames(path)]
            with torch.no_grad():
                row["musiq"] = float(np.mean([musiq(t).item() for t in ts]))
                row["clipiqa"] = float(np.mean([clipiqa(t).item() for t in ts]))
                if kind == "sd":
                    row.update(semi_pair(path, str(ROOT / r.video_file), lpips))
            rows.append(row)
            print(f"{method:12s} {r.ad_id} {kind} " + " ".join(f"{k}={v:.3f}" for k, v in row.items()
                                                              if isinstance(v, float)), flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(args.csv, index=False)
    if len(d):
        cols = [c for c in ["dover", "musiq", "clipiqa", "semi_psnr", "semi_ssim", "semi_lpips"] if c in d]
        print(d.groupby(["method", "input"])[cols].mean().round(4).to_string())


if __name__ == "__main__":
    main()
