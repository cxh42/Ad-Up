"""Score HQ clips for GT use: exposure, texture and effective resolution at the GT size, per orientation.

"4K" sources are often much softer than their pixel count (shallow depth of field, upscaled or heavily compressed
uploads): on UltraVideo only ~30% of frames lose real detail in a x2 down-up round trip at native 4K. This pre-pass
applies the pipeline's GT gate to every clip once, so sequence composition only strings together clips that pass,
instead of rejecting whole multi-shot sequences later.

It also measures the clip's own motion inside the GT crop (src_shake, src_pan; same units as adup.analysis.ugc_look),
so the UGC director only adds the handheld shake that is missing relative to real ads.

Usage (from the repo root):
  .venv-iqa/bin/python -m adup.sources.gate --manifest data/hq/ultravideo/manifest.csv --gt-short 1440 --scale 2 \
      --out data/hq/ultravideo/gate_1440.csv
"""

import argparse
import os
import random

import pandas as pd

import cv2
import numpy as np

from adup.analysis.ugc_look import motion
from adup.degrade.pipeline import (DEFAULT_CONFIG, crop_filter, gate_ok, gate_stats, gt_geometry, max_crop, probe,
                                   stream_frames)

ORIENTATIONS = ["16:9", "9:16"]


def score_clip(path, gt_short, scale, frames, config, rng):
    sw, sh, fps, nb = probe(path)
    shot = {"src": path, "src_fps": fps, "start": 0, "frames": min(frames, nb or frames)}
    rows = []
    for aspect in ORIENTATIONS:
        gw, gh, _, _ = gt_geometry(aspect, gt_short, scale)
        if max_crop(sw, sh, gw, gh)[0] < gw:
            continue                                     # would need upscaling
        vf = crop_filter(path, 0, shot["frames"], sw, sh, gw, gh, rng)
        stats = gate_stats(shot, vf, gw, gh, fps)
        mw, mh = (360, 640) if gh > gw else (640, 360)            # motion at a small size, 2 s of frames
        small = vf.rsplit(",scale=", 1)[0] + f",scale={mw}:{mh}:flags=area"
        grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in stream_frames(path, small, mw, mh, min(60, shot["frames"]))]
        mo = motion(grays, np.zeros((mh, mw), bool), fps) if len(grays) >= 10 else {}
        rows.append({"file": path, "aspect": aspect, "gt_size": f"{gw}x{gh}", **stats,
                     "src_shake": mo.get("shake_rms", np.nan), "src_pan": mo.get("pan_speed", np.nan),
                     "pass": gate_ok(stats, config)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--gt-short", type=int, default=1440)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    files = pd.concat([pd.read_csv(m) for m in args.manifest]).file.tolist()
    done = set(pd.read_csv(args.out).file) if os.path.exists(args.out) else set()
    rng = random.Random(0)
    for i, f in enumerate(files):
        if f in done:
            continue
        try:
            rows = score_clip(f, args.gt_short, args.scale, args.frames, DEFAULT_CONFIG, rng)
        except Exception as e:
            print(f"[{i + 1}/{len(files)}] {f} failed: {e}", flush=True)
            continue
        pd.DataFrame(rows).to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)
        print(f"[{i + 1}/{len(files)}] {os.path.basename(f)} " +
              " ".join(f"{r['aspect']}:{r['downup_psnr']:.1f}{'+' if r['pass'] else '-'}" for r in rows), flush=True)


if __name__ == "__main__":
    main()
