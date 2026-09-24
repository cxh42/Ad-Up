"""GT gate: is an HQ clip good enough to be GT at a given size? Exposure, texture and effective resolution.

"4K" sources are often much softer than their pixel count (shallow depth of field, upscaled or heavily compressed
uploads): on UltraVideo only ~30% of frames lose real detail in a x2 down-up round trip at native 4K. The gate crops
the clip at the GT aspect ratio, scales it to GT size and measures 3 frames (thresholds: config section `gt`).

As a command it scores every clip of a manifest once, per orientation (16:9, 9:16), so the director only uses clips
that pass. It also measures the clip's own motion inside the GT crop (src_shake, src_pan; same units as
adup.analysis.ugc_look), so the director only adds the handheld shake that is missing relative to real ads.
make_pairs applies the same gate per shot when a spec was not pre-gated.

Usage (from the repo root):
  .venv-iqa/bin/python -m adup.hq.gate --manifest data/hq/ultravideo_4k/manifest.csv --gt-short 1440 \
      --out data/hq/ultravideo_4k/gate_1440.csv
"""

import argparse
import os
import random

import cv2
import numpy as np
import pandas as pd

from adup.analysis.ugc_look import motion
from adup.config import add_config_args, load_config
from adup.media import detail, gt_geometry, max_crop, probe, read_frames, stream_frames

ORIENTATIONS = ["16:9", "9:16"]


def downup_psnr(gray, tile):
    """x2 down-up PSNR on the most detailed tile: low = real detail at this resolution, high = soft / upscaled."""
    y = gray.astype(np.float32)
    h, w = y.shape
    u = cv2.resize(cv2.resize(y, (w // 2, h // 2), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_CUBIC)
    best = max((cv2.Laplacian(y[i:i + tile, j:j + tile], cv2.CV_32F).var(), i, j)
               for i in range(0, h - tile + 1, tile // 2) for j in range(0, w - tile + 1, tile // 2))
    _, i, j = best
    mse = float(np.mean((y[i:i + tile, j:j + tile] - u[i:i + tile, j:j + tile]) ** 2))
    return 10 * np.log10(255 ** 2 / max(mse, 1e-6))


def crop_filter(src, start, n, sw, sh, gw, gh, rng, candidates=7):
    """Crop the largest region of the GT aspect ratio where a middle frame has the most texture, then downscale to GT
    size. Random crops of landscape footage often land on empty or out-of-focus background."""
    cw, ch = max_crop(sw, sh, gw, gh)
    mid = read_frames(src, f"select=eq(n\\,{start + n // 2})", sw, sh, 1)[0]
    gray = cv2.cvtColor(mid, cv2.COLOR_RGB2GRAY)
    if cw < sw:
        y, xs = (sh - ch) // 2 // 2 * 2, sorted({int(v) // 2 * 2 for v in np.linspace(0, sw - cw, candidates)})
        scored = [(detail(cv2.resize(gray[y:y + ch, x:x + cw], (gw, gh), interpolation=cv2.INTER_AREA)), x) for x in xs]
        best = max(scored)[1]
        x = min(max(best + rng.randint(-cw // 20, cw // 20) // 2 * 2, 0), sw - cw)
    else:
        x, ys = 0, sorted({int(v) // 2 * 2 for v in np.linspace(0, sh - ch, candidates)})
        scored = [(detail(cv2.resize(gray[yy:yy + ch, :cw], (gw, gh), interpolation=cv2.INTER_AREA)), yy) for yy in ys]
        y = max(scored)[1]
    return f"crop={cw}:{ch}:{x}:{y},scale={gw}:{gh}:flags=lanczos"


def shot_filter(shot, geom_vf, seq_fps, select=None):
    vf = f"trim=start_frame={shot['start']},setpts=PTS-STARTPTS"
    if seq_fps and abs(seq_fps - shot["src_fps"]) > 0.01:
        vf += f",fps={seq_fps}"
    if select:                                   # pick frames before the (expensive) crop + scale
        vf += ",select=" + "+".join(f"eq(n\\,{k})" for k in select)
    return f"{vf},{geom_vf}"


def gate_stats(shot, geom_vf, gw, gh, seq_fps):
    """Luma, detail and effective resolution of 3 frames of the shot at GT resolution."""
    ks = [shot["frames"] // 4, shot["frames"] // 2, 3 * shot["frames"] // 4]
    frames = list(stream_frames(shot["src"], shot_filter(shot, geom_vf, seq_fps, ks), gw, gh, 3))
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    tile = max(min(gw, gh) // 4, 128)
    return {"mean_luma": float(np.mean([g.mean() for g in grays])), "detail": float(np.median([detail(g) for g in grays])),
            "downup_psnr": float(np.median([downup_psnr(g, tile) for g in grays]))}


def gate_ok(stats, c):
    """c: config section `gt`."""
    return (stats["detail"] >= c["min_detail"] and c["luma_range"][0] <= stats["mean_luma"] <= c["luma_range"][1]
            and stats["downup_psnr"] <= c["max_downup_psnr"])


def score_clip(path, gt_short, scale, frames, c, rng):
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
                     "pass": gate_ok(stats, c)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--gt-short", type=int, default=1440)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--out", required=True)
    add_config_args(ap)
    args = ap.parse_args()
    c = load_config(args.config, args.set)["gt"]
    files = pd.concat([pd.read_csv(m) for m in args.manifest]).file.tolist()
    done = set(pd.read_csv(args.out).file) if os.path.exists(args.out) else set()
    rng = random.Random(0)
    for i, f in enumerate(files):
        if f in done:
            continue
        try:
            rows = score_clip(f, args.gt_short, args.scale, args.frames, c, rng)
        except Exception as e:
            print(f"[{i + 1}/{len(files)}] {f} failed: {e}", flush=True)
            continue
        pd.DataFrame(rows).to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)
        print(f"[{i + 1}/{len(files)}] {os.path.basename(f)} " +
              " ".join(f"{r['aspect']}:{r['downup_psnr']:.1f}{'+' if r['pass'] else '-'}" for r in rows), flush=True)


if __name__ == "__main__":
    main()
