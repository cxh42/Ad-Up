"""Real low-quality test segments for the human study and the real-ad benchmark tracks.

The advisor asked for real low-resolution ads as a test set ("that 360p blurry kind users actually see"); they have no
ground truth, so methods are compared by people and by no-reference metrics. From the real-ad test split
(data/stats/real_ads/splits.csv) this takes
  - every Meta ad with the 360p (SD) rendition, the low-quality version Meta serves on slow connections; the same
    frames of its 720p (HD) rendition are cut as well, a real higher-quality anchor (both renditions are frame-aligned)
  - TikTok Top Ads at their lowest resolutions (576p, 360p), one per industry
and cuts one 5-second segment from each, where the ad shows text, faces and motion, the things super-resolution most
visibly gets right or wrong (text slides that do not move, black frames and many cuts are avoided). Segments are cut
frame-exact and losslessly (x264 qp 0), so they carry the platform's own compression and nothing else.

Usage (from the repo root): .venv-iqa/bin/python -m adup.real_ads.human_eval_set [--seconds 5] [--tiktok 8]
The same creative often runs under several ad ids; near-identical segments are kept once.
Output data/eval_sets/real_ugc_v1/: inputs/<segment>.mp4, anchors_hd/<segment>.mp4 (Meta only), segments.csv and
inputs.txt (the list for adup.bench.run --list).
"""

import argparse
import os
import subprocess

import cv2
import easyocr
import numpy as np
import pandas as pd
import torch

from adup.media import FF, largest_face, probe, read_frames
from adup.paths import EVAL_SETS, REAL_STATS, ROOT

OUT = EVAL_SETS / "real_ugc_v1"
LOSSLESS = ["-c:v", "libx264", "-preset", "veryfast", "-qp", "0", "-pix_fmt", "yuv420p"]


def frame_features(path, reader, w, h, fps, n):
    """Per-frame luma, motion and cuts (at 90 px wide), and text / face presence at 2 frames per second."""
    sw, sh = 90, max(int(round(90 * h / w / 2)) * 2, 2)
    small = read_frames(path, f"scale={sw}:{sh}:flags=area", sw, sh, n or 10 ** 6).astype(np.float32).mean(-1)
    luma = small.mean((1, 2))
    motion = np.r_[0, np.abs(np.diff(small, axis=0)).mean((1, 2))]
    hist = np.stack([np.histogram(f, 32, (0, 255))[0] / f.size for f in small])
    cut = np.r_[0, 0.5 * np.abs(np.diff(hist, axis=0)).sum(1)] > 0.35
    step = max(int(round(fps / 2)), 1)
    idx = np.arange(0, len(small), step)
    text, face = np.zeros(len(small), bool), np.zeros(len(small), bool)
    sel = "+".join(f"eq(n\\,{i})" for i in idx)
    for i, f in zip(idx, read_frames(path, f"select='{sel}'", w, h, len(idx))):
        boxes, free = reader.detect(f, canvas_size=1280)
        area = sum((b[1] - b[0]) * (b[3] - b[2]) for b in boxes[0]) + sum(cv2.contourArea(np.float32(p)) for p in free[0])
        text[i:i + step] = area / (w * h) >= 0.002
        face[i:i + step] = largest_face(f) is not None
    return {"luma": luma, "motion": motion, "cut": cut, "text": text, "face": face}


def best_window(feat, n_win, fps):
    """Start frame and score of the best n_win-frame window (candidates every half second)."""
    n = len(feat["luma"])
    if n <= n_win:
        return 0, 0.0, {}
    best = (-1e9, 0, {})
    for s in range(0, n - n_win + 1, max(int(fps / 2), 1)):
        sl = slice(s, s + n_win)
        f = {"text": feat["text"][sl].mean(), "face": feat["face"][sl].mean(), "motion": feat["motion"][sl].mean(),
             "black": (feat["luma"][sl] < 20).mean(), "static": (feat["motion"][sl] < 0.3).mean(),
             "cuts": int(feat["cut"][sl][1:].sum())}
        score = (1.0 * f["text"] + 0.6 * f["face"] + 0.4 * min(f["motion"] / 4, 1) - 1.0 * f["black"]
                 - 0.5 * f["static"] - 0.3 * max(f["cuts"] - 1, 0))
        if score > best[0]:
            best = (score, s, f)
    return best[1], best[0], best[2]


def thumbs(path, n):
    """Gray 64-px frames at 10 / 50 / 90% of a segment, to find the same creative under different ad ids."""
    w, h = probe(path)[:2]
    th = max(int(64 * h / w) // 2 * 2, 2)
    sel = "+".join(f"eq(n\\,{int(n * q)})" for q in (0.1, 0.5, 0.9))
    return read_frames(path, f"select='{sel}',scale=64:{th}", 64, th, 3).astype(np.float32).mean(-1)


def cut(src, dst, start, n):
    subprocess.run([*FF, "-y", "-i", src, "-vf", f"trim=start_frame={start}:end_frame={start + n},setpts=PTS-STARTPTS",
                    "-an", *LOSSLESS, dst], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--tiktok", type=int, default=8, help="TikTok ads at 576p or below, one per industry")
    args = ap.parse_args()
    sp = pd.read_csv(REAL_STATS / "splits.csv", dtype={"ad_id": str})
    test = sp[sp.split == "test"]
    cands = [("meta", r.ad_id, r.video_sd_file, r.video_file) for r in test.itertuples() if isinstance(r.video_sd_file, str)]
    tk = test[test.source == "tiktok_topads"]
    cands += [("tiktok", r.ad_id, r.video_file, None) for r in tk.itertuples() if min(probe(ROOT / r.video_file)[:2]) <= 576]
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
    rows = []
    for k, (platform, ad, src, hd) in enumerate(cands):
        w, h, fps, n = probe(ROOT / src)
        n_win = int(round(args.seconds * fps))
        feat = frame_features(str(ROOT / src), reader, w, h, fps, n)
        start, score, f = best_window(feat, n_win, fps)
        rows.append({"platform": platform, "ad_id": ad, "src": src, "hd": hd, "industry": os.path.basename(os.path.dirname(src)),
                     "width": w, "height": h, "fps": round(fps, 3), "start": start, "frames": min(n_win, n),
                     "score": round(score, 3), **{k2: round(float(v), 3) for k2, v in f.items()}})
        print(f"[{k + 1}/{len(cands)}] {platform} {ad} {w}x{h} start {start / fps:.1f}s score {score:.2f} {f}", flush=True)
    d = pd.DataFrame(rows)
    tik = d[d.platform == "tiktok"].sort_values("score", ascending=False).groupby("industry").head(1).head(args.tiktok)
    d = pd.concat([d[d.platform == "meta"], tik])
    d["segment"] = [f"{p}_{a}_{s}" for p, a, s in zip(d.platform, d.ad_id, d.start)]
    for sub in ("inputs", "anchors_hd"):
        os.makedirs(OUT / sub, exist_ok=True)
    for r in d.itertuples():
        cut(str(ROOT / r.src), str(OUT / "inputs" / f"{r.segment}.mp4"), r.start, r.frames)
        if isinstance(r.hd, str):
            cut(str(ROOT / r.hd), str(OUT / "anchors_hd" / f"{r.segment}.mp4"), r.start, r.frames)
    # Meta often runs one creative under several ad ids: keep the first of near-identical segments
    kept, seen = [], []
    for r in d.itertuples():
        t = thumbs(str(OUT / "inputs" / f"{r.segment}.mp4"), r.frames)
        if any(t.shape == u.shape and np.abs(t - u).mean() < 6 for u in seen):
            print(f"duplicate creative, dropped: {r.segment}", flush=True)
            for sub in ("inputs", "anchors_hd"):
                if os.path.exists(OUT / sub / f"{r.segment}.mp4"):
                    os.remove(OUT / sub / f"{r.segment}.mp4")
            continue
        kept.append(r.Index)
        seen.append(t)
    d = d.loc[kept]
    d.to_csv(OUT / "segments.csv", index=False)
    with open(OUT / "inputs.txt", "w") as f:
        f.writelines(os.path.relpath(OUT / "inputs" / f"{s}.mp4", ROOT) + "\n" for s in d.segment)
    print(f"{len(d)} segments ({(d.platform == 'meta').sum()} Meta 360p with 720p anchors, {(d.platform == 'tiktok').sum()} TikTok)"
          f" -> {OUT}; with text {np.mean(d.text > 0.5):.0%}, with faces {np.mean(d.face > 0.5):.0%}")


if __name__ == "__main__":
    main()
