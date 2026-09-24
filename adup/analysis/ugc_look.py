"""Measure what makes a video look like UGC, per shot: camera motion, depth of field, colour, faces, layout.

Used to compare real ads (the target look) against our GT (UltraVideo after UGC-ification), so every UGC transform
can be calibrated like the degradations are. All measurements run at a common 720p short side.

  motion   global camera motion from tracked features (partial affine, RANSAC), with burned-in text / stickers
           masked out (edges that stay still over the shot). Trajectory split into pan (0.5 s moving average) and
           shake (residual): shake_rms in % of frame width, shake_rot_rms in degrees, pan_speed in % width / s.
  dof      sharp_frac: share of 32 px blocks whose Laplacian energy is >= 25% of the frame's 95th-percentile block;
           cinematic shallow depth of field gives low values, phone footage high values.
  colour   saturation, colourfulness (Hasler-Suesstrunk), luma mean / p5 / p95, warmth (R - B).
  faces    YuNet: face present, largest face area fraction, its centre height.
  layout   fit_blur (blurred or solid bands above and below a sharp middle band), split (persistent horizontal divider).

Usage (from the repo root):
  .venv-iqa/bin/python -m adup.analysis.ugc_look <group> <out.csv> <videos...> [--single-shot]
"""

import argparse
import os
import subprocess

import cv2
import numpy as np
import pandas as pd

from adup.analysis.shots import detect_cuts
from adup.paths import FACE_MODEL

SHORT = 720


def read_frames(path, start, n):
    vf = (f"select=between(n\\,{start}\\,{start + n - 1}),"
          f"scale='if(gt(iw,ih),-2,{SHORT})':'if(gt(iw,ih),{SHORT},-2)'")
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip().split(",")
    w, h = int(probe[0]), int(probe[1])
    ow, oh = (int(round(w * SHORT / h / 2)) * 2, SHORT) if w > h else (SHORT, int(round(h * SHORT / w / 2)) * 2)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", vf, "-fps_mode", "passthrough", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, oh, ow, 3)


def overlay_mask(grays):
    """Static high-contrast pixels over the shot (burned-in text, stickers, logos), dilated."""
    g = np.stack(grays).astype(np.float32)
    static = g.std(0) < 1.5
    edges = cv2.Canny(grays[len(grays) // 2], 60, 150) > 0
    m = (static & cv2.dilate(edges.astype(np.uint8), np.ones((3, 3))).astype(bool)).astype(np.uint8)
    return cv2.dilate(m, np.ones((15, 15))) > 0


def motion(grays, mask, fps):
    small = [cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA) for g in grays]
    feat_mask = (~cv2.resize(mask.astype(np.uint8), small[0].shape[::-1]).astype(bool)).astype(np.uint8) * 255
    w = small[0].shape[1]
    steps = []
    for a, b in zip(small[:-1], small[1:]):
        pts = cv2.goodFeaturesToTrack(a, 400, 0.01, 8, mask=feat_mask)
        if pts is None or len(pts) < 20:
            steps.append((0.0, 0.0, 0.0, 1.0))
            continue
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None)
        ok = st.ravel() == 1
        if ok.sum() < 20:
            steps.append((0.0, 0.0, 0.0, 1.0))
            continue
        M, _ = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=1.0)
        if M is None:
            steps.append((0.0, 0.0, 0.0, 1.0))
            continue
        steps.append((M[0, 2] / w * 100, M[1, 2] / w * 100, np.degrees(np.arctan2(M[1, 0], M[0, 0])),
                      float(np.hypot(M[0, 0], M[1, 0]))))
    s = np.array(steps)
    if len(s) < 8:
        return {}
    traj = np.cumsum(s[:, :3], axis=0)
    k = max(int(round(fps / 2)), 3)
    smooth = np.stack([np.convolve(np.pad(traj[:, i], (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, "valid")
                       for i in range(3)], 1)
    shake = traj - smooth
    vel = np.diff(smooth[:, :2], axis=0) * fps
    return {"shake_rms": float(np.sqrt((shake[:, :2] ** 2).sum(1).mean())),
            "shake_rot_rms": float(np.sqrt((shake[:, 2] ** 2).mean())),
            "pan_speed": float(np.hypot(vel[:, 0], vel[:, 1]).mean()) if len(vel) else 0.0,
            "zoom_rate": float(np.abs(np.log(s[:, 3])).mean() * fps * 100)}


def sharp_frac(gray, mask):
    lap = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F))
    h, w = gray.shape
    e = [lap[i:i + 32, j:j + 32].mean() for i in range(0, h - 31, 32) for j in range(0, w - 31, 32)
         if mask[i:i + 32, j:j + 32].mean() < 0.2]
    if not e:
        return np.nan
    e = np.array(e)
    return float((e >= 0.25 * np.percentile(e, 95)).mean())


def colour(rgb, mask):
    keep = ~mask
    px = rgb[keep].astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[keep]
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return {"saturation": float(hsv[:, 1].mean()), "colorfulness": float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())),
            "luma_mean": float(y.mean()), "luma_p5": float(np.percentile(y, 5)), "luma_p95": float(np.percentile(y, 95)),
            "warmth": float((r - b).mean())}


def faces(det, rgb):
    h, w = rgb.shape[:2]
    s = 640 / max(h, w)
    img = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (int(w * s), int(h * s)))
    det.setInputSize((img.shape[1], img.shape[0]))
    _, f = det.detect(img)
    if f is None:
        return {"face": 0, "face_area": 0.0, "face_cy": np.nan}
    area = f[:, 2] * f[:, 3] / (img.shape[0] * img.shape[1])
    i = int(area.argmax())
    return {"face": 1, "face_area": float(area[i]), "face_cy": float((f[i, 1] + f[i, 3] / 2) / img.shape[0])}


def layout(gray):
    h = gray.shape[0]
    lap = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F))
    top, mid, bot = lap[: h // 6].mean(), lap[2 * h // 6: 4 * h // 6].mean(), lap[5 * h // 6:].mean()
    fit_blur = gray.shape[0] > gray.shape[1] and max(top, bot) < 0.15 * mid
    rows = np.abs(np.diff(gray.astype(np.float32), axis=0)).mean(1)
    c = rows[int(h * 0.4): int(h * 0.6)]
    split = c.max() > 6 * np.median(rows) and c.max() > 20
    return {"fit_blur": int(fit_blur), "split_candidate": int(split)}


def analyse(path, single_shot, det, max_shots=6, max_len=60):
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                            "stream=r_frame_rate,nb_read_frames", "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    fr, n = probe.strip().split(",")
    fps, n = eval(fr), int(n)
    cuts = [] if single_shot else detect_cuts(path)
    bounds = list(zip([0, *cuts], [*cuts, n]))
    bounds = [b for b in bounds if b[1] - b[0] >= 12]
    idx = np.linspace(0, len(bounds) - 1, min(max_shots, len(bounds))).round().astype(int) if bounds else []
    rows = []
    for si in sorted(set(idx)):
        a, b = bounds[si]
        a2 = a + 2                                             # skip possible transition frames at the start
        frames = read_frames(path, a2, min(b - a2 - 2, max_len))
        if len(frames) < 10:
            continue
        grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
        mask = overlay_mask(grays)
        mid = len(frames) // 2
        row = {"video": os.path.splitext(os.path.basename(path))[0], "shot": int(si), "shot_frames": b - a,
               "orientation": "portrait" if frames.shape[1] > frames.shape[2] else "landscape",
               "static_frac": float(mask.mean()), **motion(grays, mask, fps), "sharp_frac": sharp_frac(grays[mid], mask),
               **colour(frames[mid], mask), **faces(det, frames[mid]), **layout(grays[mid])}
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group")
    ap.add_argument("out")
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--single-shot", action="store_true", help="videos are single shots (skip cut detection)")
    args = ap.parse_args()
    det = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), 0.7)
    rows = []
    for k, v in enumerate(args.videos):
        try:
            rows += [{"group": args.group, **r} for r in analyse(v, args.single_shot, det)]
        except Exception as e:
            print(f"{v}: {e}", flush=True)
        if (k + 1) % 10 == 0:
            print(f"[{k + 1}/{len(args.videos)}]", flush=True)
    pd.DataFrame(rows).to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)


if __name__ == "__main__":
    main()
