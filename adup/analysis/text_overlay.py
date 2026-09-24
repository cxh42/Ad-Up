"""Detect on-screen text (captions, titles, stickers) in videos with EasyOCR and log every box per sampled frame.

Usage (from the repo root):
  .venv-iqa/bin/python -m adup.analysis.text_overlay <group> <out.csv> <videos...> [--fps 2] [--max-seconds S] [--crops DIR]

One row per detected box (plus one empty row for frames without text), with normalized geometry, so prevalence,
coverage, position and size distributions can be computed later. --crops saves each box crop for visual review.
"""

import argparse
import os

import cv2
import easyocr
import numpy as np
import pandas as pd

MIN_CONF = 0.4


def sample_frames(path, fps, max_seconds=None):
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(int(round(src_fps / fps)), 1)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_seconds and i / src_fps > max_seconds):
            break
        if i % step == 0:
            yield i / src_fps, frame
        i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group")
    ap.add_argument("out")
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--fps", type=float, default=2)
    ap.add_argument("--max-seconds", type=float)
    ap.add_argument("--crops")
    args = ap.parse_args()

    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    rows = []
    for vi, path in enumerate(args.videos):
        vid = os.path.splitext(os.path.basename(path))[0]
        if vid == "gt" or vid.startswith("lq_"):     # synthetic pairs: <clip>/gt.mp4, <clip>/lq_k.mp4
            vid = f"{os.path.basename(os.path.dirname(path))}/{vid}"
        for t, frame in sample_frames(path, args.fps, args.max_seconds):
            h, w = frame.shape[:2]
            boxes = [b for b in reader.readtext(frame) if b[2] >= MIN_CONF and len(b[1].strip()) >= 2]
            if not boxes:
                rows.append({"group": args.group, "video": vid, "t": t, "w": w, "h": h, "text": ""})
            for k, (pts, text, conf) in enumerate(boxes):
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                x0, x1, y0, y1 = max(min(xs), 0), min(max(xs), w), max(min(ys), 0), min(max(ys), h)
                rows.append({"group": args.group, "video": vid, "t": t, "w": w, "h": h, "text": text,
                             "conf": conf, "x0": x0 / w, "x1": x1 / w, "y0": y0 / h, "y1": y1 / h})
                if args.crops and k < 4 and int(t * args.fps) % 4 == 0:
                    os.makedirs(args.crops, exist_ok=True)
                    pad = int((y1 - y0) * 0.3)
                    crop = frame[max(int(y0) - pad, 0):int(y1) + pad, max(int(x0) - pad, 0):int(x1) + pad]
                    cv2.imwrite(os.path.join(args.crops, f"{vid}_{t:06.2f}_{k}.jpg"), crop)
        print(f"[{vi + 1}/{len(args.videos)}] {vid}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)


if __name__ == "__main__":
    main()
