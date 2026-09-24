"""Build one contact sheet per industry (TikTok) or query (Meta): each row = one ad, 4 frames sampled across the video.

Usage (from the repo root): .venv/bin/python -m adup.collect.contact_sheets data/ads/<source>/<date>
"""

import os
import sys

import cv2
import numpy as np
import pandas as pd

FRAME_H = 320
POSITIONS = [0.05, 0.3, 0.6, 0.9]


def frames(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for p in POSITIONS:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * p))
        ok, f = cap.read()
        if ok:
            out.append(cv2.resize(f, (int(f.shape[1] * FRAME_H / f.shape[0]), FRAME_H)))
    cap.release()
    return out


def main(root):
    summary = pd.read_csv(f"{root}/summary.csv", dtype={"ad_id": str})
    os.makedirs(f"{root}/sheets", exist_ok=True)
    key = "industry" if "industry" in summary else "query"
    for industry, group in summary.groupby(key, sort=False):
        rows = []
        for _, ad in group.iterrows():
            if not isinstance(ad["video_file"], str):
                continue
            fs = frames(ad["video_file"])
            if not fs:
                continue
            strip = np.hstack(fs)
            label = np.full((40, strip.shape[1], 3), 255, np.uint8)
            info = f"{ad['duration_s']}s  like={ad['like']}" if "like" in ad else str(ad.get("page_name", ""))
            cv2.putText(label, f"{ad['ad_id']}  {info}", (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            rows.append(np.vstack([label, strip]))
        if not rows:
            continue
        width = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 10), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
        cv2.imwrite(f"{root}/sheets/{industry}.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(f"{root}/sheets/{industry}.jpg ({len(rows)} ads)")


if __name__ == "__main__":
    main(sys.argv[1])
