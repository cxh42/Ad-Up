"""Flag HQ clips whose frames already carry text (burned-in captions, titles, watermarks), so the director does not
stack its own captions on top of them.

KwaiVIR's clips are Kuaishou uploads and many carry Chinese captions; UltraVideo has occasional subtitles. The check is
EasyOCR's text detector alone (CRAFT, language-independent) on a few frames per clip: a clip has text when detected
boxes cover at least --min-area of the frame on at least two thirds of the sampled frames. Large scene text (signs,
product labels) can count too; that only makes the director more conservative.

Usage (from the repo root): .venv-iqa/bin/python -m adup.sources.source_text data/sources/kwaivir/manifest.csv [--frames 3] \
    [--gate data/sources/kwaivir/gate_1080.csv]
Output: source_text.csv next to the manifest (file, frames, frames_with_text, text_area, has_text); the director reads
it with --source-text.
"""

import argparse
import os

import easyocr
import numpy as np
import pandas as pd
import torch

from adup.media import probe, read_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--min-area", type=float, default=0.003, help="text box area / frame area for a frame to count")
    ap.add_argument("--gate", help="gate CSV (adup.sources.gate): only check clips that pass it")
    args = ap.parse_args()
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
    out = os.path.join(os.path.dirname(args.manifest), "source_text.csv")
    done = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame(columns=["file"])
    rows = done.to_dict("records")
    files = [f for f in pd.read_csv(args.manifest).file if f not in set(done.file)]
    if args.gate:
        g = pd.read_csv(args.gate)
        files = [f for f in files if f in set(g.file[g["pass"]])]
    for i, f in enumerate(files):
        w, h, _, nb = probe(f)
        s = 720 / min(w, h)
        sw, sh = int(w * s) // 2 * 2, int(h * s) // 2 * 2
        pick = "+".join(f"eq(n\\,{int(k)})" for k in np.linspace(0.1, 0.9, args.frames) * (nb or 100))
        areas = []
        for frame in read_frames(f, f"select='{pick}',scale={sw}:{sh}", sw, sh, args.frames):     # one decoding pass
            boxes, free = reader.detect(frame, canvas_size=1280)
            area = sum((b[1] - b[0]) * (b[3] - b[2]) for b in boxes[0]) + \
                sum(cv_area(p) for p in free[0])
            areas.append(area / (frame.shape[0] * frame.shape[1]))
        n_text = int(sum(a >= args.min_area for a in areas))
        rows.append({"file": f, "frames": args.frames, "frames_with_text": n_text, "text_area": round(float(np.median(areas)), 5),
                     "has_text": n_text * 3 >= args.frames * 2})
        print(f"[{i + 1}/{len(files)}] {os.path.basename(f)} text frames {n_text}/{args.frames}", flush=True)
        if (i + 1) % 20 == 0 or i + 1 == len(files):
            pd.DataFrame(rows).to_csv(out, index=False)
    d = pd.DataFrame(rows)
    print(f"{int(d.has_text.sum())} / {len(d)} clips have text -> {out}")


def cv_area(poly):
    """Area of a free-form (rotated) text box from the detector, given as four corner points."""
    p = np.asarray(poly, np.float32)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


if __name__ == "__main__":
    main()
