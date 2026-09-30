"""Manifest of KwaiVIR's high-quality training clips as a GT source: native portrait short-form UGC.

KwaiVIR (NTIRE 2026 short-form UGC video restoration, research use) ships 200 HQ clips (train/synthetic/HQ-synthetic1
and HQ-synthetic2): Kuaishou short videos at 1080x1920, 30 fps, 6 s, HEVC at 17-35 Mbps. They are the only native
portrait UGC footage we have at the GT size, so the director can make 9:16 ads from them without cropping landscape
footage. The clips are moved out of the KwaiVIR download into data/sources/kwaivir/clips/ (the rest of KwaiVIR, its
low-quality wild / validation / test inputs, is an evaluation set in data/eval_sets/KwaiVIR/ and never a GT source).

The category is the Kuaishou label in the file name, mapped to the UltraVideo categories where one fits.

Usage (from the repo root): .venv-iqa/bin/python -m adup.sources.kwaivir
Output: data/sources/kwaivir/manifest.csv; then gate it like any source (adup.sources.gate --gt-short 1080).
"""

import glob
import os

import pandas as pd

from adup.media import probe
from adup.paths import ROOT, SOURCES

CATEGORIES = {"fashion": "fashion", "beauty": "beauty", "appearance": "beauty", "pet": "pets", "food": "food",
              "selfie": "talking_head", "emotion": "talking_head", "hightech": "hands_product", "home": "lifestyle",
              "life": "lifestyle", "parenting": "lifestyle", "fitness": "lifestyle", "travel": "lifestyle"}


def main():
    rows = []
    for f in sorted(glob.glob(str(SOURCES / "kwaivir" / "clips" / "*.mp4"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        label = stem.split("_", 1)[1] if "_" in stem else "other"
        w, h, fps, nb = probe(f)
        rows.append({"file": os.path.relpath(f, ROOT), "category": CATEGORIES.get(label.lower(), label.lower()),
                     "kwai_label": label, "clip_id": os.path.basename(f), "youtube_id": f"kwai_{stem.split('_')[0]}",
                     "fps": fps, "frames": nb, "width": w, "height": h, "brief": "", "source": "KwaiVIR",
                     "license": "NTIRE 2026 KwaiVIR challenge data, research use only"})
    out = SOURCES / "kwaivir"
    out.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(rows)
    d.to_csv(out / "manifest.csv", index=False)
    print(f"{len(d)} clips -> {out / 'manifest.csv'}")
    print(d.kwai_label.value_counts().to_dict())


if __name__ == "__main__":
    main()
