"""Fetch a small, human-centric / UGC-like subset of UltraVideo 4K clips without downloading the 1.4 TB zips.

UltraVideo license: CC-BY-4.0 + NON-COMMERCIAL RESEARCH ONLY (source videos are from YouTube). Keep this data out of
any commercial training run.

Usage (from the repo root): .venv-iqa/bin/python -m adup.sources.ultravideo [per_category]
Output: data/hq/ultravideo/<category>/<clip_id>.mp4 and data/hq/ultravideo/manifest.csv
"""

import json
import os
import re
import sys

import pandas as pd
from remotezip import RemoteZip

from adup.paths import HQ, PROXIES, ROOT

REPO = "https://huggingface.co/datasets/APRIL-AIGC/UltraVideo/resolve/main"
OUT = str((HQ / "ultravideo").relative_to(ROOT))  # repo-relative paths in the manifest
N_ZIPS = 36
PER_CATEGORY = int(sys.argv[1]) if len(sys.argv) > 1 else 25

PERSON = r"\b(woman|man|girl|boy|person|lady|guy|child|couple|hands?|she|he)\b"
EXCLUDE = r"aerial|drone|bird's-eye|skyline|cityscape|mountain range|timelapse|time-lapse|underwater|animated|animation|cartoon|cgi|video game|3d render|concert|stadium|crowd"
CATEGORIES = {   # first match wins, so more specific categories go first
    "beauty": r"makeup|lipstick|mascara|eyeliner|skincare|skin care|face cream|lotion|serum|applying (makeup|cream|lotion|lipstick)|hairstyl|brushing (her|his) hair",
    "food": r"cooking|kitchen|chopping|pouring|eating|drinking|coffee|recipe|baking|meal|dish|plate of",
    "pets": r"\b(dog|cat|puppy|kitten|pet)\b",
    "hands_product": r"holding (a|an|the)|unbox|product|bottle|package|smartphone|phone|laptop|device|close-up of (her|his|a) hands?",
    "talking_head": r"talking|speaking|looking (directly )?(at|into) the camera|addresses|explaining|interview|vlog",
    "fashion": r"\bdress\b|outfit|fashion|posing|trying on|walking down",
    "lifestyle": r"living room|bedroom|sofa|couch|laughing|smiling|friends|family|home",
}


def build_index():
    path = f"{OUT}/zip_index.json"
    if os.path.exists(path):
        return json.load(open(path))
    index = {}
    for i in range(1, N_ZIPS + 1):
        with RemoteZip(f"{REPO}/clips_short/clips_short_{i}.zip", proxies=PROXIES) as z:
            for name in z.namelist():
                if name.endswith(".mp4"):
                    index[os.path.basename(name)] = [i, name]
        print(f"indexed zip {i}/{N_ZIPS} ({len(index)} clips)")
    json.dump(index, open(path, "w"))
    return index


def select(meta):
    text = (meta["Brief Description"].fillna("") + " " + meta["Shot Type"].fillna("") + " " +
            meta["Theme Description"].fillna("")).str.lower()
    ok = (meta.frame_width == 3840) & (meta.frame_height == 2160) & (meta.total_frames >= 60)
    ok &= text.str.contains(PERSON) & ~text.str.contains(EXCLUDE)
    ok &= text.str.contains("close-up|medium")
    meta = meta[ok].copy()
    brief = meta["Brief Description"].str.lower()
    meta["category"] = None
    for cat, pat in CATEGORIES.items():
        meta.loc[meta.category.isna() & brief.str.contains(pat), "category"] = cat
    meta = meta[meta.category.notna()]
    # one clip per source YouTube video for diversity, prefer stronger technical quality score
    meta = meta.sort_values("vtss_score", ascending=False).drop_duplicates("url")
    return meta.groupby("category").head(PER_CATEGORY)


def main():
    os.makedirs(OUT, exist_ok=True)
    meta_path = f"{OUT}/short.csv"
    if not os.path.exists(meta_path):
        import requests
        with open(meta_path, "wb") as f:
            f.write(requests.get(f"{REPO}/short.csv", proxies=PROXIES, timeout=120).content)
    meta = pd.read_csv(meta_path)
    index = build_index()
    picked = select(meta[meta.clip_id.isin(index)])
    print(picked.category.value_counts())

    by_zip = {}
    for _, r in picked.iterrows():
        by_zip.setdefault(index[r.clip_id][0], []).append(r)
    rows = []
    for zi, clips in sorted(by_zip.items()):
        with RemoteZip(f"{REPO}/clips_short/clips_short_{zi}.zip", proxies=PROXIES) as z:
            for r in clips:
                dst_dir = f"{OUT}/{r.category}"
                os.makedirs(dst_dir, exist_ok=True)
                dst = f"{dst_dir}/{r.clip_id}"
                if not os.path.exists(dst):
                    with z.open(index[r.clip_id][1]) as src, open(dst, "wb") as f:
                        f.write(src.read())
                rows.append({"file": dst, "category": r.category, "clip_id": r.clip_id, "youtube_id": r.url,
                             "fps": r.fps, "frames": r.total_frames, "brief": r["Brief Description"],
                             "source": "UltraVideo", "license": "CC-BY-4.0 + non-commercial research only"})
        print(f"zip {zi}: {len(clips)} clips downloaded ({len(rows)}/{len(picked)})")
    pd.DataFrame(rows).to_csv(f"{OUT}/manifest.csv", index=False)


if __name__ == "__main__":
    main()
