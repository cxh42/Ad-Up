"""Fetch a small, human-centric / UGC-like subset of UltraVideo 4K clips without downloading the 1.4 TB zips.

UltraVideo license: CC-BY-4.0 + NON-COMMERCIAL RESEARCH ONLY (source videos are from YouTube). Keep this data out of
any commercial training run.

Two kinds of clips: "human" (people on screen: talking head, hands with product, food, pets, fashion, lifestyle) and
"product" (product showcase shots without the person requirement: cosmetics, textures, jewelry, electronics, ...).

Usage (from the repo root):
  .venv-iqa/bin/python -m adup.sources.ultravideo [per_category] [--per-source N] [--res 4k|8k] [--kind human|product|both]
Output: data/hq/ultravideo[_8k]/<category>/<clip_id>.mp4 and manifest.csv (the manifest lists the current selection). 8K is needed for 9:16 portrait GT at 2K or above (a 4K frame only gives 1215x2160).
"""

import argparse
import json
import os

import pandas as pd
from remotezip import RemoteZip

from adup.paths import HQ, PROXIES, ROOT

REPO = "https://huggingface.co/datasets/APRIL-AIGC/UltraVideo/resolve/main"
OUT = str((HQ / "ultravideo").relative_to(ROOT))  # repo-relative paths in the manifest; also holds the zip index
N_ZIPS = 36

PERSON = r"\b(woman|man|girl|boy|person|lady|guy|child|couple|hands?|she|he)\b"
EXCLUDE = r"aerial|drone|bird's-eye|skyline|cityscape|mountain range|timelapse|time-lapse|underwater|animated|animation|cartoon|cgi|video game|3d render|concert|stadium|crowd"
PRODUCTS = {     # matched on the brief description; the detailed description must also read like a showcase shot
    "p_cosmetics": r"lipstick|mascara|eyeshadow|palette|nail polish|perfume|fragrance|serum|moisturi[sz]er|face cream|skincare|skin care|cosmetic|makeup brush|foundation|lip gloss",
    "p_texture": r"texture of|swatch|cream (being )?(spread|applied|dripping)|droplet|dropper|lotion",
    "p_jewelry": r"necklace|bracelet|earring|\bring\b|jewel|wristwatch|\bwatch\b",
    "p_fashion": r"sneaker|shoe|handbag|purse|sunglasses|\bbag\b",
    "p_electronics": r"smartphone|headphone|earbud|laptop|camera lens|gadget|keyboard",
    "p_packaging": r"bottle|jar|packag|\bbox\b|\bcan of\b|tube of",
    "p_food": r"chocolate|snack|cereal|sauce|beverage|soda|juice",
}
SHOWCASE = r"close-up|macro|product|display|rotat|turntable|studio|on a (table|surface|counter|pedestal)|placed|arranged|showcas"
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


def select(meta, per_category, per_source=1, res="4k", kind="human"):
    text = (meta["Brief Description"].fillna("") + " " + meta["Shot Type"].fillna("") + " " +
            meta["Theme Description"].fillna("")).str.lower()
    detailed = (meta["Brief Description"].fillna("") + " " + meta["Detailed Description"].fillna("")).str.lower()
    brief = meta["Brief Description"].fillna("").str.lower()
    size = (3840, 2160) if res == "4k" else (7680, 4320)
    base = (meta.frame_width == size[0]) & (meta.frame_height == size[1]) & (meta.total_frames >= 60)
    base &= ~text.str.contains(EXCLUDE)
    meta = meta.copy()
    meta["category"] = None
    if kind in ("human", "both"):
        human = base & text.str.contains(PERSON) & text.str.contains("close-up|medium")
        for cat, pat in CATEGORIES.items():
            meta.loc[human & meta.category.isna() & brief.str.contains(pat), "category"] = cat
    if kind in ("product", "both"):
        product = base & detailed.str.contains(SHOWCASE)
        for cat, pat in PRODUCTS.items():
            meta.loc[product & meta.category.isna() & brief.str.contains(pat), "category"] = cat
    meta = meta[meta.category.notna()]
    # a few clips per source YouTube video for diversity, preferring the stronger technical quality score
    meta = meta.sort_values("vtss_score", ascending=False).groupby("url").head(per_source)
    return meta.groupby("category").head(per_category)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("per_category", type=int, nargs="?", default=25)
    ap.add_argument("--per-source", type=int, default=1, help="clips per source YouTube video")
    ap.add_argument("--res", choices=["4k", "8k"], default="4k")
    ap.add_argument("--kind", choices=["human", "product", "both"], default="human")
    args = ap.parse_args()
    out = OUT if args.res == "4k" else f"{OUT}_8k"
    os.makedirs(OUT, exist_ok=True)
    meta_path = f"{OUT}/short.csv"
    if not os.path.exists(meta_path):
        import requests
        with open(meta_path, "wb") as f:
            f.write(requests.get(f"{REPO}/short.csv", proxies=PROXIES, timeout=120).content)
    meta = pd.read_csv(meta_path)
    index = build_index()
    picked = select(meta[meta.clip_id.isin(index)], args.per_category, args.per_source, args.res, args.kind)
    print(picked.category.value_counts())

    by_zip = {}
    for _, r in picked.iterrows():
        by_zip.setdefault(index[r.clip_id][0], []).append(r)
    rows = []
    for zi, clips in sorted(by_zip.items()):
        with RemoteZip(f"{REPO}/clips_short/clips_short_{zi}.zip", proxies=PROXIES) as z:
            for r in clips:
                dst_dir = f"{out}/{r.category}"
                os.makedirs(dst_dir, exist_ok=True)
                dst = f"{dst_dir}/{r.clip_id}"
                if not os.path.exists(dst):
                    with z.open(index[r.clip_id][1]) as src, open(dst, "wb") as f:
                        f.write(src.read())
                rows.append({"file": dst, "category": r.category, "clip_id": r.clip_id, "youtube_id": r.url,
                             "fps": r.fps, "frames": r.total_frames, "brief": r["Brief Description"],
                             "source": "UltraVideo", "license": "CC-BY-4.0 + non-commercial research only"})
        print(f"zip {zi}: {len(clips)} clips downloaded ({len(rows)}/{len(picked)})")
    pd.DataFrame(rows).to_csv(f"{out}/manifest.csv", index=False)


if __name__ == "__main__":
    main()
