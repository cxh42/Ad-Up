"""Batch-download a small sample of UGC-style video ads from the Meta Ad Library using AdDownloader.

Usage (from the repo root):
    export META_TOKEN=...            # or put META_TOKEN=... in .env at the repo root
    .venv/bin/python -m adup.collect.meta_adlib

AdDownloader always writes to ./output/, so this script runs inside data/ads/meta_adlib/.
Output: data/ads/meta_adlib/output/<project>/ads_videos/*.mp4 per query, plus output/ugc_summary.csv / .xlsx
"""

import os
import json
from datetime import datetime, timedelta

import cv2
import pandas as pd
import requests

from AdDownloader.adlib_api import AdLibAPI
from AdDownloader.helpers import transform_data
from AdDownloader import media_download as md

from adup.paths import ADS, PROXY, ROOT

# ---------------------------------------------------------------- config
# Non-political ads are only exposed by the API for ads delivered in the EU; IE gives mostly English creatives.
COUNTRY = "IE"
DAYS_BACK = 90
ADS_PER_QUERY = 8          # videos to download per query
FETCH_PER_QUERY = 100      # ads metadata fetched per query (sampled down to ADS_PER_QUERY)
HEADLESS = True

# (label, search phrase) - phrases typical of UGC / creator-style ad copy
QUERIES = [
    ("honest_review", "honest review"),
    ("i_tried", "I tried"),
    ("game_changer", "game changer"),
    ("made_me_buy", "made me buy"),
    ("obsessed", "obsessed with"),
    ("my_routine", "my routine"),
]
# ----------------------------------------------------------------

for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ[k] = PROXY
md.chrome_opts.add_argument(f"--proxy-server={PROXY}")
if HEADLESS:
    md.chrome_opts.add_argument("--headless=new")


def load_token():
    token = os.environ.get("META_TOKEN")
    env_file = ROOT / ".env"
    if not token and env_file.exists():
        for line in open(env_file):
            if line.startswith("META_TOKEN="):
                token = line.split("=", 1)[1].strip()
    if not token:
        raise SystemExit("META_TOKEN not set (env var or .env file).")
    return token


def fetch_one_page(api, params):
    """Fetch a single page of results (AdLibAPI.fetch_data follows every page, which is far more than we need)."""
    data = requests.get(api.base_url, params=params, timeout=60).json()
    if "error" in data:
        print(f"  API error: {data['error'].get('message')}")
        return False
    if not data.get("data"):
        print("  no ads found")
        return False
    folder = f"output/{api.project_name}/json"
    os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/1.json", "w") as f:
        json.dump(data, f, indent=2)
    return True


def video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    ratio = w / h if h else 0
    aspect = min({"9:16": 9 / 16, "4:5": 4 / 5, "1:1": 1.0, "16:9": 16 / 9}.items(), key=lambda kv: abs(kv[1] - ratio))[0]
    return {"width": w, "height": h, "aspect": aspect, "duration_s": round(frames / fps, 1) if fps else None}


def main():
    token = load_token()
    work_dir = ADS / "meta_adlib"
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    date_min = (datetime.today() - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")
    stamp = datetime.now().strftime("%Y%m%d")
    rows = []

    for label, phrase in QUERIES:
        project = f"ugc_{stamp}_{label}"
        print(f"\n=== {label}: \"{phrase}\" ===")
        api = AdLibAPI(token, project_name=project)
        api.add_parameters(
            search_terms=phrase,
            ad_reached_countries=COUNTRY,
            ad_delivery_date_min=date_min,
            search_type="KEYWORD_EXACT_PHRASE",
            media_type="VIDEO",
            ad_active_status="ACTIVE",
            limit=str(FETCH_PER_QUERY),
        )
        if not fetch_one_page(api, api.request_parameters):
            continue
        df = transform_data(project, country=COUNTRY, ad_type="ALL")
        if df is None or df.empty:
            continue
        # prefer ads with the largest reach, then download media for the top ones
        if "eu_total_reach" in df:
            df = df.sort_values("eu_total_reach", ascending=False)
        sample = df.head(ADS_PER_QUERY).reset_index(drop=True)
        md.start_media_download(project, nr_ads=len(sample), data=sample, random_state=0)

        for _, ad in sample.iterrows():
            vid = f"output/{project}/ads_videos/ad_{ad['id']}_video.mp4"
            start = pd.to_datetime(ad.get("ad_delivery_start_time"), errors="coerce")
            rows.append({
                "query": label,
                "ad_id": ad["id"],
                "page_name": ad.get("page_name"),
                "start_date": start.date() if pd.notna(start) else None,
                "days_running": (datetime.today() - start).days if pd.notna(start) else None,
                "eu_total_reach": ad.get("eu_total_reach"),
                "platforms": ", ".join(ad["publisher_platforms"]) if isinstance(ad.get("publisher_platforms"), list) else None,
                "languages": ", ".join(ad["languages"]) if isinstance(ad.get("languages"), list) else None,
                "body": (ad["ad_creative_bodies"][0] if isinstance(ad.get("ad_creative_bodies"), list) and ad["ad_creative_bodies"] else ""),
                "title": (ad["ad_creative_link_titles"][0] if isinstance(ad.get("ad_creative_link_titles"), list) and ad["ad_creative_link_titles"] else ""),
                "video_file": vid if os.path.exists(vid) else None,
                **(video_info(vid) if os.path.exists(vid) else {}),
                "snapshot_url": f"https://www.facebook.com/ads/library/?id={ad['id']}",
            })

    if not rows:
        print("No ads collected.")
        return
    summary = pd.DataFrame(rows)
    summary.to_csv("output/ugc_summary.csv", index=False)
    summary.to_excel("output/ugc_summary.xlsx", index=False)
    print(f"\nSaved {len(summary)} ads ({summary['video_file'].notna().sum()} videos) -> output/ugc_summary.csv")


if __name__ == "__main__":
    main()
