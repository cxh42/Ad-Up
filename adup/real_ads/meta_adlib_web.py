"""Download a small sample of video ads from the public Meta Ad Library website (no API token needed).

The first results page embeds the ads' JSON (about 30 ads per query), including `video_hd_url` (the 720p file Meta
serves) and `publisher_platform` (Facebook / Instagram / Audience Network / Messenger / Threads). We load one page
per keyword in headless Chrome and parse that JSON. See meta_adlib.py for the token-based API route.

Usage (from the repo root):  .venv/bin/python -m adup.real_ads.meta_adlib_web [--queries q1 q2 ...] [--per-query N]
Output: data/ads/meta_adlib_web/<date>/videos/<query>/*.mp4, summary.csv
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from urllib.parse import quote

import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from adup.paths import PROXIES, PROXY, REAL_ADS, ROOT
from adup.real_ads.tiktok_topads import UA, video_info

# ---------------------------------------------------------------- config
COUNTRY = "US"
ADS_PER_QUERY = 8
QUERIES = ["skincare", "makeup", "supplement", "dog food", "snack", "dress", "cleaning", "app"]
# ----------------------------------------------------------------

PAGE_URL = ("https://www.facebook.com/ads/library/?active_status=active&ad_type=all&country={country}"
            "&media_type=video&q={q}&search_type=keyword_unordered")
OUT = str((REAL_ADS / "meta_adlib_web" / f"{datetime.now():%Y%m%d}").relative_to(ROOT))


JSON_STR = r'"((?:[^"\\]|\\.)*)"'   # a JSON string body, allowing escaped quotes


def unescape(s):
    return json.loads(f'"{s}"') if s else s


def parse_ads(html):
    """Split the embedded JSON at each ad_archive_id and pull the fields we need from each segment."""
    ads = {}
    parts = html.split('"ad_archive_id":"')[1:]
    for part in parts:
        ad_id = part.split('"', 1)[0]
        hd = re.search(r'"video_hd_url":' + JSON_STR, part)
        if ad_id in ads or not hd:
            continue
        field = lambda k: (m.group(1) if (m := re.search(rf'"{k}":' + JSON_STR, part)) else None)
        platforms = re.search(r'"publisher_platform":\[(.*?)\]', part)
        ads[ad_id] = {
            "ad_id": ad_id,
            "page_name": unescape(field("page_name")),
            "display_format": field("display_format"),
            "publisher_platform": platforms.group(1).replace('"', "") if platforms else None,
            "body": unescape(field("text")),
            "video_hd_url": unescape(hd.group(1)),
        }
    return list(ads.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", nargs="+", default=QUERIES)
    ap.add_argument("--per-query", type=int, default=ADS_PER_QUERY)
    args = ap.parse_args()
    opts = Options()
    for arg in ["--headless=new", f"--proxy-server={PROXY}", "--window-size=1280,2000",
                "--lang=en-US", "--no-sandbox", f"--user-agent={UA}"]:
        opts.add_argument(arg)
    os.environ.setdefault("WDM_LOG", "0")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    rows = []
    try:
        for q in args.queries:
            driver.get(PAGE_URL.format(country=COUNTRY, q=quote(q)))
            time.sleep(10)
            ads = parse_ads(driver.page_source)
            print(f"[{q}] {len(ads)} video ads on page 1, downloading {min(len(ads), args.per_query)}")
            folder = f"{OUT}/videos/{q.replace(' ', '_')}"
            os.makedirs(folder, exist_ok=True)
            for ad in ads[:args.per_query]:
                path = f"{folder}/{ad['ad_id']}.mp4"
                if not os.path.exists(path):
                    try:
                        v = requests.get(ad["video_hd_url"], headers={"User-Agent": UA}, proxies=PROXIES, timeout=120)
                        v.raise_for_status()
                        with open(path, "wb") as f:
                            f.write(v.content)
                    except Exception as e:
                        print(f"  failed {ad['ad_id']}: {e}")
                ok = os.path.exists(path)
                rows.append({"query": q, **{k: v for k, v in ad.items() if k != "video_hd_url"},
                             "aspect": video_info(path) if ok else None, "video_file": path if ok else None,
                             "detail_url": f"https://www.facebook.com/ads/library/?id={ad['ad_id']}"})
            time.sleep(5)
    finally:
        driver.quit()

    summary = pd.DataFrame(rows)
    if os.path.exists(f"{OUT}/summary.csv"):                  # several runs on one day add to the same set
        old = pd.read_csv(f"{OUT}/summary.csv", dtype={"ad_id": str})
        summary = pd.concat([old, summary.astype({"ad_id": str})]).drop_duplicates("ad_id", keep="first")
    summary.to_csv(f"{OUT}/summary.csv", index=False)
    print(f"\nDone: {summary['video_file'].notna().sum()}/{len(summary)} videos -> {OUT}/")


if __name__ == "__main__":
    main()
