"""Download a small sample of TikTok Creative Center "Top Ads" videos, grouped by industry.

The list API needs a `user-sign` header that the page's JS computes; we load the page once in headless
Chrome, reuse its signed headers for direct API calls, and refresh them if the API starts refusing.
Anonymous users only get page 1 (<= 20 ads) per query, so we widen the pool by querying several sort orders.

Usage (from the repo root):  .venv/bin/python -m adup.collect.tiktok_topads
Output: data/ads/tiktok_topads/<date>/videos/<industry>/*.mp4, summary.csv/.xlsx, raw.json
"""

import json
import os
import time
from datetime import datetime

import cv2
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from adup.paths import ADS, PROXIES, PROXY, ROOT

# ---------------------------------------------------------------- config
COUNTRIES = ["US", "GB"]      # first one is also used to load the page
COUNTRY = COUNTRIES[0]
PERIOD = 30                       # 7 / 30 / 180 days
ADS_PER_INDUSTRY = 8              # videos downloaded per industry
ORDER_BYS = ["for_you", "like", "impression", "ctr", "play_6s_rate"]
INDUSTRIES = {                    # top-level industry ids from /top_ads/v2/filters
    "beauty_personal_care": 14000000000,
    "apparel_accessories": 22000000000,
    "health": 29000000000,
    "food_beverage": 27000000000,
    "household_products": 18000000000,
    "pets": 19000000000,
    "ecommerce": 30000000000,
    "apps": 20000000000,
}
# ----------------------------------------------------------------

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36"
PAGE_URL = f"https://ads.tiktok.com/business/creativecenter/inspiration/topads/pc/en?period={PERIOD}&region={COUNTRY}"
API = "https://ads.tiktok.com/creative_radar_api/v1/top_ads"
OUT = str((ADS / "tiktok_topads" / f"{datetime.now():%Y%m%d}").relative_to(ROOT))  # repo-relative paths in the CSV


def get_signed_session():
    """Open the Top Ads page in headless Chrome and return (headers, cookies) of its signed list request."""
    opts = Options()
    for arg in ["--headless=new", f"--proxy-server={PROXY}", "--window-size=1920,1080",
                "--lang=en-US", "--no-sandbox", f"--user-agent={UA}"]:
        opts.add_argument(arg)
    opts.set_capability("goog:loggingPrefs", {"performance": "ALL"})
    os.environ.setdefault("WDM_LOG", "0")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    try:
        driver.get(PAGE_URL)
        for _ in range(30):
            time.sleep(1)
            msgs = [json.loads(e["message"])["message"] for e in driver.get_log("performance")]
            headers = [m["params"]["request"]["headers"] for m in msgs
                       if m["method"] == "Network.requestWillBeSent" and "top_ads/v2/list" in m["params"]["request"]["url"]]
            if headers:
                cookies = {c["name"]: c["value"] for c in driver.get_cookies()}
                return headers[-1], cookies
        raise RuntimeError("Could not capture a signed request (page blocked or layout changed).")
    finally:
        driver.quit()


class Client:
    def __init__(self):
        self.refresh()

    def refresh(self):
        print("Capturing signed headers from the Top Ads page...")
        self.headers, self.cookies = get_signed_session()

    def get(self, path, params):
        for attempt in range(5):
            r = requests.get(f"{API}/{path}", params=params, headers=self.headers, cookies=self.cookies,
                             proxies=PROXIES, timeout=30).json()
            if r.get("code") == 0:
                return r["data"]
            if r.get("code") == 40101 and attempt < 4:   # signature expired
                self.refresh()
                continue
            if r.get("code") == 40100 and attempt < 4:   # rate limited
                print("  rate limited, waiting 60s...")
                time.sleep(60)
                continue
            raise RuntimeError(f"API error {r.get('code')}: {r.get('msg')}")


def video_info(path):
    cap = cv2.VideoCapture(path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    ratio = w / h if h else 0
    return min({"9:16": 9 / 16, "4:5": 4 / 5, "1:1": 1.0, "16:9": 16 / 9}.items(), key=lambda kv: abs(kv[1] - ratio))[0] if h else None


def main():
    client = Client()
    filters = client.get("v2/filters", {})
    industry_names = {f"label_{i['id']}": i["value"] for i in filters["industry"]}
    objective_names = {o["label"]: o["value"] for o in filters["objective"]}

    os.makedirs(OUT, exist_ok=True)
    raw, rows = {}, []
    for label, industry_id in INDUSTRIES.items():
        pool = {}
        for country in COUNTRIES:
            for order_by in ORDER_BYS:
                data = client.get("v2/list", {"period": PERIOD, "page": 1, "limit": 20, "order_by": order_by,
                                              "country_code": country, "industry": industry_id})
                for m in data.get("materials", []):
                    m.setdefault("country", country)
                    pool.setdefault(m["id"], m)
                time.sleep(3)
        raw[label] = list(pool.values())
        picked = sorted(pool.values(), key=lambda m: m.get("like", 0), reverse=True)[:ADS_PER_INDUSTRY]
        print(f"[{label}] pool {len(pool)} unique ads, downloading {len(picked)}")

        folder = f"{OUT}/videos/{label}"
        os.makedirs(folder, exist_ok=True)
        for m in picked:
            vi = m.get("video_info", {})
            urls = vi.get("video_url") or {}
            url = urls.get("720p") or next(iter(urls.values()), None)
            path = f"{folder}/{m['id']}.mp4"
            if url and not os.path.exists(path):
                try:
                    v = requests.get(url, headers={"User-Agent": UA}, proxies=PROXIES, timeout=120)
                    v.raise_for_status()
                    with open(path, "wb") as f:
                        f.write(v.content)
                except Exception as e:
                    print(f"  failed {m['id']}: {e}")
            ok = os.path.exists(path)
            rows.append({
                "industry": label,
                "sub_industry": industry_names.get(m.get("industry_key"), m.get("industry_key")),
                "objective": objective_names.get(m.get("objective_key"), m.get("objective_key")),
                "country": m["country"],
                "ad_id": m["id"],
                "brand_name": m.get("brand_name"),
                "like": m.get("like"),
                "ctr": m.get("ctr"),
                "duration_s": round(vi.get("duration", 0), 1),
                "width": vi.get("width"),
                "height": vi.get("height"),
                "aspect": video_info(path) if ok else None,
                "ad_title": m.get("ad_title"),
                "video_file": path if ok else None,
                "detail_url": f"https://ads.tiktok.com/business/creativecenter/topads/{m['id']}/pc/en?countryCode={m['country']}&period={PERIOD}",
            })

    with open(f"{OUT}/raw.json", "w") as f:
        json.dump(raw, f, indent=1, ensure_ascii=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(f"{OUT}/summary.csv", index=False)
    summary.to_excel(f"{OUT}/summary.xlsx", index=False)
    print(f"\nDone: {summary['video_file'].notna().sum()}/{len(summary)} videos -> {OUT}/")


if __name__ == "__main__":
    main()
