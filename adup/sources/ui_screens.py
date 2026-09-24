"""Synthetic phone-app screens as GT for "screen recording" shots (app / fintech / shopping / games-store ads).

About 15% of the frames in real UGC ads are phone screen recordings, and no video dataset has them at 2K+. Real UI is
crisp vector text and shapes, so we render our own: random app pages (shopping grid, social feed, finance dashboard,
chat, food delivery, fitness) as HTML, screenshotted by headless Chrome at a 360 px wide mobile viewport with a
device scale factor of 4 -> 1440 px wide, full page (several screens tall). Product / feed pictures are Unsplash Lite
photos (commercial-OK licence). adup.ugc.render's "screen" shots scroll through these pages like a screen recording.

Usage (from the repo root): .venv/bin/python -m adup.sources.ui_screens --n 40
Output: data/hq/ui_screens/<id>.png and manifest.csv
"""

import argparse
import base64
import os
import random

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from adup.paths import ANALYSIS, HQ, PROXY

OUT = HQ / "ui_screens"
FONTS = ["-apple-system, 'Helvetica Neue', Arial", "Roboto, Arial", "'Segoe UI', Arial", "Inter, Arial", "Georgia, serif"]
ACCENTS = ["#fe2c55", "#1877f2", "#00b37e", "#ff6b00", "#7b3ff2", "#111111", "#e1306c", "#0a84ff", "#ffb800"]
NAMES = ["Emma", "Liam", "Olivia", "Noah", "Ava", "Mia", "Lucas", "Sofia", "Ethan", "Chloe", "Maya", "Leo"]
WORDS = ("glow serum hydrating daily fresh organic bundle deal limited best seller new arrival premium soft cozy "
         "wireless compact protein vegan bestselling travel pack starter kit refill").split()


def phrase(r, n):
    return " ".join(r.choice(WORDS) for _ in range(n)).capitalize()


def img_tag(r, photos, h):
    if photos is None or not len(photos):
        return f'<div style="height:{h}px;border-radius:12px;background:linear-gradient({r.randint(0, 360)}deg,#ddd,#bbb)"></div>'
    p = photos.iloc[r.randrange(len(photos))]
    return (f'<img src="{p.photo_image_url}?w=1600&q=90&fm=jpg" '
            f'style="width:100%;height:{h}px;object-fit:cover;border-radius:12px;display:block">')


def page(r, kind, photos):
    acc = r.choice(ACCENTS)
    dark = r.random() < 0.3
    bg, fg, card = ("#000", "#fff", "#1c1c1e") if dark else ("#f2f2f7" if r.random() < 0.5 else "#fff", "#111", "#fff")
    head = (f'<div style="display:flex;justify-content:space-between;padding:10px 16px;font-size:13px;font-weight:600">'
            f'<span>{r.randint(1, 12)}:{r.randint(10, 59)}</span><span>5G ▮▮▮ {r.randint(20, 99)}%</span></div>'
            f'<div style="padding:6px 16px 12px;font-size:{r.choice([22, 26, 30])}px;font-weight:800">{phrase(r, 2)}</div>')
    body = ""
    if kind == "shopping":
        body += (f'<div style="margin:0 16px 12px;padding:10px 14px;border-radius:22px;background:{card};color:#888">'
                 f'🔍 Search {phrase(r, 2).lower()}</div>')
        cells = "".join(f'<div style="background:{card};border-radius:14px;padding:8px">{img_tag(r, photos, 150)}'
                        f'<div style="font-size:13px;margin-top:6px">{phrase(r, 3)}</div>'
                        f'<div style="font-weight:800;color:{acc}">${r.randint(5, 199)}.{r.randint(0, 99):02d}</div>'
                        f'<div style="font-size:11px;color:#999">★★★★☆ {r.randint(12, 9000)} sold</div></div>'
                        for _ in range(r.randint(16, 28)))
        body += f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:0 12px">{cells}</div>'
    elif kind == "feed":
        for _ in range(r.randint(6, 10)):
            body += (f'<div style="background:{card};margin:0 0 10px;padding:12px 0">'
                     f'<div style="display:flex;align-items:center;gap:10px;padding:0 12px 8px">'
                     f'<div style="width:36px;height:36px;border-radius:18px;background:{r.choice(ACCENTS)}"></div>'
                     f'<b>{r.choice(NAMES).lower()}_{r.randint(1, 999)}</b></div>{img_tag(r, photos, r.choice([300, 360, 420]))}'
                     f'<div style="padding:8px 12px;font-size:14px">♡ {r.randint(10, 90)}.{r.randint(1, 9)}K &nbsp; 💬 {r.randint(10, 999)}'
                     f'<br>{phrase(r, 8)}</div></div>')
    elif kind == "finance":
        body += (f'<div style="margin:0 16px;padding:20px;border-radius:20px;background:{acc};color:#fff">'
                 f'<div style="font-size:13px;opacity:.8">Total balance</div><div style="font-size:34px;font-weight:800">'
                 f'${r.randint(100, 99999):,}.{r.randint(0, 99):02d}</div></div>')
        pts = " ".join(f"{i * 20},{100 - r.randint(10, 90)}" for i in range(17))
        body += (f'<svg viewBox="0 0 320 100" style="margin:16px;width:328px;height:120px"><polyline points="{pts}" '
                 f'fill="none" stroke="{acc}" stroke-width="3"/></svg>')
        for _ in range(r.randint(18, 32)):
            amt = r.randint(1, 500)
            body += (f'<div style="display:flex;justify-content:space-between;padding:12px 16px;border-bottom:1px solid #8883">'
                     f'<span>{phrase(r, 2)}<br><small style="color:#999">Sep {r.randint(1, 30)}</small></span>'
                     f'<b style="color:{"#00b37e" if r.random() < 0.3 else fg}">{"+" if r.random() < 0.3 else "-"}${amt}.{r.randint(0, 99):02d}</b></div>')
    elif kind == "chat":
        for _ in range(r.randint(24, 40)):
            me = r.random() < 0.5
            body += (f'<div style="display:flex;justify-content:{"flex-end" if me else "flex-start"};padding:4px 12px">'
                     f'<div style="max-width:72%;padding:9px 13px;border-radius:18px;font-size:15px;'
                     f'background:{acc if me else card};color:{"#fff" if me else fg}">{phrase(r, r.randint(2, 12))}</div></div>')
    elif kind == "food":
        for _ in range(r.randint(8, 14)):
            body += (f'<div style="background:{card};margin:0 12px 12px;border-radius:16px;overflow:hidden">{img_tag(r, photos, 170)}'
                     f'<div style="padding:10px 12px"><b>{phrase(r, 2)}</b><br><small style="color:#999">'
                     f'{r.randint(10, 45)} min · ${r.randint(0, 5)}.99 delivery · ★ {r.randint(40, 50) / 10}</small></div></div>')
    else:                                                               # fitness
        body += '<div style="display:flex;justify-content:space-around;padding:12px">' + "".join(
            f'<svg width="96" height="96"><circle cx="48" cy="48" r="40" stroke="#8883" stroke-width="10" fill="none"/>'
            f'<circle cx="48" cy="48" r="40" stroke="{c}" stroke-width="10" fill="none" stroke-dasharray="{r.randint(60, 250)} 300" '
            f'transform="rotate(-90 48 48)" stroke-linecap="round"/></svg>' for c in r.sample(ACCENTS, 3)) + "</div>"
        for _ in range(r.randint(14, 24)):
            body += (f'<div style="display:flex;justify-content:space-between;margin:0 16px 10px;padding:14px;border-radius:14px;'
                     f'background:{card}"><span>{phrase(r, 2)}</span><b>{r.randint(1, 12000):,} {r.choice(["steps", "kcal", "min", "bpm"])}</b></div>')
    nav = "".join(f"<span>{i}</span>" for i in r.sample(["🏠", "🔍", "➕", "💬", "👤", "🛒", "📊"], 5))
    return (f'<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f'</head><body style="margin:0;background:{bg};color:{fg};font-family:{r.choice(FONTS)}">{head}{body}'
            f'<div style="display:flex;justify-content:space-around;padding:14px 0 24px;font-size:22px;'
            f'background:{card};margin-top:12px">{nav}</div></body></html>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpr", type=int, default=4, help="device pixel ratio: 4 -> 1440 px wide (2K GT), 6 -> 2160 (4K)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    r = random.Random(args.seed)
    photos = None
    if (ANALYSIS / "content_unsplash.csv").exists():
        photos = pd.read_csv(ANALYSIS / "content_unsplash.csv")
        photos = photos[photos.theme != "other"]
    opts = Options()
    for a in ["--headless=new", f"--proxy-server={PROXY}", "--hide-scrollbars", "--no-sandbox", "--window-size=360,740"]:
        opts.add_argument(a)
    os.environ.setdefault("WDM_LOG", "0")
    d = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    d.execute_cdp_cmd("Emulation.setDeviceMetricsOverride", {"width": 360, "height": 740, "deviceScaleFactor": args.dpr, "mobile": True})
    rows = []
    try:
        for i in range(args.n):
            kind = r.choice(["shopping", "feed", "finance", "chat", "food", "fitness"])
            path = OUT / f"ui_{args.dpr}x_{args.seed}_{i:04d}_{kind}.png"
            d.get("data:text/html;charset=utf-8;base64," + base64.b64encode(page(r, kind, photos).encode("utf-8")).decode())
            d.execute_script("return Promise.all([...document.images].map(i => i.complete ? 0 : new Promise(r => { i.onload = i.onerror = r; })))")
            h = d.execute_script("return document.body.scrollHeight")
            shot = d.execute_cdp_cmd("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True,
                                                                "clip": {"x": 0, "y": 0, "width": 360, "height": h, "scale": 1}})
            path.write_bytes(base64.b64decode(shot["data"]))
            rows.append({"file": str(path.relative_to(HQ.parent.parent)), "kind": kind, "css_height": h})
            print(f"[{i + 1}/{args.n}] {path.name} {h} css px tall", flush=True)
    finally:
        d.quit()
    m = OUT / "manifest.csv"
    pd.DataFrame(rows).to_csv(m, mode="a", header=not m.exists(), index=False)


if __name__ == "__main__":
    main()
