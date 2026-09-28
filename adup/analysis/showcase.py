"""Figures, side-by-side videos and a local gallery page of example pairs, for showing the dataset to people.

Inputs are pair directories from adup.make_pairs: the examples (random mode, 2 LQ per ad), optionally the same GTs with
heavy degradation (make_pairs --out <heavy> --set degrade.rbvsr_prob=1.0 on the same specs, so the GT is identical) and
grid examples (make_pairs --grid: 4 LQ sizes x 2 codecs). Outputs in --out:
  gt_sheet_<k>.jpg           8 frames of each GT, with its style, layout and overlays
  gt_vs_lq_<k>.jpg           a text region of the GT and of each LQ at GT pixel scale (LQ upscaled with bicubic)
  calibrated_vs_heavy_<k>.jpg  the same region: GT, everyday (calibrated) LQ, heavy LQ
  grid_<ad>.jpg              the same region: GT and the 8 LQ of a grid example
  videos/<ad>.mp4            GT next to every LQ of the ad (side by side for portrait, stacked for landscape), labelled
  index.html                 all of the above on one page (open locally; paths are relative)

Usage (from the repo root):
  .venv-iqa/bin/python -m adup.analysis.showcase --pairs data/pairs/ugc_v7_examples \\
      --heavy data/pairs/ugc_v7_examples_heavy --grid data/pairs/ugc_v7_examples_grid --out outputs/figures/examples
"""

import argparse
import glob
import html
import json
import os
import subprocess

import cv2
import numpy as np

from adup.media import ENC, FF, read_frames
from adup.paths import ASSETS

FONT = ASSETS / "fonts" / "Inter[opsz,wght].ttf"
CROP = 480


def load(pairs):
    return [(os.path.dirname(m), json.load(open(m))) for m in sorted(glob.glob(os.path.join(pairs, "*", "meta.json")))] \
        if pairs else []


def frame(path, k, w, h):
    return read_frames(path, f"select=eq(n\\,{k}),scale={w}:{h}:flags=bicubic", w, h, 1)[0]


def label(img, text, h=30):
    bar = np.full((h, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (6, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], 0)


def pad(img, p=4):
    return np.pad(img, ((p, p), (p, p), (0, 0)), constant_values=255)


def save(path, img):
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def vdesc(v):
    c = v["plan"]["final"]["compress"]
    kind = "HEAVY" if v["plan"]["preset"] == "rbvsr" else "calibrated"
    return f"LQ {v['lq_size'][0]}x{v['lq_size'][1]} {c['codec'].upper() if c else ''} {kind}"


def describe(m):
    sp = m["spec"]
    kinds = ", ".join(sorted({s.get("role") or s.get("layout") or s.get("kind", "video") for s in sp["shots"]}))
    types = ", ".join(sorted({t["type"] for t in m["text"]})) or "none"
    return f"{sp.get('style')} ad, {m['aspect']}, {m['fps']} fps, theme {sp.get('theme')}; shots: {kinds}; overlays: {types}"


def text_region(d, m):
    """(frame index, x0, y0) of a CROP x CROP region around the most text in the middle half of the ad (mask.mkv)."""
    w, h = m["gt_size"]
    best = (-1, m["frames"] // 2, w // 2, h // 2)
    for k in np.linspace(m["frames"] * 0.25, m["frames"] * 0.75, 9).astype(int):
        mask = read_frames(os.path.join(d, "mask.mkv"), f"select=eq(n\\,{k}),format=gray,format=rgb24", w, h, 1)[0][..., 0]
        s = int((mask > 127).sum())
        if s > best[0]:
            ys, xs = np.nonzero(mask > 127)
            best = (s, int(k), int(xs.mean()) if s else w // 2, int(ys.mean()) if s else h // 2)
    _, k, cx, cy = best
    return k, int(np.clip(cx - CROP / 2, 0, w - CROP)), int(np.clip(cy - CROP / 2, 0, h - CROP))


def crop(path, m, region):
    w, h = m["gt_size"]
    k, x0, y0 = region
    return frame(path, k, w, h)[y0:y0 + CROP, x0:x0 + CROP]


def rows_to_sheets(rows, out, name, per=4):
    files = []
    for s0 in range(0, len(rows), per):
        f = f"{name}_{s0 // per + 1}.jpg"
        save(os.path.join(out, f), np.concatenate(rows[s0:s0 + per], 0))
        files.append(f)
    return files


def gt_strip(d, m, tw=200, th=356):
    w, h = m["gt_size"]
    tiles = []
    for i in range(8):
        f = frame(os.path.join(d, "gt.mp4"), int((i + 0.5) / 8 * m["frames"]), w, h)
        sc = min(tw / w, th / h)
        f = cv2.resize(f, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
        c = np.full((th, tw, 3), 128, np.uint8)
        y, x = (th - f.shape[0]) // 2, (tw - f.shape[1]) // 2
        c[y:y + f.shape[0], x:x + f.shape[1]] = f
        tiles.append(pad(c, 2))
    return label(np.concatenate(tiles, 1), describe(m))


def video(out, d, m, extra=()):
    """GT next to every LQ (and the heavy LQ of the same GT), each upscaled to the GT size and labelled."""
    w, h = m["gt_size"]
    ins = [(os.path.join(d, "gt.mp4"), f"GT {w}x{h}")] + [(os.path.join(d, v["name"] + ".mp4"), vdesc(v)) for v in m["variants"]]
    ins += list(extra)
    fs = max(int(min(w, h) * 0.03), 18)
    chains = [f"[{i}:v]scale={w}:{h}:flags=bicubic,drawtext=fontfile='{FONT}':text='{t}':x=20:y=20:fontsize={fs}:"
              f"fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8[v{i}]" for i, (_, t) in enumerate(ins)]
    stack = "hstack" if h > w else "vstack"
    graph = ";".join(chains) + ";" + "".join(f"[v{i}]" for i in range(len(ins))) + f"{stack}=inputs={len(ins)}"
    subprocess.run([*FF, "-y", *[a for p, _ in ins for a in ("-i", p)], "-filter_complex", graph, "-c:v", "libx264",
                    "-preset", "medium", "-crf", "20", *ENC, "-pix_fmt", "yuv420p", "-an", out], check=True)


PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>UGC Ad Pairs</title><style>
:root {{ --bg: #fafaf8; --fg: #1d1d1b; --muted: #6b6b66; --line: #e2e1dc; --card: #fff; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #161615; --fg: #ecebe6; --muted: #9a9993; --line: #2e2e2b; --card: #1f1f1d; }} }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.6 system-ui, "Noto Sans CJK SC", sans-serif; margin: 0; }}
main {{ max-width: 1200px; margin: 0 auto; padding: 32px 16px 64px; }}
h1 {{ font-size: 24px; margin: 0 0 4px; }} h2 {{ font-size: 18px; margin: 36px 0 8px; }}
p.lead {{ color: var(--muted); margin: 0 0 20px; }}
.ad {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px; margin: 14px 0; }}
.ad h3 {{ font-size: 14px; margin: 0 0 8px; font-weight: 600; }}
.ad table {{ font-size: 13px; border-collapse: collapse; margin: 8px 0; }}
.ad td, .ad th {{ padding: 2px 10px 2px 0; text-align: left; color: var(--muted); }}
video, img {{ width: 100%; height: auto; border-radius: 6px; display: block; background: #000; }}
.heavy {{ color: #c0392b; font-weight: 600; }}
nav a {{ margin-right: 14px; color: inherit; }}
</style></head><body><main>
<h1>UGC 广告配对示例</h1>
<p class="lead">{lead}</p>
<nav>{nav}</nav>
{body}
</main></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="example pairs (random mode)")
    ap.add_argument("--heavy", help="the same GTs with heavy degradation")
    ap.add_argument("--grid", help="grid examples (make_pairs --grid)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-videos", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "videos"), exist_ok=True)
    ex, heavy, grid = load(args.pairs), {os.path.basename(d): (d, m) for d, m in load(args.heavy)}, load(args.grid)
    regions = {d: text_region(d, m) for d, m in ex + grid}

    sheets = rows_to_sheets([gt_strip(d, m) for d, m in ex], args.out, "gt_sheet")
    rows = []
    for d, m in ex:
        w, h = m["gt_size"]
        tiles = [label(crop(os.path.join(d, "gt.mp4"), m, regions[d]), f"GT {w}x{h}")]
        tiles += [label(crop(os.path.join(d, v["name"] + ".mp4"), m, regions[d]), vdesc(v)) for v in m["variants"]]
        rows.append(pad(np.concatenate([pad(t) for t in tiles], 1)))
    crops = rows_to_sheets(rows, args.out, "gt_vs_lq")
    rows = []
    for d, m in ex:
        if os.path.basename(d) not in heavy:
            continue
        hd, hm = heavy[os.path.basename(d)]
        cal = next((v for v in m["variants"] if v["plan"]["preset"] == "calibrated"), m["variants"][0])
        tiles = [label(crop(os.path.join(d, "gt.mp4"), m, regions[d]), f"GT {m['gt_size'][0]}x{m['gt_size'][1]}"),
                 label(crop(os.path.join(d, cal["name"] + ".mp4"), m, regions[d]), vdesc(cal)),
                 label(crop(os.path.join(hd, "lq_0.mp4"), m, regions[d]), vdesc(hm["variants"][0]))]
        rows.append(pad(np.concatenate([pad(t) for t in tiles], 1)))
    heavy_sheets = rows_to_sheets(rows, args.out, "calibrated_vs_heavy")
    grid_figs = []
    for d, m in grid:
        tiles = [label(crop(os.path.join(d, "gt.mp4"), m, regions[d]), f"GT {m['gt_size'][0]}x{m['gt_size'][1]}")]
        tiles += [label(crop(os.path.join(d, v["name"] + ".mp4"), m, regions[d]), vdesc(v)) for v in m["variants"]]
        tiles += [np.full_like(tiles[0], 255)] * (-len(tiles) % 3)
        img = pad(np.concatenate([np.concatenate([pad(t) for t in tiles[i:i + 3]], 1) for i in range(0, len(tiles), 3)], 0))
        f = f"grid_{os.path.basename(d)}.jpg"
        save(os.path.join(args.out, f), img)
        grid_figs.append((f, m))
    print(f"figures: {len(sheets)} GT sheets, {len(crops)} crop sheets, {len(heavy_sheets)} heavy sheets, {len(grid_figs)} grid",
          flush=True)

    blocks = []
    for d, m in ex:
        name = os.path.basename(d)
        extra = []
        if name in heavy:
            hd, hm = heavy[name]
            extra = [(os.path.join(hd, "lq_0.mp4"), vdesc(hm["variants"][0]))]
        if not args.no_videos and not os.path.exists(os.path.join(args.out, "videos", f"{name}.mp4")):
            video(os.path.join(args.out, "videos", f"{name}.mp4"), d, m, extra)
            print(f"video {name}", flush=True)
        lq = [(v, False) for v in m["variants"]] + ([(heavy[name][1]["variants"][0], True)] if name in heavy else [])
        rows = "".join(
            f"<tr><td>{'重退化版本（同一 GT）' if extra_ else v['name']}</td><td>{v['lq_size'][0]}×{v['lq_size'][1]}</td>"
            f"<td>{(v['plan']['final']['compress'] or {}).get('codec', '-')}</td>"
            f"<td{' class=heavy' if v['plan']['preset'] == 'rbvsr' else ''}>{'重退化' if v['plan']['preset'] == 'rbvsr' else '日常（校准）'}</td></tr>"
            for v, extra_ in lq)
        blocks.append(f"<div class=ad><h3>{html.escape(describe(m))}</h3>"
                      f"<table><tr><th>LQ</th><th>尺寸</th><th>编码</th><th>退化</th></tr>{rows}</table>"
                      f"<video src='videos/{name}.mp4' controls muted loop preload=none></video></div>")
    img = lambda fs: "".join(f"<img src='{f}' loading=lazy alt=''>" for f in fs)
    body = (f"<h2 id=gt>GT 总览</h2>{img(sheets)}"
            f"<h2 id=crop>GT 与 LQ 的文字区域（原始像素，LQ 双三次放大）</h2>{img(crops)}"
            + (f"<h2 id=heavy>同一 GT：日常退化 vs 重退化</h2>{img(heavy_sheets)}" if heavy_sheets else "")
            + ("<h2 id=grid>测试集的网格：同一 GT 的 8 个 LQ（4 种尺寸 × 2 种编码）</h2>"
               + "".join(f"<p>{html.escape(describe(m))}</p><img src='{f}' loading=lazy alt=''>" for f, m in grid_figs)
               if grid_figs else "")
            + "<h2 id=videos>逐条视频（GT 与各 LQ 并排）</h2>" + "".join(blocks))
    n_lq = sum(len(m["variants"]) for _, m in ex)
    n_heavy = sum(v["plan"]["preset"] == "rbvsr" for _, m in ex for v in m["variants"])
    lead = (f"{len(ex)} 条合成广告、{n_lq} 个 LQ（其中 {n_heavy} 个抽到重退化，训练设置为 10%）；"
            f"{len(heavy)} 条附同一 GT 的重退化版本；{len(grid)} 条测试集网格例子。GT 1080p，LQ 360 / 540 / 720p（网格另有 270p）。")
    nav = "".join(f"<a href='#{a}'>{t}</a>" for a, t in [("gt", "GT 总览"), ("crop", "文字区域"), ("heavy", "重退化"),
                                                        ("grid", "网格"), ("videos", "视频")])
    with open(os.path.join(args.out, "index.html"), "w") as f:
        f.write(PAGE.format(lead=lead, nav=nav, body=body))
    print(f"-> {os.path.join(args.out, 'index.html')}")


if __name__ == "__main__":
    main()
