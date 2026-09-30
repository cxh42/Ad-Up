"""Score benchmark outputs on synthetic pairs: picture quality, text fidelity, temporal stability, per factor.

For every (method, ad, LQ variant) it reads the 1080p output and the GT and computes:
  quality    luma PSNR / SSIM on every frame; LPIPS and DISTS (pyiqa) every --every frames
  text       PSNR inside the overlay mask (mask.mkv); OCR (EasyOCR) of every text segment's box in the middle frame of
             the segment, against the words the composer drew (meta.json text[].texts). OCR itself errs on clean text
             (CER ~0.1 on stylised captions), so everything is relative to OCR on the GT crop: segments whose GT reading
             has CER > --max-gt-cer are left out, and text_dcer = CER(output) - CER(GT) is the text the method lost.
             Each segment is
               correct       the output reads no worse than the GT
               hallucinated  worse, yet read confidently (mean OCR confidence >= --conf) and differently from the GT:
                             crisp, wrong text, e.g. a price digit changed by a generative model - the costly error
               illegible     worse and not confident: blurred or broken text
  temporal   mean |(out[t+1] - out[t]) - (gt[t+1] - gt[t])| on luma inside shots (lower = no extra flicker)
  cuts       PSNR of frames within --near-cut frames of a shot cut vs the rest
and joins the factors from meta.json (LQ size, codec, preset, aspect, theme, split) and the cost records of
adup.bench.run. Tables by method x LQ size and method x codec are printed; the rows go to --csv.

Usage (from the repo root):
  .venv-iqa/bin/python -m adup.bench.evaluate --pairs data/pairs/v7_dev --runs outputs/bench/v7_dev \\
      --methods bicubic realesrgan dove --csv outputs/bench/v7_dev/results.csv
"""

import argparse
import glob
import json
import os
import subprocess

import numpy as np
import pandas as pd
import pyiqa
import torch

from adup.bench.metrics import cer, luma, norm_text, psnr, ssim


def read(path, w, h):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def read_mask(path, w, h):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w)


def text_segments(meta, max_n):
    """(frame, box, text) for the middle frame of every text segment with words, at most max_n spread evenly."""
    segs = []
    for item in meta["text"]:
        for (f0, f1, x0, y0, x1, y1), t in zip(item["boxes"], item.get("texts", [])):
            if norm_text(t):
                segs.append(((f0 + f1) // 2, (x0, y0, x1, y1), t))
    if len(segs) > max_n:
        segs = [segs[int(i)] for i in np.linspace(0, len(segs) - 1, max_n)]
    return segs


def crop(f, box, pad=6):
    x0, y0, x1, y1 = box
    return f[max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad]


class Scorer:
    def __init__(self, every, max_ocr, near_cut, conf, max_gt_cer):
        import easyocr
        self.lpips = pyiqa.create_metric("lpips", device="cuda")
        self.dists = pyiqa.create_metric("dists", device="cuda")
        self.ocr = easyocr.Reader(["en"], gpu=True, verbose=False)
        self.every, self.max_ocr, self.near_cut, self.conf, self.max_gt_cer = every, max_ocr, near_cut, conf, max_gt_cer
        self.gt_cache = {}

    def read_text(self, img):
        """Text in reading order and its mean OCR confidence."""
        det = self.ocr.readtext(np.ascontiguousarray(img))
        det = sorted(det, key=lambda d: (round(min(p[1] for p in d[0]) / 20), min(p[0] for p in d[0])))
        return " ".join(d[1] for d in det), float(np.mean([d[2] for d in det])) if det else 0.0

    def gt(self, pair_dir, meta):
        if pair_dir not in self.gt_cache:
            self.gt_cache.clear()
            w, h = meta["gt_size"]
            gt = read(os.path.join(pair_dir, "gt.mp4"), w, h)
            mask = read_mask(os.path.join(pair_dir, "mask.mkv"), w, h) > 127
            segs = [(f, b, t, self.read_text(crop(gt[f], b))[0]) for f, b, t in text_segments(meta, self.max_ocr)]
            segs = [(f, b, t, g, cer(g, t)) for f, b, t, g in segs]
            self.gt_cache[pair_dir] = (gt, mask, segs)
        return self.gt_cache[pair_dir]

    @torch.no_grad()
    def __call__(self, pair_dir, meta, out_path):
        gt, mask, segs = self.gt(pair_dir, meta)
        w, h = meta["gt_size"]
        out = read(out_path, w, h)
        n = min(len(gt), len(out))
        cuts = [s["start_frame"] for s in meta.get("shots", [])[1:]]
        tens = lambda x: torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].float().cuda() / 255
        ps, ss, pm, lp, ds, near, far, td = [], [], [], [], [], [], [], []
        prev = None
        for i in range(n):
            yo, yg = luma(out[i]), luma(gt[i])
            p = psnr(yo, yg)
            ps.append(p)
            ss.append(ssim(yo, yg))
            m = psnr(yo, yg, mask[i]) if i < len(mask) else None
            if m is not None:
                pm.append(m)
            (near if any(abs(i - c) < self.near_cut or abs(i - c + 1) < self.near_cut for c in cuts) else far).append(p)
            if i % self.every == 0:
                lp.append(float(self.lpips(tens(out[i]), tens(gt[i]))))
                ds.append(float(self.dists(tens(out[i]), tens(gt[i]))))
            if prev is not None and i not in cuts:
                td.append(float(np.mean(np.abs((yo - prev[0]) - (yg - prev[1])))))
            prev = (yo, yg)
        res = {"psnr": np.mean(ps), "ssim": np.mean(ss), "lpips": np.mean(lp), "dists": np.mean(ds),
               "text_psnr": np.mean(pm) if pm else np.nan, "tdiff": np.mean(td) if td else np.nan,
               "psnr_near_cut": np.mean(near) if near else np.nan, "psnr_far": np.mean(far) if far else np.nan}
        verdicts, cers, dcers = [], [], []
        for f, box, truth, gt_read, gt_cer in segs:
            if f >= n or gt_cer > self.max_gt_cer:
                continue                                     # OCR cannot read even the GT: not a fair test
            text, conf = self.read_text(crop(out[f], box))
            c = cer(text, truth)
            cers.append(c)
            dcers.append(c - gt_cer)
            if c <= gt_cer:
                verdicts.append("correct")
            elif conf >= self.conf and norm_text(text) != norm_text(gt_read):
                verdicts.append("hallucinated")
            else:
                verdicts.append("illegible")
        k = max(len(verdicts), 1)
        res.update({"text_segments": len(verdicts), "text_cer": np.mean(cers) if cers else np.nan,
                    "text_dcer": np.mean(dcers) if dcers else np.nan,
                    "text_correct": verdicts.count("correct") / k if verdicts else np.nan,
                    "text_halluc": verdicts.count("hallucinated") / k if verdicts else np.nan,
                    "text_illegible": verdicts.count("illegible") / k if verdicts else np.nan})
        return res


def factors(meta, variant):
    first = next((s for s in meta["spec"]["shots"] if s.get("kind", "video") == "video"), {})
    return {"ad": meta["id"], "variant": variant["name"], "split": meta.get("split"), "aspect": meta["aspect"],
            "lq_short": min(variant["lq_size"]), "scale": round(variant["scale"], 2),
            "codec": variant.get("codec") or variant["plan"]["final"]["compress"]["codec"],
            "preset": variant["plan"]["preset"], "category": first.get("category"), "frames": meta["frames"],
            "shots": len(meta["spec"]["shots"]), "text_items": len(meta["text"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--runs", required=True, help="adup.bench.run output root (<runs>/<method>/<ad>__<variant>.mp4)")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--every", type=int, default=4, help="LPIPS / DISTS on every n-th frame")
    ap.add_argument("--max-ocr", type=int, default=30, help="text segments OCR'd per ad")
    ap.add_argument("--near-cut", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.8, help="OCR confidence above which a wrong reading is a hallucination")
    ap.add_argument("--max-gt-cer", type=float, default=0.3, help="skip text segments OCR misreads on the GT beyond this")
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    scorer = Scorer(args.every, args.max_ocr, args.near_cut, args.conf, args.max_gt_cer)
    done = pd.read_csv(args.csv) if os.path.exists(args.csv) else pd.DataFrame()
    have = set(zip(done.get("method", []), done.get("ad", []), done.get("variant", [])))
    rows = []
    metas = [(os.path.dirname(m), json.load(open(m))) for m in sorted(glob.glob(os.path.join(args.pairs, "*", "meta.json")))]
    for pair_dir, meta in metas:
        for v in meta["variants"]:
            for method in args.methods:
                if (method, meta["id"], v["name"]) in have:
                    continue
                out = os.path.join(args.runs, method, f"{meta['id']}__{v['name']}.mp4")
                if not os.path.exists(out):
                    continue
                rec = json.load(open(out[:-4] + ".json")) if os.path.exists(out[:-4] + ".json") else {}
                row = {"method": method, **factors(meta, v), **scorer(pair_dir, meta, out),
                       "s_per_frame": rec.get("s_per_frame"), "peak_gpu_gib": rec.get("peak_gpu_gib"), "gpu": rec.get("gpu")}
                rows.append(row)
                print(f"{method:12s} {meta['id'][:28]} {v['name']:14s} PSNR {row['psnr']:.2f} LPIPS {row['lpips']:.3f} "
                      f"text {row['text_correct']:.2f}/{row['text_halluc']:.2f}/{row['text_illegible']:.2f} "
                      f"(n={row['text_segments']})", flush=True)
        pd.concat([done, pd.DataFrame(rows)]).to_csv(args.csv, index=False)
    d = pd.concat([done, pd.DataFrame(rows)])
    if not len(d):
        return
    cols = ["psnr", "ssim", "lpips", "dists", "text_psnr", "text_dcer", "text_correct", "text_halluc", "text_illegible",
            "tdiff", "s_per_frame"]
    pd.set_option("display.width", 200)
    print("\nby method:")
    print(d.groupby("method")[cols].mean().round(4).to_string())
    for f in ["lq_short", "codec"]:
        for m in ["psnr", "lpips", "text_correct"]:
            print(f"\n{m} by method x {f}:")
            print(d.pivot_table(index="method", columns=f, values=m, aggfunc="mean").round(4).to_string())


if __name__ == "__main__":
    main()
