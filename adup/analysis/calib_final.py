"""Calibrate the final step of the degradation (resize to LQ size + encode) on real "semi-pairs" from the Meta Ad Library.

The Ad Library serves each ad as two basic H.264 renditions made by Meta from the same upload: HD (720p short side) and
SD (360p). They are not (HQ, LQ) pairs, but together they show what Meta's transcode does to one video at two sizes.
For each ad this script:
  1. reads the real SD's stream facts (codec, profile, bits per pixel, GOP) - the SD encode settings are simply measured;
  2. simulates SD from the real HD with every candidate final step (interpolation x sinc x codec x bpp) of
     adup.degrade.second_order and scores it against the real SD frame by frame (PSNR, SSIM on luma);
  3. reports which candidates come closest, overall and per ad.
Caveat: Meta derives SD from the upload, not from HD, so HD's own H.264 artifacts are in every simulation; the ranking
of candidates is what matters, not the absolute PSNR.

Usage (from the repo root):
  .venv-iqa/bin/python -m adup.analysis.calib_final --split calibration --out outputs/calibration/final_step.csv
"""

import argparse
import itertools
import json
import os
import subprocess
import tempfile

import cv2
import numpy as np
import pandas as pd

from adup.bench.metrics import luma, psnr, ssim
from adup.degrade.second_order import INTERP, codec_args, make_kernel
from adup.media import Writer, stream_frames
from adup.paths import REAL_STATS, ROOT


def stream_info(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=codec_name,profile,width,height,r_frame_rate,nb_read_frames,bit_rate", "-of", "json",
                          path], capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    keys = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey", "-show_entries",
                           "frame=pts_time", "-of", "csv=p=0", path], capture_output=True, text=True).stdout.split()
    n = int(s["nb_read_frames"])
    br = float(s.get("bit_rate") or os.path.getsize(path) * 8 / (n / fps))
    return {"codec": s["codec_name"], "profile": s.get("profile"), "w": int(s["width"]), "h": int(s["height"]), "fps": fps,
            "frames": n, "kbps": br / 1000, "bpp": br / (int(s["width"]) * int(s["height"]) * fps),
            "gop": n / max(len(keys), 1)}


def score(a, b):
    """Mean luma PSNR and SSIM between two frame lists."""
    pairs = [(luma(x), luma(y)) for x, y in zip(a, b)]
    return float(np.mean([psnr(x, y) for x, y in pairs])), float(np.mean([ssim(x, y) for x, y in pairs]))


def simulate(hd, size, fps, interp, sinc, comp, tmp):
    """HD frames -> resize (+ sinc) -> encode -> decoded frames of `size`."""
    kern = make_kernel({"kernel": "sinc", "size": 11, "omega": sinc}) if sinc else None
    frames = []
    for f in hd:
        g = cv2.resize(f, size, interpolation=INTERP[interp])
        if kern is not None:
            g = np.clip(cv2.filter2D(g.astype(np.float32), -1, kern), 0, 255).astype(np.uint8)
        frames.append(g)
    path = os.path.join(tmp, "sim.mp4")
    w = Writer(path, *size, fps, codec_args(comp, *size, fps))
    for g in frames:
        w.write(g)
    w.close()
    kbps = os.path.getsize(path) * 8 / 1000 / (len(frames) / fps)
    return list(stream_frames(path, "null", *size, len(frames))), kbps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="calibration", help="real-ad split to use (data/stats/real_ads/splits.csv)")
    ap.add_argument("--frames", type=int, default=60, help="frames per ad (from the start)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bpp", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.5, 2.0],
                    help="candidate bitrates, as multiples of the real SD's bpp")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sp = pd.read_csv(REAL_STATS / "splits.csv", dtype={"ad_id": str})
    sp = sp[(sp.split == args.split) & sp.video_sd_file.notna()]
    if args.limit:
        sp = sp.head(args.limit)
    rows = []
    for k, r in enumerate(sp.itertuples()):
        hd_p, sd_p = str(ROOT / r.video_file), str(ROOT / r.video_sd_file)
        hd_i, sd_i = stream_info(hd_p), stream_info(sd_p)
        if abs(hd_i["fps"] - sd_i["fps"]) > 0.01 or hd_i["w"] / hd_i["h"] != sd_i["w"] / sd_i["h"]:
            print(f"[{k + 1}/{len(sp)}] {r.ad_id}: HD / SD differ in fps or aspect, skipped", flush=True)
            continue
        n = min(args.frames, hd_i["frames"], sd_i["frames"])
        hd = list(stream_frames(hd_p, "null", hd_i["w"], hd_i["h"], n))
        sd = list(stream_frames(sd_p, "null", sd_i["w"], sd_i["h"], n))
        size = (sd_i["w"], sd_i["h"])
        base = {"ad_id": r.ad_id, "source": r.source, "hd_size": f"{hd_i['w']}x{hd_i['h']}", "sd_size": f"{size[0]}x{size[1]}",
                "sd_codec": sd_i["codec"], "sd_profile": sd_i["profile"], "sd_kbps": round(sd_i["kbps"], 1),
                "sd_bpp": round(sd_i["bpp"], 4), "sd_gop": round(sd_i["gop"], 1), "hd_bpp": round(hd_i["bpp"], 4)}
        with tempfile.TemporaryDirectory() as tmp:
            for interp, sinc, codec, mult in itertools.product(INTERP, [None, 2.0], ["h264", "vp9"], args.bpp):
                comp = {"codec": codec, "bpp": sd_i["bpp"] * mult, "keyint": int(round(sd_i["gop"]))}
                sim, kbps = simulate(hd, size, sd_i["fps"], interp, sinc, comp, tmp)
                p, s = score(sim, sd)
                rows.append({**base, "interp": interp, "sinc": sinc or 0, "codec": codec, "bpp_mult": mult,
                             "sim_kbps": round(kbps, 1), "psnr": round(p, 3), "ssim": round(s, 4)})
        best = max((x for x in rows if x["ad_id"] == r.ad_id), key=lambda x: x["psnr"])
        print(f"[{k + 1}/{len(sp)}] {r.ad_id} SD {base['sd_size']} {base['sd_codec']} {base['sd_bpp']} bpp | best "
              f"{best['interp']} sinc={best['sinc']} {best['codec']} x{best['bpp_mult']}: {best['psnr']} dB", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(args.out, index=False)
    if len(d):
        ads = d.drop_duplicates("ad_id")
        print(f"\n{len(ads)} ads. real SD: codec {ads.sd_codec.value_counts().to_dict()}, bpp median "
              f"{ads.sd_bpp.median():.3f} (q25 {ads.sd_bpp.quantile(.25):.3f}, q75 {ads.sd_bpp.quantile(.75):.3f}), "
              f"GOP median {ads.sd_gop.median():.0f}; HD bpp median {ads.hd_bpp.median():.3f}")
        g = d.groupby(["interp", "sinc", "codec", "bpp_mult"])[["psnr", "ssim"]].mean()
        print("\nmean over ads, best candidates first:")
        print(g.sort_values("psnr", ascending=False).head(12).round(4).to_string())
        for col in ["interp", "sinc", "codec", "bpp_mult"]:
            print(f"\nby {col}:", d.groupby(col).psnr.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
