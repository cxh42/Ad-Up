"""Synthesize paired (GT, LQ) videos that mimic how real UGC ads are degraded between the phone and the viewer.

Chain (parameters are sampled once per LQ variant and held fixed over the clip, like a real upload):
  0. GT:        portrait crop of the HQ source -> resize to --gt-size -> optional burned-in captions/stickers
  1. capture:   defocus blur, sensor noise, ISP denoise / skin smoothing, ISP unsharp-mask sharpening
  2. edit:      export from an editing app (x264, moderate CRF)
  3. platform:  resize to GT/scale -> x264 High profile at low CRF/bitrate (BVC/x264-like settings seen on TikTok)
  4. re-upload: optional extra generation (fake-HD upscale -> re-encode -> back to LQ size)
Frame count and fps are preserved end to end so GT and LQ stay frame-aligned.

Usage:
  .venv-iqa/bin/python -m adup.degrade.pipeline --manifest data/hq/ultravideo/manifest.csv --out data/pairs/uv_p1080_x2 \
      --gt-size 1080x1920 --scale 2 --variants 2
"""

import argparse
import json
import os
import random
import subprocess
import tempfile

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
CAPTIONS = ["wait for it...", "this changed my skin", "honestly obsessed", "3 reasons you need this", "POV: you finally found it",
            "link in bio", "50% OFF TODAY", "I was today years old", "don't skip this", "game changer!!", "unboxing my new fave",
            "ok but why is nobody talking about this", "day 7 update", "before vs after", "run don't walk", "FREE SHIPPING"]


# ---------------------------------------------------------------- ffmpeg helpers
def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_frames", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(num) / float(den), int(s.get("nb_frames") or 0)


def read_frames(path, vf, w, h, max_frames):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", vf, "-frames:v", str(max_frames),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def write_frames(frames, path, fps, x264_args):
    h, w = frames.shape[1:3]
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}",
           "-i", "-", "-c:v", "libx264", *x264_args, "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError(f"ffmpeg failed writing {path}")


def transcode(src, dst, vf, x264_args):
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", vf, "-c:v", "libx264", *x264_args,
           "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-an", dst]
    subprocess.run(cmd, check=True)


def platform_x264(crf, maxrate_k, keyint, preset="medium"):
    return ["-preset", preset, "-profile:v", "high", "-crf", f"{crf}", "-maxrate", f"{maxrate_k}k",
            "-bufsize", f"{2 * maxrate_k}k", "-g", f"{keyint}", "-bf", "3", "-refs", "4"]


# ---------------------------------------------------------------- stage 0: GT
def detail(gray):
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def portrait_crop_filter(src, sw, sh, gw, gh, rng, candidates=7):
    """Crop the largest region of the target aspect ratio, placed where a middle frame has the most texture, then
    downscale to GT size. Random crops of landscape footage often land on empty or out-of-focus background."""
    target_ar = gw / gh
    cw, ch = (int(sh * target_ar) // 2 * 2, sh) if sw / sh > target_ar else (sw, int(sw / target_ar) // 2 * 2)
    if cw < gw or ch < gh:
        raise ValueError(f"source {sw}x{sh} too small for GT {gw}x{gh} (crop would be {cw}x{ch})")
    y = (sh - ch) // 2 // 2 * 2
    mid = read_frames(src, "select=eq(n\\,30)", sw, sh, 1)[0]
    gray = cv2.cvtColor(mid, cv2.COLOR_RGB2GRAY)
    xs = sorted({int(v) // 2 * 2 for v in np.linspace(0, sw - cw, candidates)})
    scored = [(detail(cv2.resize(gray[y:y + ch, x:x + cw], (gw, gh), interpolation=cv2.INTER_AREA)), x) for x in xs]
    best = max(scored)[1]
    x = min(max(best + rng.randint(-cw // 20, cw // 20) // 2 * 2, 0), sw - cw)
    return f"crop={cw}:{ch}:{x}:{y},scale={gw}:{gh}:flags=lanczos"


def gt_stats(frames):
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames[:: max(len(frames) // 8, 1)]]
    return {"gt_mean_luma": float(np.mean([g.mean() for g in grays])),
            "gt_detail": float(np.median([detail(g) for g in grays]))}


def burn_overlays(frames, fps, rng):
    """Burn short caption segments (like auto-captions / CapCut text) and an occasional sticker into the GT."""
    n, h, w = frames.shape[:3]
    out = frames.copy()
    seg = max(int(fps * rng.uniform(1.0, 2.5)), 1)
    size = int(w * rng.uniform(0.045, 0.075))
    font = ImageFont.truetype(FONT, size)
    style = rng.choice(["stroke", "box"])
    y_pos = int(h * rng.uniform(0.55, 0.8))
    sticker = rng.random() < 0.4
    for start in range(0, n, seg):
        text = rng.choice(CAPTIONS)
        for i in range(start, min(start + seg, n)):
            img = Image.fromarray(out[i])
            d = ImageDraw.Draw(img)
            tw = d.textlength(text, font=font)
            x = (w - tw) / 2
            if style == "box":
                d.rounded_rectangle([x - size * 0.4, y_pos - size * 0.25, x + tw + size * 0.4, y_pos + size * 1.25],
                                    radius=size // 3, fill=(255, 255, 255))
                d.text((x, y_pos), text, font=font, fill=(0, 0, 0))
            else:
                d.text((x, y_pos), text, font=font, fill=(255, 255, 255), stroke_width=max(size // 12, 2),
                       stroke_fill=(0, 0, 0))
            if sticker:
                sw_, sh_ = int(w * 0.32), int(w * 0.11)
                d.rounded_rectangle([w * 0.06, h * 0.12, w * 0.06 + sw_, h * 0.12 + sh_], radius=sh_ // 3,
                                    fill=(255, 45, 85))
                d.text((w * 0.06 + sh_ * 0.35, h * 0.12 + sh_ * 0.2), "SALE", font=ImageFont.truetype(FONT, int(sh_ * 0.55)),
                       fill=(255, 255, 255))
            out[i] = np.asarray(img)
    return out


# ---------------------------------------------------------------- stage 1: capture / ISP
# Sampling ranges, calibrated against real TikTok ads at matched 720x1280 LQ size (adup.analysis.compare):
# v1 (first guess, configs/degradation/v1.json) was too harsh; this v3 matches DOVER, DOVER-technical, noise, sharpening
# overshoot and blockiness (W/IQR <= 0.4). MUSIQ / CLIP-IQA / effective resolution stay slightly low, but at the level
# of the undegraded GT itself, i.e. the remaining gap comes from the soft cinematic UltraVideo sources, not the chain.
DEFAULT_CONFIG = {
    "blur_prob": 0.2, "blur_sigma": [0.3, 1.2],
    "noise_prob": 0.25, "noise_sigma": [1.0, 4.0],
    "denoise_prob": 0.4, "denoise_strength": [20, 50],
    "sharpen_prob": 0.8, "sharpen_amount": [0.5, 1.5], "sharpen_sigma": [0.8, 2.0],
    "edit_prob": 0.7, "edit_crf": [16, 22],
    "resize_flags": ["bicubic", "bilinear", "area", "lanczos"],
    "platform_crf": [20, 28], "platform_maxrate_k": [1200, 1600, 2000, 3000, 4000],
    "keyint": [60, 110, 150, 250],
    "reupload_prob": 0.2, "reupload_up": [1.0, 1.5, 2.0], "reupload_crf": [22, 28],
}


def sample_capture(rng, c):
    u = lambda k: rng.uniform(*c[k])
    return {
        "blur_sigma": u("blur_sigma") if rng.random() < c["blur_prob"] else 0.0,
        "noise_sigma": u("noise_sigma") if rng.random() < c["noise_prob"] else 0.0,
        "denoise": rng.choice(["bilateral", "skin"]) if rng.random() < c["denoise_prob"] else "none",
        "denoise_strength": u("denoise_strength"),
        "sharpen_amount": u("sharpen_amount") if rng.random() < c["sharpen_prob"] else 0.0,
        "sharpen_sigma": u("sharpen_sigma"),
    }


def skin_mask(rgb):
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    m = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))
    return cv2.GaussianBlur(m, (0, 0), 3)[..., None].astype(np.float32) / 255


def apply_capture(frames, p, rng):
    out = np.empty_like(frames)
    nprng = np.random.default_rng(rng.randint(0, 2 ** 31))
    for i, f in enumerate(frames):
        x = f.astype(np.float32)
        if p["blur_sigma"] > 0:
            x = cv2.GaussianBlur(x, (0, 0), p["blur_sigma"])
        if p["noise_sigma"] > 0:
            luma = nprng.normal(0, p["noise_sigma"], x.shape[:2])[..., None]
            chroma = nprng.normal(0, p["noise_sigma"] * 0.5, x.shape)
            x = x + luma + chroma
        x = np.clip(x, 0, 255)
        if p["denoise"] != "none":
            smooth = cv2.bilateralFilter(x.astype(np.uint8), 9, p["denoise_strength"], 7).astype(np.float32)
            x = smooth if p["denoise"] == "bilateral" else x * (1 - skin_mask(x.astype(np.uint8))) + smooth * skin_mask(x.astype(np.uint8))
        if p["sharpen_amount"] > 0:
            x = x + p["sharpen_amount"] * (x - cv2.GaussianBlur(x, (0, 0), p["sharpen_sigma"]))
        out[i] = np.clip(x, 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------- stages 2-4
def sample_chain(rng, c):
    return {
        "edit_export": rng.random() < c["edit_prob"],
        "edit_crf": rng.randint(*c["edit_crf"]),
        "resize_flags": rng.choice(c["resize_flags"]),
        "platform_crf": rng.randint(*c["platform_crf"]),
        "platform_maxrate_k": rng.choice(c["platform_maxrate_k"]),
        "keyint": rng.choice(c["keyint"]),
        "reupload": rng.random() < c["reupload_prob"],
        "reupload_up": rng.choice(c["reupload_up"]),
        "reupload_crf": rng.randint(*c["reupload_crf"]),
    }


def make_variant(gt_frames, gt_path, fps, lq_w, lq_h, out_path, rng, tmpdir, config):
    cap = sample_capture(rng, config)
    chain = sample_chain(rng, config)
    frames = apply_capture(gt_frames, cap, rng)
    stage = os.path.join(tmpdir, "edit.mp4")
    edit_args = ["-preset", "fast", "-crf", str(chain["edit_crf"] if chain["edit_export"] else 8)]
    write_frames(frames, stage, fps, edit_args)

    platform = os.path.join(tmpdir, "platform.mp4")
    transcode(stage, platform, f"scale={lq_w}:{lq_h}:flags={chain['resize_flags']}",
              platform_x264(chain["platform_crf"], chain["platform_maxrate_k"], chain["keyint"]))
    final = platform
    if chain["reupload"]:
        up_w, up_h = int(lq_w * chain["reupload_up"]) // 2 * 2, int(lq_h * chain["reupload_up"]) // 2 * 2
        mid = os.path.join(tmpdir, "reupload_mid.mp4")
        transcode(platform, mid, f"scale={up_w}:{up_h}:flags=bicubic",
                  platform_x264(chain["reupload_crf"], chain["platform_maxrate_k"], chain["keyint"]))
        final = os.path.join(tmpdir, "reupload.mp4")
        transcode(mid, final, f"scale={lq_w}:{lq_h}:flags={chain['resize_flags']}",
                  platform_x264(chain["platform_crf"], chain["platform_maxrate_k"], chain["keyint"]))
    os.replace(final, out_path)
    return {"capture": cap, "chain": chain}


def process_clip(src, out_dir, gt_w, gt_h, scale, variants, max_frames, overlay_prob, seed, min_detail, luma_range,
                 config):
    rng = random.Random(seed)
    sw, sh, fps, _ = probe(src)
    fps = round(fps, 3)
    vf = portrait_crop_filter(src, sw, sh, gt_w, gt_h, rng)
    gt_frames = read_frames(src, vf, gt_w, gt_h, max_frames)
    stats = gt_stats(gt_frames)
    if stats["gt_detail"] < min_detail or not luma_range[0] <= stats["gt_mean_luma"] <= luma_range[1]:
        raise ValueError(f"GT rejected: detail={stats['gt_detail']:.1f} luma={stats['gt_mean_luma']:.0f}")
    overlay = rng.random() < overlay_prob
    if overlay:
        gt_frames = burn_overlays(gt_frames, fps, rng)
    os.makedirs(out_dir, exist_ok=True)
    gt_path = os.path.join(out_dir, "gt.mp4")
    write_frames(gt_frames, gt_path, fps, ["-preset", "slow", "-crf", "10"])
    lq_w, lq_h = int(round(gt_w / scale)) // 2 * 2, int(round(gt_h / scale)) // 2 * 2
    meta = {"src": src, "fps": fps, "frames": len(gt_frames), "gt_size": [gt_w, gt_h], "lq_size": [lq_w, lq_h],
            "scale": scale, "crop_filter": vf, "overlay": overlay, **stats, "config": config, "variants": []}
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        for k in range(variants):
            out = os.path.join(out_dir, f"lq_{k}.mp4")
            params = make_variant(gt_frames, gt_path, fps, lq_w, lq_h, out, rng, tmp, config)
            meta["variants"].append({"file": out, **params})
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV with a `file` column of HQ source videos")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt-size", default="1080x1920")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--overlay-prob", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-detail", type=float, default=12.0, help="min median Laplacian variance of GT frames")
    ap.add_argument("--luma-range", type=float, nargs=2, default=(30, 230))
    ap.add_argument("--config", help="JSON file overriding DEFAULT_CONFIG sampling ranges")
    args = ap.parse_args()
    config = {**DEFAULT_CONFIG, **(json.load(open(args.config)) if args.config else {})}
    gt_w, gt_h = map(int, args.gt_size.split("x"))
    files = pd.read_csv(args.manifest)["file"].tolist()
    if args.limit:
        files = files[:args.limit]
    for i, src in enumerate(files):
        name = os.path.splitext(os.path.basename(src))[0]
        out_dir = os.path.join(args.out, name)
        if os.path.exists(os.path.join(out_dir, "meta.json")):
            continue
        try:
            process_clip(src, out_dir, gt_w, gt_h, args.scale, args.variants, args.max_frames, args.overlay_prob,
                         args.seed * 100003 + i, args.min_detail, args.luma_range, config)
            print(f"[{i + 1}/{len(files)}] {name}", flush=True)
        except Exception as e:
            print(f"[{i + 1}/{len(files)}] {name} failed: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
