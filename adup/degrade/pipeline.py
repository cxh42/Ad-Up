"""Synthesize paired (GT, LQ) videos that mimic how real UGC ads are degraded between the phone and the viewer.

Input is either a manifest of single-shot HQ clips (--manifest) or multi-shot sequence specs from adup.shots.compose
(--sequences). Everything streams frame by frame, so 2K / 4K GT and 40 s sequences fit in memory.

Chain (parameters are sampled once per LQ variant and held fixed over the clip, like a real upload):
  0. GT:        crop of each HQ shot at the sequence's aspect ratio (9:16 / 4:5 / 1:1 / 16:9), downscaled (never
                upscaled) to the GT size set by --gt-short; edit transitions between shots (cut, dissolve, whip, dip);
                burned-in text across the whole sequence (captions, titles, stickers, fine print; see overlays.py)
  1. capture:   defocus blur, sensor noise, ISP denoise / skin smoothing, ISP unsharp-mask sharpening
  2. edit:      export from an editing app (x264, moderate CRF)
  3. platform:  resize to GT/scale -> H.264 (TikTok / Meta-like x264 High profile), VP9 or AV1 (YouTube-like) or HEVC
  4. re-upload: optional extra generation (fake-HD upscale -> re-encode with any codec -> back to LQ size)
Frame count and fps are preserved end to end so GT and LQ stay frame-aligned. VP9 / AV1 LQ is decoded and stored as
lossless H.264, because the pip builds of OpenCV and decord cannot decode AV1; the pixels are exactly the decoded ones.

Output per pair: gt.mp4, lq_<k>.mp4 and meta.json for the whole sequence; for multi-shot sequences also
shots/shot_<i>_{gt,lq_<k>}.mp4 (frame-exact, lossless) and meta.json["shots"] with each shot's source and boundaries.

Usage:
  .venv-iqa/bin/python -m adup.degrade.pipeline --manifest data/hq/ultravideo/manifest.csv --out data/pairs/uv_2k_x2 \
      --gt-short 1440 --scale 2 --variants 2
  .venv-iqa/bin/python -m adup.degrade.pipeline --sequences data/hq/sequences/train.jsonl --out data/pairs/seq_2k_x2 \
      --gt-short 1440 --scale 2
"""

import argparse
import json
import os
import random
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd

from adup.degrade.overlays import draw_text, plan_text
from adup.media import detail, max_crop, probe, read_frames, stream_frames      # noqa: F401 (re-exported)
from adup.ugc.look import LOOK_STATS, apply_look, match_look, real_look_targets, sample_look, tone_lut
from adup.ugc.render import ShotRenderer

os.environ.setdefault("SVT_LOG", "1")   # SVT-AV1 prints its config banner to stderr unless told otherwise

# height / width of each GT aspect ratio
ASPECTS = {"9:16": 16 / 9, "4:5": 5 / 4, "1:1": 1.0, "16:9": 9 / 16}


# ---------------------------------------------------------------- ffmpeg helpers






class Writer:
    def __init__(self, path, w, h, fps, codec_args):
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", *codec_args, "-pix_fmt", "yuv420p",
                                   "-fps_mode", "passthrough", path], stdin=subprocess.PIPE)
        self.path = path

    def write(self, frame):
        self.p.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait():
            raise RuntimeError(f"ffmpeg failed writing {self.path}")


def transcode(src, dst, vf, codec_args):
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", vf, *codec_args,
           "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-an", dst]
    subprocess.run(cmd, check=True)


def platform_codec(codec, crf, maxrate_k, keyint):
    """Encoder settings per platform family. H.264 mirrors the BVC / x264 streams measured on TikTok and Meta; VP9 and
    AV1 CRF ranges were chosen to reproduce the bits-per-pixel of YouTube Shorts' 720p / 1080p rungs."""
    if codec == "h264":
        return ["-c:v", "libx264", "-preset", "medium", "-profile:v", "high", "-crf", f"{crf}",
                "-maxrate", f"{maxrate_k}k", "-bufsize", f"{2 * maxrate_k}k", "-g", f"{keyint}", "-bf", "3", "-refs", "4"]
    if codec == "hevc":
        return ["-c:v", "libx265", "-preset", "medium", "-crf", f"{crf}", "-tag:v", "hvc1", "-x265-params",
                f"log-level=error:keyint={keyint}:vbv-maxrate={maxrate_k}:vbv-bufsize={2 * maxrate_k}"]
    if codec == "vp9":
        return ["-c:v", "libvpx-vp9", "-crf", f"{crf}", "-b:v", "0", "-deadline", "good", "-cpu-used", "4",
                "-row-mt", "1", "-g", f"{keyint}"]
    if codec == "av1":
        return ["-c:v", "libsvtav1", "-preset", "8", "-crf", f"{crf}", "-g", f"{keyint}", "-svtav1-params", "tune=0"]
    raise ValueError(codec)


LOSSLESS_H264 = ["-c:v", "libx264", "-preset", "veryfast", "-qp", "0"]
GT_H264 = ["-c:v", "libx264", "-preset", "medium", "-crf", "10"]


def split_at(src, pattern, cuts):
    """Split src at frame indices `cuts` into pattern % i files, frame-exact and lossless, in one pass."""
    if not cuts:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, *LOSSLESS_H264, "-pix_fmt", "yuv420p",
                        "-fps_mode", "passthrough", "-an", pattern % 0], check=True)
        return
    keys = "+".join(f"eq(n,{c})" for c in cuts)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, *LOSSLESS_H264, "-pix_fmt", "yuv420p",
                    "-force_key_frames", f"expr:{keys}", "-fps_mode", "passthrough", "-an", "-f", "segment",
                    "-segment_frames", ",".join(map(str, cuts)), "-reset_timestamps", "1", pattern], check=True)


# ---------------------------------------------------------------- stage 0: GT geometry and quality gate
def gt_geometry(aspect, gt_short, scale):
    """(gt_w, gt_h, lq_w, lq_h) with the given short side; LQ sides are even and GT = LQ * scale."""
    r = ASPECTS[aspect]
    lq_short = int(round(gt_short / scale)) // 2 * 2
    lq_w, lq_h = (lq_short, int(round(lq_short * r)) // 2 * 2) if r >= 1 else (int(round(lq_short / r)) // 2 * 2, lq_short)
    return int(round(lq_w * scale)) // 2 * 2, int(round(lq_h * scale)) // 2 * 2, lq_w, lq_h




def feasible_aspects(sources, gt_short, scale, weights):
    """Aspects whose GT can be cut from every source without upscaling."""
    ok = {}
    for a, wgt in weights.items():
        gw, gh, _, _ = gt_geometry(a, gt_short, scale)
        if all(max_crop(sw, sh, gw, gh)[0] >= gw for sw, sh in sources):
            ok[a] = wgt
    return ok




def downup_psnr(gray, tile):
    """x2 down-up PSNR on the most detailed tile: low = real detail at this resolution, high = soft / upscaled."""
    y = gray.astype(np.float32)
    h, w = y.shape
    u = cv2.resize(cv2.resize(y, (w // 2, h // 2), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_CUBIC)
    best = max((cv2.Laplacian(y[i:i + tile, j:j + tile], cv2.CV_32F).var(), i, j)
               for i in range(0, h - tile + 1, tile // 2) for j in range(0, w - tile + 1, tile // 2))
    _, i, j = best
    mse = float(np.mean((y[i:i + tile, j:j + tile] - u[i:i + tile, j:j + tile]) ** 2))
    return 10 * np.log10(255 ** 2 / max(mse, 1e-6))


def crop_filter(src, start, n, sw, sh, gw, gh, rng, candidates=7):
    """Crop the largest region of the GT aspect ratio where a middle frame has the most texture, then downscale to GT
    size. Random crops of landscape footage often land on empty or out-of-focus background."""
    cw, ch = max_crop(sw, sh, gw, gh)
    mid = read_frames(src, f"select=eq(n\\,{start + n // 2})", sw, sh, 1)[0]
    gray = cv2.cvtColor(mid, cv2.COLOR_RGB2GRAY)
    if cw < sw:
        y, xs = (sh - ch) // 2 // 2 * 2, sorted({int(v) // 2 * 2 for v in np.linspace(0, sw - cw, candidates)})
        scored = [(detail(cv2.resize(gray[y:y + ch, x:x + cw], (gw, gh), interpolation=cv2.INTER_AREA)), x) for x in xs]
        best = max(scored)[1]
        x = min(max(best + rng.randint(-cw // 20, cw // 20) // 2 * 2, 0), sw - cw)
    else:
        x, ys = 0, sorted({int(v) // 2 * 2 for v in np.linspace(0, sh - ch, candidates)})
        scored = [(detail(cv2.resize(gray[yy:yy + ch, :cw], (gw, gh), interpolation=cv2.INTER_AREA)), yy) for yy in ys]
        y = max(scored)[1]
    return f"crop={cw}:{ch}:{x}:{y},scale={gw}:{gh}:flags=lanczos"


def shot_filter(shot, geom_vf, seq_fps, select=None):
    vf = f"trim=start_frame={shot['start']},setpts=PTS-STARTPTS"
    if seq_fps and abs(seq_fps - shot["src_fps"]) > 0.01:
        vf += f",fps={seq_fps}"
    if select:                                   # pick frames before the (expensive) crop + scale
        vf += ",select=" + "+".join(f"eq(n\\,{k})" for k in select)
    return f"{vf},{geom_vf}"


def gate_stats(shot, geom_vf, gw, gh, seq_fps):
    """Luma, detail and effective resolution of 3 frames of the shot at GT resolution."""
    ks = [shot["frames"] // 4, shot["frames"] // 2, 3 * shot["frames"] // 4]
    frames = list(stream_frames(shot["src"], shot_filter(shot, geom_vf, seq_fps, ks), gw, gh, 3))
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    tile = max(min(gw, gh) // 4, 128)
    return {"mean_luma": float(np.mean([g.mean() for g in grays])), "detail": float(np.median([detail(g) for g in grays])),
            "downup_psnr": float(np.median([downup_psnr(g, tile) for g in grays]))}


def gate_ok(stats, c):
    return (stats["detail"] >= c["gt_min_detail"] and c["gt_luma_range"][0] <= stats["mean_luma"] <= c["gt_luma_range"][1]
            and stats["downup_psnr"] <= c["gt_max_downup_psnr"])


# ---------------------------------------------------------------- stage 0b: edit transitions
def transition_effect(f, kind, p, direction):
    """p in (0, 1]: strength of the effect at this frame (1 = at the cut)."""
    x = f.astype(np.float32)
    if kind == "dip_black":
        x *= 1 - p
    elif kind == "dip_white":
        x = x * (1 - p) + 255 * p
    elif kind == "whip":
        k = max(int(p * f.shape[1] * 0.12) // 2 * 2 + 1, 3)
        x = cv2.filter2D(x, -1, np.full((1, k), 1 / k, np.float32))
        x = np.roll(x, int(direction * p * f.shape[1] * 0.08), axis=1)
    return np.clip(x, 0, 255).astype(np.uint8)


def sequence_frames(shots, transitions, renderers, rng):
    """Yield the GT frames of the whole sequence with transitions applied between shots."""
    direction = rng.choice([-1, 1])
    held = []                                     # dissolve: tail of the previous shot, blended into this shot's head
    for i, (shot, ren) in enumerate(zip(shots, renderers)):
        t_in = transitions[i - 1] if i > 0 else None
        t_out = transitions[i] if i < len(shots) - 1 else None
        n = shot["frames"]
        prev_tail, held = held, []
        vel = ren.velocity()
        for j, f in enumerate(ren.frames()):
            if t_out and t_out["type"] == "dissolve" and j >= n - t_out["frames"]:
                held.append(f)
                continue
            if t_out and t_out["type"] in ("dip_black", "dip_white", "whip"):
                k = t_out["frames"] // 2
                if j >= n - k:
                    f = transition_effect(f, t_out["type"], (j - (n - k) + 1) / (k + 1), direction)
            if t_in and t_in["type"] == "dissolve" and j < len(prev_tail):
                a = (j + 1) / (len(prev_tail) + 1)
                f = (prev_tail[j].astype(np.float32) * (1 - a) + f.astype(np.float32) * a + 0.5).astype(np.uint8)
            if t_in and t_in["type"] in ("dip_black", "dip_white", "whip"):
                k = t_in["frames"] - t_in["frames"] // 2
                if j < k:
                    f = transition_effect(f, t_in["type"], 1 - j / (k + 1), -direction)
            yield f, vel[min(j, len(vel) - 1)], ren.camera_captured


def shot_ranges(shots, transitions):
    """Output frame range [start, end) of every shot; a dissolve's blended frames count as the head of the next shot."""
    ranges, pos = [], 0
    for i, s in enumerate(shots):
        n = s["frames"] - (transitions[i]["frames"] if i < len(shots) - 1 and transitions[i]["type"] == "dissolve" else 0)
        ranges.append((pos, pos + n))
        pos += n
    return ranges


# ---------------------------------------------------------------- stage 1: capture / ISP
# Sampling ranges, calibrated against real TikTok and Meta ads at matched 720x1280 LQ size (adup.analysis.compare;
# history in docs/ugc_degradations.md §6). v1 (configs/degradation/v1.json) was too harsh; v3 (configs/degradation/v3.json)
# matched TikTok with the old single-style captions. v4 adds realistic text, the codec mix and aspect ratios, and softens
# sharpening / re-upload a little; it matches both platforms on DOVER, noise, overshoot and effective resolution.
DEFAULT_CONFIG = {
    "blur_prob": 0.2, "blur_sigma": [0.3, 1.2],
    "noise_prob": 0.25, "noise_sigma": [1.0, 4.0],
    "denoise_prob": 0.4, "denoise_strength": [20, 50],
    "sharpen_prob": 0.7, "sharpen_amount": [0.4, 1.2], "sharpen_sigma": [0.8, 2.0],
    # handheld motion blur along the virtual camera's velocity: blur length = speed x shutter fraction of a frame
    "motion_blur_prob": 0.6, "shutter": [0.2, 1.0],
    "edit_prob": 0.7, "edit_crf": [16, 22],
    "resize_flags": ["bicubic", "bilinear", "area", "lanczos"],
    "platform_crf": [20, 27], "platform_maxrate_k": [1200, 1600, 2000, 3000, 4000],
    "keyint": [60, 110, 150, 250],
    "reupload_prob": 0.15, "reupload_up": [1.0, 1.5, 2.0], "reupload_crf": [22, 28],
    # platform pre-processing before the platform encode (ByteDance publishes pre-encode adaptive sharpening,
    # RPO-AdaSharp; KVQ lists pre-processing / enhancement among short-video workflows): at LQ size, after the resize
    "platform_pre": {"none": 0.6, "sharpen": 0.2, "denoise": 0.2}, "pre_sharpen": [0.3, 0.8], "pre_denoise": [1.5, 4.0],
    # v4: platform codec mix (H.264 for TikTok / Meta; VP9 / AV1 for YouTube and Meta's in-app AV1; HEVC for iOS and
    # Chinese platforms). Non-H.264 CRF ranges hit YouTube Shorts' bpp q10-q90 at 720p (see docs/ugc_degradations.md).
    "platform_codecs": {"h264": 0.6, "vp9": 0.15, "av1": 0.15, "hevc": 0.1},
    "vp9_crf": [36, 50], "av1_crf": [40, 55], "hevc_crf": [24, 32],
    # GT aspect ratios: both orientations. An aspect is only used when every shot can supply it without upscaling
    # (9:16 at 1440+ short side needs 8K landscape or native portrait sources).
    "gt_aspects": {"9:16": 0.45, "16:9": 0.35, "4:5": 0.1, "1:1": 0.1},
    # GT quality gate at GT resolution: exposure, texture, and effective resolution (x2 down-up PSNR on the most
    # detailed tile; above the threshold the "4K" source has no real detail at GT size).
    "gt_min_detail": 12.0, "gt_luma_range": [30, 230], "gt_max_downup_psnr": 38.0,
    # LQ short side when --scale auto (TikTok serves 720 and 576, some 540)
    "lq_short": {"720": 0.6, "576": 0.3, "540": 0.1},
    # burned-in text (OCR on real ads: ~95% have text)
    "text_prob": 0.95, "caption_prob": 0.85, "hook_prob": 0.5, "sticker_prob": 0.45, "fine_print_prob": 0.2,
    # UGC look of the GT (adup.ugc; ranges calibrated against real ads with adup.analysis.ugc_look, see configs/ugc)
    "ugc": {
        "camera": {"shake_prob": 0.5, "shake_rms": [0.15, 1.2], "walk_prob": 0.15, "drift_prob": 0.5, "drift_pct": [0.3, 2.0],
                   "push_prob": 0.3, "push_zoom": [1.03, 1.12], "base_zoom": [1.0, 1.0], "roll_per_pct": 0.15},
        "still_camera": {"shake_prob": 0.5, "drift_prob": 0.8, "push_prob": 0.8, "push_zoom": [1.05, 1.2]},
        # look: match_look towards a real ad shot's colour statistics (limits m_*); the ranges without m_ are for
        # sample_look, used only when no real statistics exist
        "look": {"look_prob": 0.9, "m_exposure": [0.75, 1.5], "m_white": [0.78, 1.0], "m_shadow_lift": [0.0, 0.1],
                 "m_saturation": [0.7, 2.0], "m_warmth": [-0.06, 0.1], "contrast": [-0.05, 0.12], "tint": [-0.01, 0.02],
                 "exposure": [1.03, 1.25], "shadow_lift": [0.0, 0.05], "highlight_rolloff": [0.5, 1.0], "knee": [0.6, 0.75],
                 "white": [0.86, 0.98], "saturation": [1.1, 1.6], "warmth": [-0.01, 0.06]},
    },
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
        "shutter": u("shutter") if rng.random() < c["motion_blur_prob"] else 0.0,
    }


def skin_mask(rgb):
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    m = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))
    return cv2.GaussianBlur(m, (0, 0), 3)[..., None].astype(np.float32) / 255


def motion_blur(x, vx, vy, shutter):
    """Linear blur along the frame's camera velocity (GT pixels / frame) over the shutter fraction."""
    length = np.hypot(vx, vy) * shutter
    if length < 1.5:
        return x
    k = int(np.ceil(length)) | 1
    kern = np.zeros((k, k), np.float32)
    c = k // 2
    dx, dy = vx / np.hypot(vx, vy), vy / np.hypot(vx, vy)
    for t in np.linspace(-length / 2, length / 2, max(int(length * 2), 3)):
        kern[int(round(c + t * dy)), int(round(c + t * dx))] += 1
    return cv2.filter2D(x, -1, kern / kern.sum())


def apply_capture(f, p, nprng, noise_gain=1.0, vel=(0.0, 0.0)):
    """Camera / ISP degradations on one frame. Blur / sharpen radii are defined at 1080p width and scale with the frame;
    noise_gain compensates for the GT -> LQ downscale averaging noise away (calibrated at a 1.5x downscale)."""
    s = min(f.shape[:2]) / 1080
    x = f.astype(np.float32)
    if p.get("shutter", 0) > 0:
        x = motion_blur(x, vel[0], vel[1], p["shutter"])
    if p["blur_sigma"] > 0:
        x = cv2.GaussianBlur(x, (0, 0), p["blur_sigma"] * s)
    if p["noise_sigma"] > 0:
        cv2.setRNGSeed(int(nprng.integers(2 ** 31)))
        luma = np.empty(x.shape[:2], np.float32)
        chroma = np.empty(x.shape, np.float32)
        cv2.randn(luma, 0, p["noise_sigma"] * noise_gain)
        cv2.randn(chroma, 0, p["noise_sigma"] * noise_gain * 0.5)
        x = x + luma[..., None] + chroma
    x = np.clip(x, 0, 255)
    if p["denoise"] != "none":
        u8 = x.astype(np.uint8)
        smooth = cv2.bilateralFilter(u8, min(int(9 * s) | 1, 15), p["denoise_strength"], 7 * s).astype(np.float32)
        if p["denoise"] == "bilateral":
            x = smooth
        else:
            m = skin_mask(u8)
            x = x * (1 - m) + smooth * m
    if p["sharpen_amount"] > 0:
        x = x + p["sharpen_amount"] * (x - cv2.GaussianBlur(x, (0, 0), p["sharpen_sigma"] * s))
    return np.clip(x, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- stages 2-4
def pick_weighted(rng, weights):
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def codec_crf(rng, c, codec):
    return rng.randint(*c["platform_crf" if codec == "h264" else f"{codec}_crf"])


def sample_chain(rng, c):
    codec, reupload_codec = pick_weighted(rng, c["platform_codecs"]), pick_weighted(rng, c["platform_codecs"])
    return {
        "edit_export": rng.random() < c["edit_prob"],
        "edit_crf": rng.randint(*c["edit_crf"]),
        "resize_flags": rng.choice(c["resize_flags"]),
        "platform_codec": codec,
        "platform_crf": codec_crf(rng, c, codec),
        "platform_maxrate_k": rng.choice(c["platform_maxrate_k"]),
        "keyint": rng.choice(c["keyint"]),
        "reupload": rng.random() < c["reupload_prob"],
        "reupload_up": rng.choice(c["reupload_up"]),
        "platform_pre": pick_weighted(rng, c["platform_pre"]),
        "pre_amount": rng.uniform(*c["pre_sharpen"]),
        "pre_denoise": rng.uniform(*c["pre_denoise"]),
        "reupload_codec": reupload_codec,
        "reupload_crf": rng.randint(*c["reupload_crf"]) if reupload_codec == "h264" else codec_crf(rng, c, reupload_codec),
    }


def finish_variant(edit_path, out_path, chain, lq_w, lq_h, seconds, tmpdir):
    """Platform transcode (+ optional re-upload) of the edit export -> LQ file."""
    platform = os.path.join(tmpdir, "platform.mp4")
    rate = (chain["platform_maxrate_k"], chain["keyint"])
    vf = f"scale={lq_w}:{lq_h}:flags={chain['resize_flags']}"
    pre = chain.get("platform_pre", "none")
    if pre == "sharpen":
        vf += f",unsharp=5:5:{chain['pre_amount']:.2f}:5:5:0"
    elif pre == "denoise":
        d = chain["pre_denoise"]
        vf += f",hqdn3d={d:.2f}:{d * 0.75:.2f}:{d * 1.5:.2f}:{d * 1.1:.2f}"
    transcode(edit_path, platform, vf, platform_codec(chain["platform_codec"], chain["platform_crf"], *rate))
    platform_kbps = os.path.getsize(platform) * 8 / 1000 / seconds
    final, final_codec = platform, chain["platform_codec"]
    if chain["reupload"]:
        up_w, up_h = int(lq_w * chain["reupload_up"]) // 2 * 2, int(lq_h * chain["reupload_up"]) // 2 * 2
        mid = os.path.join(tmpdir, "reupload_mid.mp4")
        transcode(platform, mid, f"scale={up_w}:{up_h}:flags=bicubic",
                  platform_codec(chain["reupload_codec"], chain["reupload_crf"], *rate))
        final = os.path.join(tmpdir, "reupload.mp4")
        transcode(mid, final, f"scale={lq_w}:{lq_h}:flags={chain['resize_flags']}",
                  platform_codec(chain["reupload_codec"], chain["reupload_crf"], *rate))
        final_codec = chain["reupload_codec"]
    stored_as = final_codec
    if final_codec in ("vp9", "av1"):
        decoded = os.path.join(tmpdir, "decoded.mp4")
        transcode(final, decoded, "null", LOSSLESS_H264)
        final, stored_as = decoded, "h264_lossless"
    os.replace(final, out_path)
    return {"stored_as": stored_as, "platform_kbps": round(platform_kbps, 1)}


# ---------------------------------------------------------------- one pair
def process_sequence(seq, out_dir, gt_short, scale, variants, seed, config):
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    if scale == "auto":                     # real inputs are 540-720p: sample the LQ short side, scale = GT / LQ
        lq_short = int(pick_weighted(rng, {int(k): v for k, v in config["lq_short"].items()}))
        scale = gt_short / lq_short
    shots, transitions, fps = seq["shots"], seq["transitions"], seq["fps"]
    if seq.get("aspect"):
        aspect = seq["aspect"]
    else:
        sizes = [probe(s["src"])[:2] for s in shots]
        aspects = feasible_aspects(sizes, gt_short, scale, config["gt_aspects"])
        if not aspects:
            raise ValueError(f"no aspect ratio fits GT short side {gt_short} in sources {sizes}")
        aspect = pick_weighted(rng, aspects)
    gw, gh, lw, lh = gt_geometry(aspect, gt_short, scale)
    gates = []
    if not seq.get("pregated"):                  # clip-level gate already applied by the director otherwise
        for s in shots:
            if s.get("kind", "video") != "video":
                continue
            sw, sh = probe(s["src"])[:2]
            vf = crop_filter(s["src"], s["start"], int(s["frames"] * s["src_fps"] / fps), sw, sh, gw, gh, rng)
            stats = gate_stats(s, vf, gw, gh, fps)
            if not gate_ok(stats, config):
                raise ValueError(f"GT rejected ({os.path.basename(s['src'])}): " +
                                 ", ".join(f"{k}={v:.1f}" for k, v in stats.items()))
            gates.append(stats)
    ugc = config["ugc"]
    renderers = [ShotRenderer(s, gw, gh, fps, nrng, ugc) for s in shots]
    if "look" in seq:
        look = seq["look"]
    elif nrng.random() >= ugc["look"]["look_prob"]:
        look = None
    elif (targets := real_look_targets()) is not None:
        ref = next((r for r in renderers if r.kind != "slide"), renderers[0])
        look = match_look(ref.preview, dict(zip(LOOK_STATS, targets[nrng.integers(len(targets))])), nrng, ugc["look"])
    else:
        look = sample_look(nrng, {**ugc["look"], "look_prob": 1.0})
    lut = tone_lut(look) if look else None
    ranges = shot_ranges(shots, transitions)
    n = ranges[-1][1]
    items, text_meta = plan_text(n, fps, gw, gh, rng, config)

    os.makedirs(out_dir, exist_ok=True)
    meta = {"id": seq["id"], "purpose": seq.get("purpose"), "fps": fps, "frames": n, "gt_size": [gw, gh],
            "lq_size": [lw, lh], "scale": scale, "aspect": aspect, "gate": gates, "look": look,
            "render": [r.info for r in renderers], "text": text_meta,
            "config": config, "variants": []}
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        caps = [sample_capture(rng, config) for _ in range(variants)]
        chains = [sample_chain(rng, config) for _ in range(variants)]
        nprngs = [np.random.default_rng(rng.randint(0, 2 ** 31)) for _ in range(variants)]
        gt_w = Writer(os.path.join(out_dir, "gt.mp4"), gw, gh, fps, GT_H264)
        edits = [Writer(os.path.join(tmp, f"edit_{k}.mp4"), gw, gh, fps,
                        ["-c:v", "libx264", "-preset", "fast", "-crf", str(ch["edit_crf"] if ch["edit_export"] else 8)])
                 for k, ch in enumerate(chains)]
        noise_gain = (gw / lw) / 1.5
        with ThreadPoolExecutor(max_workers=variants + 1) as pool:      # variants in parallel (cv2 / numpy drop the GIL)
            for i, (f, vel, camera) in enumerate(sequence_frames(shots, transitions, renderers, rng)):
                if camera:                          # screen recordings and text slides are digital: no phone look / ISP
                    f = apply_look(f, look, lut)
                draw_text(f, i, items)
                jobs = [pool.submit(apply_capture, f, cap, nprng, noise_gain, vel) if camera else None
                        for cap, nprng in zip(caps, nprngs)]
                gt_w.write(f)
                for w, j in zip(edits, jobs):
                    w.write(j.result() if j is not None else f)
        gt_w.close()
        for w in edits:
            w.close()
        for k, (cap, chain) in enumerate(zip(caps, chains)):
            out = os.path.join(out_dir, f"lq_{k}.mp4")
            vt = os.path.join(tmp, f"v{k}")
            os.makedirs(vt)
            info = finish_variant(os.path.join(tmp, f"edit_{k}.mp4"), out, chain, lw, lh, n / fps, vt)
            meta["variants"].append({"file": out, "capture": cap, "chain": chain, **info})

    if len(shots) > 1:
        cuts = [a for a, _ in ranges[1:]]
        sd = os.path.join(out_dir, "shots")
        os.makedirs(sd, exist_ok=True)
        for name in ["gt"] + [f"lq_{k}" for k in range(variants)]:
            split_at(os.path.join(out_dir, f"{name}.mp4"), os.path.join(sd, f"shot_%03d_{name}.mp4"), cuts)
        meta["cut_method"] = {"type": "composition", "note": "boundaries are exact, known from the sequence spec; "
                              "a dissolve's blended frames are the head of the following shot"}
        meta["shots"] = [{"index": i, "start_frame": a, "end_frame": b, "frames": b - a,
                          "source": {k: s[k] for k in ("kind", "src", "clip_id", "source_id", "category", "src_fps", "start",
                                                       "layout") if k in s},
                          "source_frames_used": s["frames"],
                          "transition_in": transitions[i - 1] if i else None,
                          "transition_out": transitions[i] if i < len(shots) - 1 else None,
                          "files": {name: os.path.join(sd, f"shot_{i:03d}_{name}.mp4")
                                    for name in ["gt"] + [f"lq_{k}" for k in range(variants)]}}
                         for i, (s, (a, b)) in enumerate(zip(shots, ranges))]
    else:
        meta["source"] = {k: shots[0][k] for k in ("kind", "src", "src_fps", "start") if k in shots[0]}
    meta["spec"] = seq
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return meta


def clip_specs(manifest, max_frames):
    """Single-shot specs for a manifest of clips, at each clip's native frame rate."""
    specs = []
    for f in pd.read_csv(manifest)["file"]:
        _, _, fps, nb = probe(f)
        fps = round(fps, 3)
        specs.append({"id": os.path.splitext(os.path.basename(f))[0], "purpose": "clip", "fps": fps,
                      "shots": [{"src": f, "src_fps": fps, "start": 0, "frames": min(max_frames, nb or max_frames)}],
                      "transitions": []})
    return specs


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="CSV with a `file` column of single-shot HQ clips")
    src.add_argument("--sequences", help="JSONL of multi-shot specs from adup.shots.compose")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt-short", type=int, default=1440, help="GT short side (1440 = 2K, 2160 = 4K)")
    ap.add_argument("--scale", default="2", help="GT / LQ factor, or 'auto' to sample the LQ short side (config lq_short)")
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=150, help="--manifest only: frames per clip")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", help="JSON file overriding DEFAULT_CONFIG")
    args = ap.parse_args()
    config = {**DEFAULT_CONFIG, **(json.load(open(args.config)) if args.config else {})}
    args.scale = args.scale if args.scale == "auto" else float(args.scale)
    specs = clip_specs(args.manifest, args.max_frames) if args.manifest else [json.loads(l) for l in open(args.sequences)]
    if args.limit:
        specs = specs[:args.limit]
    for i, seq in enumerate(specs):
        out_dir = os.path.join(args.out, seq["id"])
        if os.path.exists(os.path.join(out_dir, "meta.json")):
            continue
        try:
            process_sequence(seq, out_dir, args.gt_short, args.scale, args.variants, args.seed * 100003 + i, config)
            print(f"[{i + 1}/{len(specs)}] {seq['id']}", flush=True)
        except Exception as e:
            print(f"[{i + 1}/{len(specs)}] {seq['id']} failed: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
