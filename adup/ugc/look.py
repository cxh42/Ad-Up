"""Phone "look" for GT frames: white balance, exposure, HDR-style tone curve and saturation of phone video.

Real TikTok / Meta ads are warmer, more saturated, brighter, with lifted shadows and rolled-off highlights than the
cinematic UltraVideo sources (adup.analysis.ugc_look). This is a style, not a degradation: it is applied to the GT, so
the model learns to keep it. One look per sequence (one creator, one phone).

match_look (default) is distribution matching: it draws the colour statistics of one real ad shot as the target
(saturation, warmth, brightness, shadows, highlights together, keeping their correlations) and sets the look so the
source moves towards that target, within limits, so already warm / saturated footage is not pushed further.
sample_look draws independent random parameters (used when no real statistics are available).
"""

import cv2
import numpy as np
import pandas as pd

from adup.paths import LOOK_TABLE

LOOK_STATS = ["saturation", "warmth", "luma_mean", "luma_p5", "luma_p95"]


def real_look_targets():
    if not LOOK_TABLE.exists():
        return None
    d = pd.read_csv(LOOK_TABLE)
    d = d[d.group.isin(["tiktok", "meta"]) & (d.static_frac < 0.5)]
    return d[LOOK_STATS].dropna().to_numpy()


def match_look(preview, target, rng, c, iters=6):
    """Look that moves `preview` (a small RGB frame of the sequence) towards the colour statistics `target` (dict).

    Starts from ratio estimates, then corrects exposure, saturation and warmth multiplicatively by re-measuring the
    styled preview, since the tone curve and the saturation mix interact. Every parameter stays within its limits."""
    from adup.analysis.ugc_look import colour                   # local import: analysis imports media helpers
    clip = lambda v, lo_hi: float(np.clip(v, *lo_hi))
    mask = np.zeros(preview.shape[:2], bool)
    src = colour(preview, mask)
    look = {"exposure": clip(target["luma_mean"] / max(src["luma_mean"], 1), c["m_exposure"]),
            "shadow_lift": clip((target["luma_p5"] - src["luma_p5"]) / 255 * 1.5, c["m_shadow_lift"]),
            "highlight_rolloff": 1.0, "knee": float(rng.uniform(0.65, 0.8)), "white": 1.0,
            "contrast": float(rng.uniform(*c["contrast"])), "tint": float(rng.uniform(*c["tint"])),
            "saturation": clip(target["saturation"] / max(src["saturation"], 1), c["m_saturation"]),
            "warmth": clip((target["warmth"] - src["warmth"]) / 220, c["m_warmth"])}
    for _ in range(iters):
        got = colour(apply_look(preview, look), mask)
        look["exposure"] = clip(look["exposure"] * target["luma_mean"] / max(got["luma_mean"], 1), c["m_exposure"])
        look["saturation"] = clip(look["saturation"] * target["saturation"] / max(got["saturation"], 1), c["m_saturation"])
        look["warmth"] = clip(look["warmth"] + (target["warmth"] - got["warmth"]) / 220, c["m_warmth"])
        look["white"] = clip(look["white"] * target["luma_p95"] / max(got["luma_p95"], 1), c["m_white"])
        look["shadow_lift"] = clip(look["shadow_lift"] + (target["luma_p5"] - got["luma_p5"]) / 255, c["m_shadow_lift"])
    look["target"] = {k: round(float(target[k]), 1) for k in LOOK_STATS}
    return look


def sample_look(rng, c):
    u = lambda k: float(rng.uniform(*c[k]))
    if rng.random() >= c["look_prob"]:
        return None
    return {"exposure": u("exposure"), "shadow_lift": u("shadow_lift"), "highlight_rolloff": u("highlight_rolloff"),
            "knee": u("knee"), "white": u("white"), "contrast": u("contrast"), "saturation": float(np.exp(rng.uniform(*np.log(c["saturation"])))),
            "warmth": u("warmth"), "tint": u("tint")}


def tone_lut(look):
    """Per-channel 256-entry LUTs: white balance and exposure, a soft shoulder that rolls highlights off towards the
    white level (mid-tones untouched), lifted shadows, and a mild S-curve."""
    x = np.arange(256, dtype=np.float32) / 255
    w = look.get("white", 1.0)
    k = min(look.get("knee", 0.8), w - 0.12)
    luts = []
    for g in [1 + look["warmth"], 1 + look["tint"], 1 - look["warmth"]]:          # R, G, B
        y = np.clip(x * g * look["exposure"], 0, None)
        soft = k + (w - k) * np.tanh((y - k) / (w - k))
        y = np.where(y > k, (1 - look["highlight_rolloff"]) * np.minimum(y, w) + look["highlight_rolloff"] * soft, y)
        y = y + look["shadow_lift"] * (1 - y) ** 3
        y = y + look["contrast"] * (y - 0.5) * (1 - np.abs(2 * y - 1))
        luts.append(np.clip(y * 255 + 0.5, 0, 255).astype(np.uint8))
    return np.stack(luts, 1)


def apply_look(frame, look, lut=None):
    if look is None:
        return frame
    lut = tone_lut(look) if lut is None else lut
    out = cv2.merge([cv2.LUT(np.ascontiguousarray(frame[..., ch]), lut[:, ch]) for ch in range(3)])
    sat = look["saturation"]
    if abs(sat - 1) > 1e-3:                                  # scale chroma around neutral in YCrCb (uint8, multithreaded)
        y, cr, cb = cv2.split(cv2.cvtColor(out, cv2.COLOR_RGB2YCrCb))
        zero = np.zeros_like(cr)
        cr = cv2.addWeighted(cr, sat, zero, 0, 128 * (1 - sat))
        cb = cv2.addWeighted(cb, sat, zero, 0, 128 * (1 - sat))
        out = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2RGB)
    return out
