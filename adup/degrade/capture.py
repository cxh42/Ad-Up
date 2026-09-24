"""Degradation stage 1, capture / ISP: what the phone does to the picture before anything is encoded.

Applied per frame at GT size to camera-captured shots only (screen recordings and text slides are digital):
motion blur along the virtual camera's velocity, defocus blur, sensor noise, ISP denoise or skin smoothing, and ISP
unsharp-mask sharpening. Parameters are sampled once per LQ variant (config section `degrade`) and held fixed over
the clip, like one phone's settings.
"""

import cv2
import numpy as np


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
