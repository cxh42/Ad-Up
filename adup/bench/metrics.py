"""Frame-level metrics shared by the benchmark and the calibration tools (luma PSNR / SSIM, masked PSNR, text CER)."""

import re

import cv2
import numpy as np


def luma(f):
    return cv2.cvtColor(f, cv2.COLOR_RGB2YCrCb)[..., 0].astype(np.float32)


def psnr(x, y, mask=None):
    """PSNR of two luma arrays, optionally over a boolean mask (None if the mask is empty)."""
    d = (x - y) ** 2
    if mask is not None:
        if not mask.any():
            return None
        d = d[mask]
    return 10 * np.log10(255 ** 2 / max(float(np.mean(d)), 1e-10))


def ssim(x, y):
    """SSIM of two luma arrays (Gaussian window 11, sigma 1.5)."""
    mu_x, mu_y = cv2.GaussianBlur(x, (11, 11), 1.5), cv2.GaussianBlur(y, (11, 11), 1.5)
    sxx = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x ** 2
    syy = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y ** 2
    sxy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(np.mean((2 * mu_x * mu_y + c1) * (2 * sxy + c2) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))))


def norm_text(s):
    """Case-insensitive letters, digits, $ and %: what OCR reads reliably and what matters in ads (words, prices)."""
    return re.sub(r"[^a-z0-9$%]", "", s.lower())


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(read, truth):
    t = norm_text(truth)
    return edit_distance(norm_text(read), t) / max(len(t), 1)
