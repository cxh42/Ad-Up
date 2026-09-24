"""Virtual phone camera: handheld shake, drift, slow push-ins and punch-in framing, without ever upscaling the GT.

Each shot is read from the source at a working size of S x GT (S >= 1, bounded by how much larger the source crop is
than the GT), then every output frame is a similarity transform (zoom >= 1 relative to the working frame's GT-sized
window, translation, roll) of the working frame. Motion therefore always comes from real extra source pixels.

Shake is band-limited noise with a ~1/f spectrum in 0.5-6 Hz (physiological hand tremor sits at 1-3.5 Hz; walking adds
~2 Hz bob), drift is a slow random walk, and a push-in is a smooth zoom ramp. Amplitudes are sampled so that the
measured shake_rms / pan_speed / zoom_rate (adup.analysis.ugc_look) match real TikTok / Meta ads; see configs/ugc.
"""

import cv2
import numpy as np


def band_noise(rng, n, fps, lo, hi, slope=1.0):
    """Unit-RMS noise with power only in [lo, hi] Hz and amplitude ~ 1/f^slope."""
    m = 1 << int(np.ceil(np.log2(max(n, 8) * 2)))
    f = np.fft.rfftfreq(m, 1 / fps)
    amp = np.where((f >= lo) & (f <= hi), 1 / np.maximum(f, lo) ** slope, 0)
    spec = amp * np.exp(2j * np.pi * rng.random(len(f)))
    x = np.fft.irfft(spec, m)[:n]
    return x / (x.std() + 1e-9)


def plan_camera(rng, n, fps, cfg, headroom):
    """Per-frame (zoom, dx, dy, roll) and the working scale S. dx / dy are in % of GT width / height, roll in degrees.

    headroom: how many times larger than the GT the source crop is (never read the source above that scale)."""
    shake = rng.random() < cfg["shake_prob"]
    amp = float(np.exp(rng.uniform(*np.log(cfg["shake_rms"])))) if shake else 0.0        # % of width
    walk = shake and rng.random() < cfg["walk_prob"]
    drift = float(rng.uniform(*cfg["drift_pct"])) if rng.random() < cfg["drift_prob"] else 0.0
    push = float(rng.uniform(*cfg["push_zoom"])) if rng.random() < cfg["push_prob"] else 1.0
    z0 = float(rng.uniform(*cfg["base_zoom"]))

    t = np.arange(n) / fps
    dx = amp * band_noise(rng, n, fps, 0.5, 6.0)
    dy = amp * 0.8 * band_noise(rng, n, fps, 0.5, 6.0)
    roll = amp * cfg["roll_per_pct"] * band_noise(rng, n, fps, 0.5, 4.0)
    if walk:                                                    # vertical bob of walking, ~2 Hz
        dy += amp * 0.8 * np.sin(2 * np.pi * rng.uniform(1.6, 2.2) * t + rng.uniform(0, 6.3))
    if drift:
        dx += drift * band_noise(rng, n, fps, 0.02, 0.4, slope=2.0)
        dy += drift * 0.6 * band_noise(rng, n, fps, 0.02, 0.4, slope=2.0)
    zoom = z0 * np.linspace(1.0, push, n) if push != 1.0 else np.full(n, z0)
    if rng.random() < 0.5 and push != 1.0:                      # pull-out instead of push-in
        zoom = zoom[::-1].copy()

    # Absolute zoom z(t) is relative to the full source crop region. Translation / roll need a margin inside it, so the
    # visible window is z * (1 + reach) tighter; the source is read at S = max z_eff times GT size (never above headroom)
    # and each frame keeps rel = z_eff / S <= 1 output pixels per working pixel, i.e. only downscaling.
    reach = 2 * (np.abs(dx).max() + np.abs(dy).max()) / 100 + 1.2 * np.abs(np.radians(roll)).max()
    z_eff = zoom * (1 + reach)
    if z_eff.max() > headroom:                                  # not enough source pixels: shrink zoom, then motion
        if zoom.max() > headroom:                               # keep zoom >= 1: shrink the zoom excursion above 1
            zoom = 1 + (zoom - 1) * (headroom - 1) / (zoom.max() - 1)
        room = headroom / zoom.max() - 1
        k = min(1.0, room / reach) if reach > 0 else 1.0
        dx, dy, roll = dx * k, dy * k, roll * k
        z_eff = zoom * (1 + reach * k)
    S = float(z_eff.max())
    params = {"shake_rms": amp, "walk": walk, "drift": drift, "push": push, "base_zoom": z0, "working_scale": round(S, 4)}
    return np.stack([z_eff / S, dx, dy, roll], 1), S, params


def warp(frame, rel, dx, dy, roll, gw, gh):
    """GT frame from a working frame of size (gw*S, gh*S). rel <= 1 is output pixels per working pixel (the visible
    window is gw / rel working pixels wide); dx, dy in % of GT width / height; roll in degrees."""
    H, W = frame.shape[:2]
    cx, cy = W / 2 + dx / 100 * gw / rel, H / 2 + dy / 100 * gh / rel
    M = cv2.getRotationMatrix2D((cx, cy), roll, rel)
    M[0, 2] += gw / 2 - cx
    M[1, 2] += gh / 2 - cy
    return cv2.warpAffine(frame, M, (gw, gh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)
