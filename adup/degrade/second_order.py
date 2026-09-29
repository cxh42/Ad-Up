"""Second-order real-world video degradation (Real-ESRGAN / RealBasicVSR), streamed frame by frame.

    GT --stage 1--> --stage 2--> --final--> LQ
    stage 1   blur -> resize -> noise -> JPEG -> video compression        (RealBasicVSR's first order)
    stage 2   blur -> resize -> noise -> JPEG                              (second order)
    final     {video compression, [resize to the LQ size -> sinc filter]} in random order

This follows RealBasicVSR's training pipeline (and DOVE's finetune/configs/degradation.yaml, which copies it) with three
changes for offline video pairs:
  - every parameter is drawn once per LQ variant and held over the whole clip (temporal coherence); only noise is
    drawn per frame
  - video compression is a real ffmpeg encode with H.264 (libx264) or VP9 (libvpx-vp9); its bitrate is given in bits per
    pixel so that it means the same at every frame size (RealBasicVSR's 1e4-1e5 bit/frame on 256 px crops is 0.15-1.5)
  - optional UGC extras in stage 1, before the resize: motion blur along the virtual camera's velocity and phone-ISP
    unsharp-mask sharpening

Parameters come in two presets (config section `degrade`): `rbvsr`, RealBasicVSR's original ranges (light to unreadable;
the training default, as in Real-ESRGAN / RealBasicVSR / DOVE), and `calibrated`, fitted to real Meta ads (dev / test
sets and training ablations). A variant uses `rbvsr` with probability rbvsr_prob. Blur, sharpening and motion blur
lengths are in pixels of the frame they are applied to.
"""

import os

import cv2
import numpy as np

from adup.degrade.kernels import random_mixed_kernels
from adup.media import LOSSLESS_H264, Writer, stream_frames, transcode

os.environ.setdefault("SVT_LOG", "1")

INTERP = {"bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC, "area": cv2.INTER_AREA, "lanczos": cv2.INTER_LANCZOS4}
LOSSLESS_RGB = ["-c:v", "libx264rgb", "-preset", "veryfast", "-qp", "0"]      # exact intermediate between stages


# ---------------------------------------------------------------- sampling
def pick(rng, weights):
    """Key of a {key: weight} dict, drawn by weight."""
    keys = list(weights)
    return keys[int(rng.choice(len(keys), p=np.array(list(weights.values()), float) / sum(weights.values())))]


def uniform(rng, lo_hi):
    return float(rng.uniform(*lo_hi))


def sample_blur(rng, c):
    if c is None or rng.random() >= c.get("prob", 1.0):
        return None
    kind = pick(rng, c["kernels"])
    p = {"kernel": kind, "size": int(rng.choice(c["kernel_size"]))}
    if kind == "sinc":
        lo, hi = c.get("omega", [np.pi / 3, np.pi] if p["size"] < 13 else [np.pi / 5, np.pi])
        p["omega"] = float(rng.uniform(lo, hi))
    else:
        p["sigma_x"], p["sigma_y"] = uniform(rng, c["sigma"]), uniform(rng, c["sigma"])
        p["angle"] = uniform(rng, [-np.pi, np.pi])
        p["beta_gaussian"], p["beta_plateau"] = uniform(rng, c["beta_gaussian"]), uniform(rng, c["beta_plateau"])
    return p


def make_kernel(p):
    """The blur kernel of a sampled parameter set (random_mixed_kernels with every range collapsed to one value)."""
    one = lambda k, default=0.0: [p.get(k, default), p.get(k, default)]
    return random_mixed_kernels([p["kernel"]], [1], p["size"], one("sigma_x", 1), one("sigma_y", 1), one("angle"),
                                one("beta_gaussian", 1), one("beta_plateau", 1), one("omega", np.pi)).astype(np.float32)


def sample_resize(rng, c):
    mode = pick(rng, c["mode"])
    lo, hi = c["scale"]
    scale = 1.0 if mode == "keep" else uniform(rng, [1.0, hi] if mode == "up" else [lo, 1.0])
    return {"mode": mode, "scale": scale, "interp": pick(rng, c["interp"])}


def sample_noise(rng, c):
    if c is None or rng.random() >= c.get("prob", 1.0):
        return None
    kind = pick(rng, c["type"])
    level = uniform(rng, c["gaussian_sigma"] if kind == "gaussian" else c["poisson_scale"])
    return {"type": kind, "level": level, "gray": bool(rng.random() < c["gray_prob"])}


def sample_jpeg(rng, c):
    if c is None or rng.random() >= c.get("prob", 1.0):
        return None
    return int(round(uniform(rng, c["quality"])))


def sample_compress(rng, c):
    if c is None or rng.random() >= c.get("prob", 1.0):
        return None
    return {"codec": pick(rng, c["codec"]), "bpp": float(np.exp(uniform(rng, np.log(c["bpp"])))),
            "keyint": int(rng.choice(c["keyint"]))}


def sample_plan(rng, cfg):
    """All random choices of one LQ variant. cfg: config section `degrade`."""
    preset = "rbvsr" if rng.random() < cfg["rbvsr_prob"] else "calibrated"
    c = cfg[preset]
    s1, s2, fin = c["stage1"], c["stage2"], c["final"]
    extras = c.get("extras", {})
    return {
        "preset": preset,
        "stage1": {"motion_blur": uniform(rng, extras["shutter"]) if rng.random() < extras.get("motion_blur_prob", 0) else 0.0,
                   "blur": sample_blur(rng, s1.get("blur")),
                   "sharpen": ({"amount": uniform(rng, extras["sharpen_amount"]), "sigma": uniform(rng, extras["sharpen_sigma"])}
                               if rng.random() < extras.get("sharpen_prob", 0) else None),
                   "resize": sample_resize(rng, s1["resize"]), "noise": sample_noise(rng, s1.get("noise")),
                   "jpeg": sample_jpeg(rng, s1.get("jpeg")), "compress": sample_compress(rng, s1.get("compress"))},
        "stage2": {"blur": sample_blur(rng, s2.get("blur")), "resize": sample_resize(rng, s2["resize"]),
                   "noise": sample_noise(rng, s2.get("noise")), "jpeg": sample_jpeg(rng, s2.get("jpeg"))},
        "final": {"compress_first": bool(rng.random() < fin["compress_first_prob"]), "interp": pick(rng, fin["interp"]),
                  "sinc": sample_blur(rng, fin.get("sinc")), "compress": sample_compress(rng, fin["compress"])},
    }


# ---------------------------------------------------------------- per-frame operations
def even(x):
    return max(int(x) // 2 * 2, 16)


def motion_blur(x, vx, vy, shutter):
    """Linear blur along the frame's camera velocity (pixels / frame) over the shutter fraction."""
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


def add_noise(x, p, nprng):
    """Gaussian (sigma in 0-255 units) or Poisson (Real-ESRGAN's scaled shot noise) noise, colour or gray."""
    h, w = x.shape[:2]
    if p["type"] == "gaussian":
        # cv2.randn on an (h, w, 3) array puts a scalar sigma on the first channel only: draw a single-channel array
        c = 1 if p["gray"] else 3
        cv2.setRNGSeed(int(nprng.integers(2 ** 31)))
        n = np.empty((h, w * c), np.float32)
        cv2.randn(n, 0, p["level"])
        n = n.reshape(h, w, c)
    else:
        # Poisson(lam) - lam with lam = 256 * intensity: exact draws only where lam < 16 (dark pixels), elsewhere the
        # normal approximation of the same variance (skew <= 0.25); numpy's Poisson sampler is ~10x slower per frame
        base = np.clip(x, 0, 255) / 255
        if p["gray"]:
            base = cv2.cvtColor(base.astype(np.float32), cv2.COLOR_RGB2GRAY)[..., None]
        lam = (base * 256).astype(np.float32)
        cv2.setRNGSeed(int(nprng.integers(2 ** 31)))
        n = np.empty((lam.shape[0], lam.shape[1] * lam.shape[2]), np.float32)
        cv2.randn(n, 0, 1)
        n = n.reshape(lam.shape) * np.sqrt(lam)
        dark = lam < 16
        if dark.any():
            n[dark] = nprng.poisson(lam[dark]) - lam[dark]
        n *= 255 / 256 * p["level"]
    return x + n


def jpeg(x, quality):
    bgr = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)


class Stage:
    """The per-frame part of a stage: [motion blur] -> blur -> [sharpen] -> resize -> noise -> JPEG, frames of in_size
    (w, h) in, frames of out_size out. Kernels and sizes are fixed for the clip."""

    def __init__(self, p, in_size, nprng):
        self.p, self.nprng = p, nprng
        self.kernel = make_kernel(p["blur"]) if p.get("blur") else None
        self.sep = None                                # rank-1 kernels (isotropic Gaussian): two 1-D passes, ~3x faster
        if self.kernel is not None:
            u, s, vt = np.linalg.svd(self.kernel.astype(np.float64))
            if s[1] < 1e-7 * s[0]:
                self.sep = ((u[:, 0] * np.sqrt(s[0])).astype(np.float32), (vt[0] * np.sqrt(s[0])).astype(np.float32))
        w, h = in_size
        r = p["resize"]
        self.out_size = (even(round(w * r["scale"])), even(round(h * r["scale"])))
        self.resize = self.out_size != (w, h)

    def __call__(self, frame, vel=(0.0, 0.0)):
        p = self.p
        x = frame.astype(np.float32)
        if p.get("motion_blur", 0) > 0:
            x = motion_blur(x, vel[0], vel[1], p["motion_blur"])
        if self.sep is not None:
            x = cv2.sepFilter2D(x, -1, self.sep[1], self.sep[0])
        elif self.kernel is not None:
            x = cv2.filter2D(x, -1, self.kernel)
        if p.get("sharpen"):
            s = p["sharpen"]
            x = x + s["amount"] * (x - cv2.GaussianBlur(x, (0, 0), s["sigma"]))
        if self.resize:
            x = cv2.resize(x, self.out_size, interpolation=INTERP[p["resize"]["interp"]])
        if p.get("noise"):
            x = add_noise(x, p["noise"], self.nprng)
        if p.get("jpeg"):
            x = jpeg(x, p["jpeg"])
        return np.clip(x, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- video compression
def codec_args(comp, w, h, fps):
    """ffmpeg encoder args for a compression step; the bitrate is bpp x pixels x fps."""
    kbps = max(comp["bpp"] * w * h * fps / 1000, 16)
    if comp["codec"] == "h264":
        return ["-c:v", "libx264", "-preset", "medium", "-b:v", f"{kbps:.0f}k", "-bufsize", f"{2 * kbps:.0f}k",
                "-g", str(comp["keyint"]), "-bf", "3"]
    if comp["codec"] == "vp9":
        return ["-c:v", "libvpx-vp9", "-b:v", f"{kbps:.0f}k", "-deadline", "good", "-cpu-used", "4", "-row-mt", "1",
                "-g", str(comp["keyint"])]
    raise ValueError(comp["codec"])


def stage1_writer(plan, path, gt_size, fps, nprng):
    """(Stage, Writer) for stage 1: frames go through Stage and into its compression (or a lossless file)."""
    st = Stage(plan["stage1"], gt_size, nprng)
    comp = plan["stage1"]["compress"]
    args = codec_args(comp, *st.out_size, fps) if comp else LOSSLESS_RGB
    return st, Writer(path, *st.out_size, fps, args, pix_fmt="yuv420p" if comp else "bgr0")


def finish(plan, stage1_path, s1_size, out_path, lq_size, fps, n, nprng, tmpdir):
    """Stage 2 and the final step on the stage-1 file -> LQ file of lq_size. Returns what was stored and the bitrates."""
    st2 = Stage(plan["stage2"], s1_size, nprng)
    fin = plan["final"]
    comp = fin["compress"]
    sinc = make_kernel(fin["sinc"]) if fin.get("sinc") else None
    interp = INTERP[fin["interp"]]

    def to_lq(f):
        f = cv2.resize(f, lq_size, interpolation=interp) if f.shape[1::-1] != tuple(lq_size) else f
        if sinc is not None:
            f = np.clip(cv2.filter2D(f.astype(np.float32), -1, sinc), 0, 255).astype(np.uint8)
        return f

    frames2 = (st2(f) for f in stream_frames(stage1_path, "null", *s1_size, n))
    info = {"stage2_size": list(st2.out_size)}
    if fin["compress_first"]:                              # compress at the stage-2 size, then resize (+ sinc) to LQ
        coded = os.path.join(tmpdir, "final_coded.mp4")
        w = Writer(coded, *st2.out_size, fps, codec_args(comp, *st2.out_size, fps))
        for f in frames2:
            w.write(f)
        w.close()
        info["final_kbps"] = round(os.path.getsize(coded) * 8 / 1000 / (n / fps), 1)
        w = Writer(out_path, *lq_size, fps, LOSSLESS_H264)
        for f in stream_frames(coded, "null", *st2.out_size, n):
            w.write(to_lq(f))
        w.close()
        info["stored_as"] = "h264_lossless"
    else:                                                  # resize (+ sinc) to LQ, then the final encode
        coded = os.path.join(tmpdir, "final_coded.mp4")
        w = Writer(coded, *lq_size, fps, codec_args(comp, *lq_size, fps))
        for f in frames2:
            w.write(to_lq(f))
        w.close()
        info["final_kbps"] = round(os.path.getsize(coded) * 8 / 1000 / (n / fps), 1)
        if comp["codec"] == "h264":
            os.replace(coded, out_path)
            info["stored_as"] = "h264"
        else:                                              # VP9 is stored as its exact decode in lossless H.264
            transcode(coded, out_path, "null", LOSSLESS_H264)
            info["stored_as"] = "h264_lossless"
    return info
