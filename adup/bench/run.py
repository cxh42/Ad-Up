"""Run a VSR method on benchmark inputs under one protocol, writing 1080p outputs plus cost records.

Protocol (docs/benchmark.md):
  - every output has the benchmark's output size (short side --out-short, default 1080), whatever the input size
  - fixed-scale models run at their native factor (repeated if still below the output size), and anything larger is
    area-downsampled to the output size; inputs are never downscaled first
  - models that restore at the output size (DOVE and other diffusion models) get the input upsampled to it
  - video models are run shot by shot (cuts from meta.json for synthetic pairs, detected cuts otherwise)
  - cost: time per output frame without model loading (the model's own timing where it reports one), and peak GPU
    memory, on the machine given in the record

Inputs: a pairs directory (every LQ variant listed in its meta.json files; output name <ad id>__<variant>.mp4) or a
text file listing videos (real ads; output name <file stem>.mp4).

Usage (from the repo root):
  .venv-iqa/bin/python -m adup.bench.run --method bicubic --pairs data/pairs/v7_dev --out outputs/bench/v7_dev
  .venv-iqa/bin/python -m adup.bench.run --method dove --list data/stats/real_ads/groups/dev_sd.txt --out outputs/bench/real_dev
"""

import argparse
import glob
import json
import os
import platform
import subprocess
import sys
import time

import torch

from adup.media import Writer, probe, stream_frames
from adup.paths import ASSETS, ROOT

OUT_H264 = ["-c:v", "libx264", "-preset", "medium", "-crf", "10"]


def out_size(w, h, out_short):
    s = out_short / min(w, h)
    return int(round(w * s / 2)) * 2, int(round(h * s / 2)) * 2


def area_down(frame, size):
    import cv2
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if frame.shape[1::-1] != tuple(size) else frame


# ---------------------------------------------------------------- methods
def run_ffmpeg_scale(flags):
    def run(src, dst, size, meta):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", f"scale={size[0]}:{size[1]}:flags={flags}",
                        *OUT_H264, "-pix_fmt", "yuv420p", dst], check=True)
    return run


class ImageModel:
    """A per-frame image SR model loaded with spandrel (Real-ESRGAN etc.), run at its native scale then area-down."""

    def __init__(self, weights):
        import spandrel
        self.m = spandrel.ModelLoader().load_from_file(str(weights)).cuda().eval()
        self.scale = self.m.scale

    @torch.no_grad()
    def __call__(self, src, dst, size, meta):
        w, h, fps, n = probe(src)
        wr = Writer(dst, *size, fps, OUT_H264)
        for f in stream_frames(src, "null", w, h, n):
            x = torch.from_numpy(f).cuda().permute(2, 0, 1)[None].float() / 255
            while x.shape[-1] < size[0]:                         # native factor, repeated until large enough
                x = self.m(x).clamp(0, 1)
            y = (x[0].permute(1, 2, 0) * 255 + 0.5).byte().cpu().numpy()
            wr.write(area_down(y, size))
        wr.close()


def run_dove(src, dst, size, meta):
    """DOVE through training/dove/infer.py in .venv-dove (shot-aware, restores at the output size)."""
    tmp = dst + ".dove"
    subprocess.run([str(ROOT / ".venv-dove/bin/python"), str(ROOT / "training/dove/infer.py"), "--out", tmp,
                    "--out-short", str(min(size)), src], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    stem = os.path.splitext(os.path.basename(src))[0]
    os.replace(os.path.join(tmp, f"{stem}.mp4"), dst)
    info = json.load(open(os.path.join(tmp, f"{stem}.json")))
    os.remove(os.path.join(tmp, f"{stem}.json"))
    os.rmdir(tmp)
    return {"peak_gpu_gib": info["peak_gpu_gib"], "model_seconds": info["seconds"]}


METHODS = {
    "bicubic": lambda: run_ffmpeg_scale("bicubic"),
    "lanczos": lambda: run_ffmpeg_scale("lanczos"),
    "realesrgan": lambda: ImageModel(ASSETS / "models" / "RealESRGAN_x4plus.pth"),
    "dove": lambda: run_dove,
}


# ---------------------------------------------------------------- inputs
def pair_inputs(pairs):
    for m in sorted(glob.glob(os.path.join(pairs, "*", "meta.json"))):
        meta = json.load(open(m))
        for v in meta["variants"]:
            yield os.path.join(os.path.dirname(m), os.path.basename(v["file"])), f"{meta['id']}__{v['name']}", meta


def list_inputs(path):
    for line in open(path):
        p = line.strip()
        if p:
            yield str(ROOT / p), os.path.splitext(os.path.basename(p))[0], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(METHODS))
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pairs", help="pairs directory (synthetic dev / test)")
    src.add_argument("--list", help="text file of videos (real ads)")
    ap.add_argument("--out", required=True, help="results root; outputs go to <out>/<method>/")
    ap.add_argument("--out-short", type=int, default=1080)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="only the first N frames of each input (quick local tests on long real ads)")
    args = ap.parse_args()
    run = METHODS[args.method]()
    od = os.path.join(args.out, args.method)
    os.makedirs(od, exist_ok=True)
    inputs = list(pair_inputs(args.pairs) if args.pairs else list_inputs(args.list))
    if args.limit:
        inputs = inputs[:args.limit]
    host = {"host": platform.node(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    for k, (path, name, meta) in enumerate(inputs):
        dst = os.path.join(od, f"{name}.mp4")
        if os.path.exists(dst) and os.path.exists(dst[:-4] + ".json"):
            continue
        tmp = None
        if args.max_frames:
            w, h, fps, n = probe(path)
            if n > args.max_frames:                      # trimmed, losslessly, to a temporary input
                tmp = dst[:-4] + ".in.mp4"                # lossless and in the source's own yuv420p: pixels unchanged
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-frames:v", str(args.max_frames), "-an",
                                "-c:v", "libx264", "-preset", "veryfast", "-qp", "0", "-pix_fmt", "yuv420p", tmp],
                               check=True)
        src_path = tmp or path
        w, h, fps, n = probe(src_path)
        size = out_size(w, h, args.out_short)
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        t = time.time()
        extra = run(src_path, dst, size, meta) or {}
        sec = time.time() - t
        if tmp:
            os.remove(tmp)
        rec = {"method": args.method, "input": os.path.relpath(path, ROOT), "output": os.path.relpath(dst, ROOT),
               "input_size": [w, h], "output_size": list(size), "frames": n, "seconds": round(sec, 2),
               "s_per_frame": round(extra.get("model_seconds", sec) / max(n, 1), 4), **host,
               "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None,
               **extra}
        json.dump(rec, open(dst[:-4] + ".json", "w"), indent=1)
        print(f"[{k + 1}/{len(inputs)}] {name}: {w}x{h} -> {size[0]}x{size[1]}, {rec['s_per_frame']} s/frame", flush=True)


if __name__ == "__main__":
    sys.exit(main())
