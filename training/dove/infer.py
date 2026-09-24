"""Shot-aware, memory-bounded DOVE inference for long ads, reusing upstream DOVE's model call unchanged.

Differences from third_party/DOVE/inference_script.py, none of which change the model's computation:
  - shots: the input is split at detected cuts (adup.analysis.shots) and each shot is restored on its own, so the
    temporal VAE / attention never mixes two shots. --no-shots processes the video as one piece, like upstream.
  - memory: frames stay uint8 on the CPU and are upsampled on the GPU chunk by chunk; output streams to ffmpeg.
    Upstream keeps the whole upsampled video as float32 on the CPU (a 100-frame 4K output needs ~30 GB of RAM).
  - the T5 text encoder is not loaded: DOVE always uses the pre-computed empty-prompt embedding.
  - the transformer's feed-forward runs over tokens in chunks (per-token op, bit-identical output) to cut peak memory.
  - long shots: one pass holds ~33 frames at 4K / ~73 at 2K on a 32 GB GPU (bf16, no offload); longer shots
    are split into overlapping temporal chunks (upstream's scheme), still longer than DOVE's 25-frame training clips.
  - padding is removed as pad * upscale (upstream hard-codes pad * 4, which is wrong for --upscale != 4).
Spatial tiling (--tile) follows upstream's valid-region scheme and is off by default.

Usage (from the repo root, DOVE venv):
  .venv-dove/bin/python training/dove/infer.py --out outputs/runs/dove/test --upscale 4 <videos...>
"""

import argparse
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DOVE = ROOT / "third_party" / "DOVE"
sys.path[:0] = [str(ROOT), str(DOVE)]
sys.modules.setdefault("pyiqa", types.ModuleType("pyiqa"))     # upstream imports it for metrics only

from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline      # noqa: E402
from safetensors.torch import load_file                             # noqa: E402

import inference_script as upstream                                 # noqa: E402
from adup.analysis.shots import detect_cuts                           # noqa: E402

# output pixels x frames per pass on a 32 GB GPU (RTX 5090, measured end to end: 33 frames at 4K fit, 81 at 2K do not)
PASS_BUDGET = 33 * 3840 * 2160
EMPTY_PROMPT = DOVE / "pretrained_models/prompt_embeddings/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.safetensors"


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(num) / float(den), int(s["nb_read_frames"])


def read_range(path, start, end, w, h):
    vf = f"select=between(n\\,{start}\\,{end - 1})"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", vf, "-fps_mode", "passthrough", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


class ChunkedFF(torch.nn.Module):
    """Feed-forward applied to chunks of tokens; identical output, lower peak memory."""

    def __init__(self, ff, chunk):
        super().__init__()
        self.ff, self.chunk = ff, chunk

    def forward(self, x):
        return torch.cat([self.ff(c) for c in x.split(self.chunk, dim=1)], dim=1)


def pass_frames(h, w):
    """Largest 8N+1 frame count whose output fits PASS_BUDGET."""
    return max((PASS_BUDGET // (h * w) - 1) // 8 * 8 + 1, 9)


def load_pipe(model_path, vae_tiling):
    pipe = CogVideoXPipeline.from_pretrained(model_path, text_encoder=None, tokenizer=None, torch_dtype=torch.bfloat16)
    for blk in pipe.transformer.transformer_blocks:
        blk.ff = ChunkedFF(blk.ff, 16384)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.to("cuda")
    if vae_tiling:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    return pipe


def restore(pipe, prompt_emb, frames, upscale, chunk_len, overlap_t, tile_hw, overlap_hw):
    """frames: uint8 [F, H, W, 3] of one shot -> uint8 [F, H*upscale, W*upscale, 3]."""
    F, H, W, _ = frames.shape
    pad_f = (8 - (F - 1) % 8) % 8                                   # DOVE needs 8N+1 frames and H, W % 16 == 0
    pad_h, pad_w = (16 - H % 16) % 16, (16 - W % 16) % 16
    x = torch.from_numpy(frames).permute(0, 3, 1, 2)                # [F, C, H, W] uint8
    if pad_f:
        x = torch.cat([x, x[-1:].repeat(pad_f, 1, 1, 1)])
    x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
    Fp, Hp, Wp = x.shape[0], (H + pad_h) * upscale, (W + pad_w) * upscale
    if not chunk_len and Fp > pass_frames(Hp, Wp):
        chunk_len = pass_frames(Hp, Wp)
    t_chunks = upstream.make_temporal_chunks(Fp, chunk_len, overlap_t if chunk_len else 0)
    tiles = upstream.make_spatial_tiles(Hp, Wp, tile_hw, overlap_hw if tile_hw != (0, 0) else (0, 0))
    out = torch.empty((Fp, 3, Hp, Wp), dtype=torch.uint8)
    for t0, t1 in t_chunks:
        lq = x[t0:t1].to("cuda", torch.float32)
        up = torch.nn.functional.interpolate(lq, size=(Hp, Wp), mode="bilinear", align_corners=False)
        up = (up / 255.0 * 2.0 - 1.0).to(torch.bfloat16).permute(1, 0, 2, 3).unsqueeze(0)   # [1, C, F, H, W]
        for h0, h1, w0, w1 in tiles:
            y = upstream.process_video(pipe=pipe, video=up[:, :, :, h0:h1, w0:w1], prompt="",
                                       empty_prompt_embedding=prompt_emb)
            r = upstream.get_valid_tile_region(t0, t1, h0, h1, w0, w1, video_shape=(1, 3, Fp, Hp, Wp),
                                               overlap_t=overlap_t if chunk_len else 0, overlap_h=overlap_hw[0] if tiles[1:] else 0,
                                               overlap_w=overlap_hw[1] if tiles[1:] else 0)
            y = y[0, :, r["valid_t_start"]:r["valid_t_end"], r["valid_h_start"]:r["valid_h_end"],
                  r["valid_w_start"]:r["valid_w_end"]]
            out[r["out_t_start"]:r["out_t_end"], :, r["out_h_start"]:r["out_h_end"], r["out_w_start"]:r["out_w_end"]] = \
                (y.float().clamp(0, 1) * 255 + 0.5).to(torch.uint8).permute(1, 0, 2, 3).cpu()
        del lq, up
    return out[:F, :, :H * upscale, :W * upscale].permute(0, 2, 3, 1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-path", default=str(DOVE / "pretrained_models" / "DOVE"))
    ap.add_argument("--upscale", type=int, default=4)
    ap.add_argument("--no-shots", action="store_true", help="process the whole video as one piece (upstream behaviour)")
    ap.add_argument("--chunk-len", type=int, default=0, help="0 = whole shot in one pass if it fits, else automatic")
    ap.add_argument("--overlap-t", type=int, default=8)
    ap.add_argument("--tile", type=int, nargs=2, default=(0, 0), help="spatial tile (h w) in output pixels; 0 0 = none")
    ap.add_argument("--overlap-hw", type=int, nargs=2, default=(32, 32))
    ap.add_argument("--no-vae-tiling", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--crf", type=int, default=10, help="x264 CRF of the saved output")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(42)
    pipe = load_pipe(args.model_path, not args.no_vae_tiling)
    prompt_emb = load_file(str(EMPTY_PROMPT))["prompt_embedding"]
    print(f"model loaded, GPU memory {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

    for path in args.videos:
        w, h, fps, n = probe(path)
        n = min(n, args.max_frames) if args.max_frames else n
        cuts = [] if args.no_shots else [c for c in detect_cuts(path) if c < n]
        bounds = [0, *cuts, n]
        name = os.path.splitext(os.path.basename(path))[0]
        dst = os.path.join(args.out, f"{name}.mp4")
        W, H = w * args.upscale, h * args.upscale
        writer = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                   "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf",
                                   str(args.crf), "-pix_fmt", "yuv420p", dst], stdin=subprocess.PIPE)
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        shots = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            ts = time.time()
            lq, chunk = read_range(path, a, b, w, h), args.chunk_len
            while True:
                try:
                    y = restore(pipe, prompt_emb, lq, args.upscale, chunk, args.overlap_t, tuple(args.tile),
                                tuple(args.overlap_hw))
                    break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    chunk = max(((chunk or pass_frames(h * args.upscale, w * args.upscale)) // 2 - 1) // 8 * 8 + 1, 17)
                    print(f"  OOM, retrying with chunk_len {chunk}", flush=True)
            writer.stdin.write(np.ascontiguousarray(y).tobytes())
            shots.append({"start_frame": a, "end_frame": b, "seconds": round(time.time() - ts, 1), "chunk_len": chunk})
        writer.stdin.close()
        writer.wait()
        info = {"input": path, "output": dst, "frames": n, "input_size": [w, h], "output_size": [W, H],
                "upscale": args.upscale, "shot_aware": not args.no_shots, "shots": shots,
                "cut_method": None if args.no_shots else "PySceneDetect AdaptiveDetector (adup.analysis.shots)",
                "chunk_len": args.chunk_len, "tile": args.tile, "vae_tiling": not args.no_vae_tiling,
                "seconds": round(time.time() - t, 1), "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        json.dump(info, open(os.path.join(args.out, f"{name}.json"), "w"), indent=1)
        print(f"{name}: {n} frames {w}x{h} -> {W}x{H}, {len(shots)} shots, {info['seconds']} s, "
              f"peak GPU {info['peak_gpu_gib']} GiB", flush=True)


if __name__ == "__main__":
    main()
