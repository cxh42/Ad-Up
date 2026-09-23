"""Score video quality with the metrics DOVE used to filter HQ-VSR (DOVER, CLIP-IQA) plus MUSIQ, bitrate and an
effective-resolution probe, so different video sources can be compared on the same scale.

Usage (from the repo root): .venv-iqa/bin/python -m adup.analysis.quality <group_name> <out.csv> <video files...>
"""

import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
import pyiqa
import torch
import yaml

from adup.paths import THIRD_PARTY  # noqa: E402

DOVER_DIR = str(THIRD_PARTY / "DOVER")
sys.path.insert(0, DOVER_DIR)
from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition  # noqa: E402
from dover.models import DOVER  # noqa: E402

DEVICE = "cuda"
N_FRAMES = 8
MEAN = torch.FloatTensor([123.675, 116.28, 103.53])
STD = torch.FloatTensor([58.395, 57.12, 57.375])


def load_dover():
    with open(f"{DOVER_DIR}/dover.yml") as f:
        opt = yaml.safe_load(f)
    model = DOVER(**opt["model"]["args"]).to(DEVICE).eval()
    model.load_state_dict(torch.load(f"{DOVER_DIR}/pretrained_weights/DOVER.pth", map_location=DEVICE))
    dopt = opt["data"]["val-l1080p"]["args"]
    samplers = {}
    for stype, s in dopt["sample_types"].items():
        if "t_frag" not in s:
            samplers[stype] = UnifiedFrameSampler(s["clip_len"], s["num_clips"], s["frame_interval"])
        else:
            samplers[stype] = UnifiedFrameSampler(s["clip_len"] // s["t_frag"], s["t_frag"], s["frame_interval"], s["num_clips"])
    return model, dopt, samplers


@torch.no_grad()
def dover_scores(path, model, dopt, samplers):
    views, _ = spatial_temporal_view_decomposition(path, dopt["sample_types"], samplers)
    for k, v in views.items():
        n = dopt["sample_types"][k].get("num_clips", 1)
        views[k] = (((v.permute(1, 2, 3, 0) - MEAN) / STD).permute(3, 0, 1, 2)
                    .reshape(v.shape[0], n, -1, *v.shape[2:]).transpose(0, 1).to(DEVICE))
    tech, aes = [r.mean().item() for r in model(views)]
    # official score-level fusion -> overall score in [0, 1]
    x = (tech - 0.1107) / 0.07355 * 0.6104 + (aes + 0.08285) / 0.03774 * 0.3896
    return tech, aes, 1 / (1 + np.exp(-x))


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name,width,height,avg_frame_rate,nb_frames:format=bit_rate,duration",
                          "-of", "json", path], capture_output=True, text=True).stdout
    d = json.loads(out)
    s, f = d["streams"][0], d["format"]
    num, den = s["avg_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 0
    br = float(f.get("bit_rate") or 0)
    w, h = int(s["width"]), int(s["height"])
    return {"codec": s["codec_name"], "width": w, "height": h, "fps": round(fps, 2),
            "duration_s": round(float(f.get("duration") or 0), 1), "bitrate_kbps": round(br / 1000),
            "bpp": round(br / (w * h * fps), 4) if fps else None}


def sample_frames(path, n=N_FRAMES):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(total * 0.1, total * 0.9, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if ok:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def downup_psnr(rgb, factor=2):
    """PSNR between a frame and its bicubic down->up version. High (> ~38 dB) means little detail above 1/factor res."""
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (w // factor, h // factor), interpolation=cv2.INTER_AREA)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    mse = np.mean((rgb.astype(np.float64) - up.astype(np.float64)) ** 2)
    return 10 * np.log10(255 ** 2 / mse) if mse > 0 else 99.0


def main(group, out_csv, paths):
    model, dopt, samplers = load_dover()
    clipiqa = pyiqa.create_metric("clipiqa", device=DEVICE)
    musiq = pyiqa.create_metric("musiq", device=DEVICE)
    rows = []
    for i, p in enumerate(paths):
        try:
            row = {"group": group, "file": p, **probe(p)}
            row["dover_tech"], row["dover_aes"], row["dover"] = dover_scores(p, model, dopt, samplers)
            frames = sample_frames(p)
            ts = [torch.from_numpy(f).permute(2, 0, 1).float().div(255).unsqueeze(0).to(DEVICE) for f in frames]
            with torch.no_grad():
                row["clipiqa"] = float(np.mean([clipiqa(t).item() for t in ts]))
                row["musiq"] = float(np.mean([musiq(t).item() for t in ts]))
            row["downup_psnr_x2"] = float(np.mean([downup_psnr(f) for f in frames]))
            rows.append(row)
            print(f"[{group} {i + 1}/{len(paths)}] {os.path.basename(p)} dover={row['dover']:.3f} "
                  f"clipiqa={row['clipiqa']:.3f} musiq={row['musiq']:.1f} {row['width']}x{row['height']} {row['bitrate_kbps']}kbps")
        except Exception as e:
            print(f"  failed {p}: {type(e).__name__}: {e}")
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
