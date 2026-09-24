"""Detect the shots of existing videos (real ads, KwaiVIR, HQ clips), keeping the originals untouched.

For each input video this writes <out_root>/<video_id>.json recording how it was cut: parent path, frame and time
ranges of every shot, the detector, its parameters and its version. With --split it also writes one frame-exact file
per shot (<out_root>/<video_id>/shot_000.mp4, ...), re-encoded losslessly (x264 -qp 0) so the split adds no degradation.
--table aggregates every <out_root>/*.json into a shot-length table (one row per shot); the director draws its cut
rhythm from data/stats/real_ads/shots.csv. training/dove/infer.py uses detect_cuts for shot-aware inference.

Usage (from the repo root):
  .venv-iqa/bin/python -m adup.analysis.shots data/real_ads/tiktok_topads/<date>/shots <videos...> [--split] \
      [--table data/stats/real_ads/shots.csv]
"""

import argparse
import glob
import json
import os
import subprocess

import pandas as pd
import scenedetect
from scenedetect import AdaptiveDetector, detect

DETECTOR_PARAMS = {"adaptive_threshold": 3.0, "min_scene_len": 8, "window_width": 2, "min_content_val": 15.0}
LOSSLESS_H264 = ["-c:v", "libx264", "-preset", "veryfast", "-qp", "0", "-pix_fmt", "yuv420p"]


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(num) / float(den), int(s["nb_read_frames"])


def detect_cuts(path):
    """Frame indices where a new shot starts (excluding 0)."""
    scenes = detect(path, AdaptiveDetector(**DETECTOR_PARAMS))
    return [getattr(s[0], "frame_num", None) or s[0].get_frames() for s in scenes[1:]]


def cut_range(src, dst, start, end):
    """Frames [start, end) of src -> dst, frame-exact and lossless."""
    vf = f"select=between(n\\,{start}\\,{end - 1}),setpts=N/FRAME_RATE/TB"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", vf, *LOSSLESS_H264, "-fps_mode", "passthrough",
                    "-an", dst], check=True)


def shots_record(path, cuts, n, fps, w, h):
    bounds = [0, *cuts, n]
    return {
        "parent": path, "parent_frames": n, "fps": fps, "size": [w, h],
        "cut_method": {"type": "detected", "tool": "PySceneDetect", "version": scenedetect.__version__,
                       "detector": "AdaptiveDetector", "params": DETECTOR_PARAMS,
                       "note": "hard cuts only; dissolves / fades may be missed or split mid-transition"},
        "shots": [{"index": i, "start_frame": a, "end_frame": b, "frames": b - a,
                   "start_time": round(a / fps, 3), "end_time": round(b / fps, 3),
                   "boundary_in": "start" if i == 0 else "cut", "boundary_out": "end" if b == n else "cut"}
                  for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]))],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_root")
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--split", action="store_true", help="also write one lossless file per shot")
    ap.add_argument("--table", help="write a shot-length CSV from every <out_root>/*.json")
    args = ap.parse_args()
    os.makedirs(args.out_root, exist_ok=True)
    for k, path in enumerate(args.videos):
        vid = os.path.splitext(os.path.basename(path))[0]
        rec_path = os.path.join(args.out_root, f"{vid}.json")
        if os.path.exists(rec_path):
            continue
        w, h, fps, n = probe(path)
        rec = shots_record(path, detect_cuts(path), n, fps, w, h)
        if args.split:
            out = os.path.join(args.out_root, vid)
            os.makedirs(out, exist_ok=True)
            for s in rec["shots"]:
                s["file"] = os.path.join(out, f"shot_{s['index']:03d}.mp4")
                cut_range(path, s["file"], s["start_frame"], s["end_frame"])
        json.dump(rec, open(rec_path, "w"), indent=1)
        print(f"[{k + 1}/{len(args.videos)}] {vid}: {len(rec['shots'])} shots", flush=True)
    if args.table:
        rows = []
        for f in sorted(glob.glob(os.path.join(args.out_root, "*.json"))):
            rec = json.load(open(f))
            rows += [{"video": os.path.splitext(os.path.basename(f))[0], "parent": rec["parent"], "frames": s["frames"],
                      "fps": rec["fps"], "total": rec["parent_frames"]} for s in rec["shots"]]
        pd.DataFrame(rows).to_csv(args.table, index=False)
        print(f"{len(rows)} shots -> {args.table}")


if __name__ == "__main__":
    main()
