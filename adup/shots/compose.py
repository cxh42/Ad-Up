"""Compose multi-shot sequences from single-shot HQ clips, the way ads are edited, as JSONL specs for the pipeline.

Real ads cut every ~3 s (3.5 cuts per 10 s; 27% of 25-frame windows contain a cut), while our HQ clips are single
shots. Each sequence strings several clips together with an edit transition between them. Shot lengths are drawn from
the shots measured in real TikTok ads (outputs/analysis/shots.csv). Following shots come from the same source video
where possible (later clips of the same YouTube video, like an edit of one shoot), otherwise from the same category.

  train: 2+ shots filling --max-frames (default 150 = 5 s), so training windows contain cuts at the real rate
  eval:  15-40 s "whole ads" with 5-15 shots, for testing long-video inference and cut handling

The pipeline renders each spec, and writes both the whole sequence and its per-shot split with the exact boundaries.

Clips that failed the GT gate (adup.sources.gate, --gate) are left out, so sequences are not rejected downstream.

Usage (from the repo root):
  .venv-iqa/bin/python -m adup.shots.compose --manifest data/hq/ultravideo/manifest.csv --gate data/hq/ultravideo/gate_1440.csv \
      --purpose train --n 300 --out data/hq/sequences/train_4k.jsonl
"""

import argparse
import json
import math
import os
import random

import numpy as np
import pandas as pd

from adup.paths import ANALYSIS, HQ

SEQ_FPS = 30
TRANSITIONS = {"cut": 0.80, "dissolve": 0.08, "whip": 0.06, "dip_black": 0.03, "dip_white": 0.03}
TRANSITION_FRAMES = {"dissolve": (4, 12), "whip": (4, 8), "dip_black": (6, 12), "dip_white": (6, 12)}


def shot_lengths():
    path = ANALYSIS / "shots.csv"
    if path.exists():
        s = pd.read_csv(path)
        return (s.frames * SEQ_FPS / s.fps).round().astype(int).clip(lower=8).tolist()
    return np.clip(np.random.default_rng(0).lognormal(np.log(50), 0.8, 2000), 8, None).astype(int).tolist()


def load_clips(manifests, gates=()):
    clips = pd.concat([pd.read_csv(m) for m in manifests], ignore_index=True)
    if gates:                                       # keep clips that pass the gate in every orientation they support
        g = pd.concat([pd.read_csv(x) for x in gates]).groupby("file")["pass"].all()
        clips = clips[clips.file.map(g).fillna(False).astype(bool)]
    short = HQ / "ultravideo" / "short.csv"
    if short.exists() and "start_frame" not in clips:   # natural order of clips within a source video
        starts = pd.read_csv(short, usecols=["clip_id", "start_frame"])
        clips = clips.merge(starts, on="clip_id", how="left")
    clips["start_frame"] = clips.get("start_frame", 0)
    clips = clips[clips.file.map(os.path.exists)]
    return clips.sort_values(["youtube_id", "start_frame"]).reset_index(drop=True)


def available(clip, start=0):
    """Sequence frames (at SEQ_FPS) that a clip can supply from source frame `start`."""
    return int((clip.frames - start) * SEQ_FPS / clip.fps)


def next_clip(rng, clips, cur, used):
    same_src = clips[(clips.youtube_id == cur.youtube_id) & (clips.start_frame > cur.start_frame) & ~clips.index.isin(used)]
    r = rng.random()
    if r < 0.7 and len(same_src):
        return clips.loc[same_src.index[0]]
    unused = clips[~clips.index.isin(used)]
    pool = unused[unused.category == cur.category] if r < 0.9 else unused
    pool = pool if len(pool) else unused              # small categories run out: fall back to any clip
    return clips.loc[rng.choice(list(pool.index))] if len(pool) else None


def compose_one(rng, clips, lengths, total, min_shots):
    seed = clips.loc[rng.choice(list(clips.index))]
    shots, transitions, used, cur, n = [], [], set(), seed, 0
    while cur is not None and (n < total or len(shots) < min_shots):
        want = rng.choice(lengths)
        if len(shots) == 0 and want >= total:            # force at least one cut inside the sequence
            want = int(total * rng.uniform(0.3, 0.7))
        want = min(want, available(cur), max(total - n, 8))
        if want < 8:
            used.add(cur.name)
            cur = next_clip(rng, clips, cur, used)
            continue
        slack = cur.frames - math.ceil(want * cur.fps / SEQ_FPS)
        start = rng.randint(0, max(slack, 0))
        if shots:
            kind = rng.choices(list(TRANSITIONS), weights=list(TRANSITIONS.values()))[0]
            k = rng.randint(*TRANSITION_FRAMES[kind]) if kind != "cut" else 0
            k = min(k, want - 4, shots[-1]["frames"] - 4) if k else 0
            transitions.append({"type": kind if k > 0 else "cut", "frames": max(k, 0)})
            n -= transitions[-1]["frames"] if kind == "dissolve" else 0
        shots.append({"src": cur.file, "clip_id": cur.clip_id, "source_id": cur.youtube_id, "category": cur.category,
                      "src_fps": float(cur.fps), "start": int(start), "frames": int(want)})
        n += want
        used.add(cur.name)
        cur = next_clip(rng, clips, cur, used)
    return shots, transitions, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--gate", nargs="*", default=[], help="CSV(s) from adup.sources.gate")
    ap.add_argument("--purpose", choices=["train", "eval"], default="train")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-frames", type=int, default=150, help="train: sequence length in frames")
    ap.add_argument("--eval-seconds", type=float, nargs=2, default=(15, 40))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    clips, lengths = load_clips(args.manifest, args.gate), shot_lengths()
    print(f"{len(clips)} clips available")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for i in range(args.n):
            if args.purpose == "train":
                total, min_shots = args.max_frames, 2
            else:
                total, min_shots = int(rng.uniform(*args.eval_seconds) * SEQ_FPS), 5
            shots, transitions, n = compose_one(rng, clips, lengths, total, min_shots)
            if len(shots) < min_shots:
                continue
            spec = {"id": f"{args.purpose}_{os.path.splitext(os.path.basename(args.out))[0]}_{i:05d}",
                    "purpose": args.purpose, "fps": SEQ_FPS, "frames": n, "shots": shots, "transitions": transitions}
            f.write(json.dumps(spec) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
