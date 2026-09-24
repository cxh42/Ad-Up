"""Make (GT, LQ) pairs: render each ad spec into a UGC-style GT, then degrade it the way real ads are degraded.

Input is either ad specs from adup.ugc.director (--sequences specs.jsonl, the normal route) or a manifest of single
HQ clips (--manifest, one plain shot per clip). Parameters come from one dataset config (configs/pairs/<version>.yaml,
sections gt / ugc / degrade). Everything streams frame by frame, so 2K / 4K GT and 40 s ads fit in memory.

Per ad (random parameters are drawn once per LQ variant and held fixed over the ad, like a real upload):
  GT        1. geometry: aspect ratio from the spec, GT short side --gt-short (never upscaled), LQ size from --scale
               (a factor, or "auto": LQ short side drawn from degrade.lq_short)
            2. gate (hq/gate.py) unless the spec's clips were pre-gated
            3. render each shot (ugc/render.py: face-aware crop, virtual handheld camera, layouts, stills, slides,
               screen recordings) and join them with transitions (ugc/sequence.py)
            4. phone colour look matched to a real ad shot (ugc/look.py), camera-captured shots only
            5. burned-in text: captions, hook titles, stickers, fine print (ugc/text.py)
  LQ        6. capture / ISP: motion blur, defocus, noise, denoise / skin smoothing, sharpening (degrade/capture.py)
            7. editing-app export, platform resize + pre-processing + H.264 / VP9 / AV1 / HEVC, re-upload
               (degrade/platform.py)
Frame count and fps are preserved end to end, so GT and LQ stay frame-aligned.

Output (data/pairs/<dataset>/): config.yaml (the config used) and one directory per ad with gt.mp4, lq_<k>.mp4 and
meta.json (every sampled parameter); multi-shot ads also get shots/shot_<i>_{gt,lq_<k>}.mp4 (frame-exact, lossless)
and meta.json["shots"] with each shot's source and boundaries.

Usage (from the repo root):
  .venv-iqa/bin/python -m adup.make_pairs --sequences data/pairs/ugc_v5_2k/specs.jsonl --gt-short 1440 --scale auto
  .venv-iqa/bin/python -m adup.make_pairs --manifest data/hq/ultravideo_4k/manifest.csv --out data/pairs/plain_2k_x2 \
      --gt-short 1440 --scale 2
"""

import argparse
import json
import os
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yaml

from adup.config import add_config_args, load_config
from adup.degrade.capture import apply_capture, sample_capture
from adup.degrade.platform import edit_codec, finish_variant, pick_weighted, sample_chain
from adup.hq.gate import crop_filter, gate_ok, gate_stats
from adup.media import GT_H264, Writer, feasible_aspects, gt_geometry, max_gt_short, probe, split_at
from adup.ugc.look import LOOK_STATS, apply_look, match_look, real_look_targets, sample_look, tone_lut
from adup.ugc.render import ShotRenderer
from adup.ugc.sequence import sequence_frames, shot_ranges
from adup.ugc.text import draw_text, plan_text


def process_sequence(seq, out_dir, gt_short, scale, variants, seed, config):
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    deg, ugc = config["degrade"], config["ugc"]
    gt_short = seq.get("gt_short", gt_short)       # the director reduces it for 9:16 ads cut from 4K landscape clips
    lq_short = None
    if scale == "auto":                     # real inputs are 540-720p: sample the LQ short side, scale = GT / LQ
        lq_short = int(pick_weighted(rng, {int(k): v for k, v in deg["lq_short"].items()}))
        scale = gt_short / lq_short
    shots, transitions, fps = seq["shots"], seq["transitions"], seq["fps"]
    if seq.get("aspect"):
        aspect = seq["aspect"]
    else:
        sizes = [probe(s["src"])[:2] for s in shots]
        aspects = feasible_aspects(sizes, gt_short, scale, config["gt"]["aspects"])
        portrait_short = None
        if "9:16" in config["gt"]["aspects"] and "9:16" not in aspects and config["gt"].get("portrait_min_short"):
            fit = [max_gt_short(sw, sh, "9:16", gt_short, scale, config["gt"]["portrait_min_short"]) for sw, sh in sizes]
            if all(fit):
                portrait_short, aspects["9:16"] = min(fit), config["gt"]["aspects"]["9:16"]
        if not aspects:
            raise ValueError(f"no aspect ratio fits GT short side {gt_short} in sources {sizes}")
        aspect = pick_weighted(rng, aspects)
        if aspect == "9:16" and portrait_short:
            gt_short = portrait_short
            scale = gt_short / lq_short if lq_short else scale
    gw, gh, lw, lh = gt_geometry(aspect, gt_short, scale)
    gates = []
    if not seq.get("pregated"):                  # clip-level gate already applied by the director otherwise
        for s in shots:
            if s.get("kind", "video") != "video":
                continue
            sw, sh = probe(s["src"])[:2]
            vf = crop_filter(s["src"], s["start"], int(s["frames"] * s["src_fps"] / fps), sw, sh, gw, gh, rng)
            stats = gate_stats(s, vf, gw, gh, fps)
            if not gate_ok(stats, config["gt"]):
                raise ValueError(f"GT rejected ({os.path.basename(s['src'])}): " +
                                 ", ".join(f"{k}={v:.1f}" for k, v in stats.items()))
            gates.append(stats)
    renderers = [ShotRenderer(s, gw, gh, fps, nrng, ugc) for s in shots]
    if "look" in seq:
        look = seq["look"]
    elif nrng.random() >= ugc["look"]["look_prob"]:
        look = None
    elif (targets := real_look_targets()) is not None:
        ref = next((r for r in renderers if r.kind != "slide"), renderers[0])
        look = match_look(ref.preview, dict(zip(LOOK_STATS, targets[nrng.integers(len(targets))])), nrng, ugc["look"])
    else:
        look = sample_look(nrng, {**ugc["look"], "look_prob": 1.0})
    lut = tone_lut(look) if look else None
    ranges = shot_ranges(shots, transitions)
    n = ranges[-1][1]
    items, text_meta = plan_text(n, fps, gw, gh, rng, ugc["text"])

    os.makedirs(out_dir, exist_ok=True)
    meta = {"id": seq["id"], "purpose": seq.get("purpose"), "fps": fps, "frames": n, "gt_size": [gw, gh],
            "lq_size": [lw, lh], "scale": scale, "aspect": aspect, "gate": gates, "look": look,
            "render": [r.info for r in renderers], "text": text_meta,
            "config": config, "variants": []}
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        caps = [sample_capture(rng, deg) for _ in range(variants)]
        chains = [sample_chain(rng, deg) for _ in range(variants)]
        nprngs = [np.random.default_rng(rng.randint(0, 2 ** 31)) for _ in range(variants)]
        gt_w = Writer(os.path.join(out_dir, "gt.mp4"), gw, gh, fps, GT_H264)
        edits = [Writer(os.path.join(tmp, f"edit_{k}.mp4"), gw, gh, fps, edit_codec(ch)) for k, ch in enumerate(chains)]
        noise_gain = (gw / lw) / 1.5
        with ThreadPoolExecutor(max_workers=variants + 1) as pool:      # variants in parallel (cv2 / numpy drop the GIL)
            for i, (f, vel, camera) in enumerate(sequence_frames(shots, transitions, renderers, rng)):
                if camera:                          # screen recordings and text slides are digital: no phone look / ISP
                    f = apply_look(f, look, lut)
                draw_text(f, i, items)
                jobs = [pool.submit(apply_capture, f, cap, nprng, noise_gain, vel) if camera else None
                        for cap, nprng in zip(caps, nprngs)]
                gt_w.write(f)
                for w, j in zip(edits, jobs):
                    w.write(j.result() if j is not None else f)
        gt_w.close()
        for w in edits:
            w.close()
        for k, (cap, chain) in enumerate(zip(caps, chains)):
            out = os.path.join(out_dir, f"lq_{k}.mp4")
            vt = os.path.join(tmp, f"v{k}")
            os.makedirs(vt)
            info = finish_variant(os.path.join(tmp, f"edit_{k}.mp4"), out, chain, lw, lh, n / fps, vt)
            meta["variants"].append({"file": out, "capture": cap, "chain": chain, **info})

    if len(shots) > 1:
        cuts = [a for a, _ in ranges[1:]]
        sd = os.path.join(out_dir, "shots")
        os.makedirs(sd, exist_ok=True)
        for name in ["gt"] + [f"lq_{k}" for k in range(variants)]:
            split_at(os.path.join(out_dir, f"{name}.mp4"), os.path.join(sd, f"shot_%03d_{name}.mp4"), cuts)
        meta["cut_method"] = {"type": "composition", "note": "boundaries are exact, known from the sequence spec; "
                              "a dissolve's blended frames are the head of the following shot"}
        meta["shots"] = [{"index": i, "start_frame": a, "end_frame": b, "frames": b - a,
                          "source": {k: s[k] for k in ("kind", "src", "clip_id", "source_id", "category", "src_fps", "start",
                                                       "layout") if k in s},
                          "source_frames_used": s["frames"],
                          "transition_in": transitions[i - 1] if i else None,
                          "transition_out": transitions[i] if i < len(shots) - 1 else None,
                          "files": {name: os.path.join(sd, f"shot_{i:03d}_{name}.mp4")
                                    for name in ["gt"] + [f"lq_{k}" for k in range(variants)]}}
                         for i, (s, (a, b)) in enumerate(zip(shots, ranges))]
    else:
        meta["source"] = {k: shots[0][k] for k in ("kind", "src", "src_fps", "start") if k in shots[0]}
    meta["spec"] = seq
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return meta


def clip_specs(manifest, max_frames):
    """Single-shot specs for a manifest of clips, at each clip's native frame rate."""
    specs = []
    for f in pd.read_csv(manifest)["file"]:
        _, _, fps, nb = probe(f)
        fps = round(fps, 3)
        specs.append({"id": os.path.splitext(os.path.basename(f))[0], "purpose": "clip", "fps": fps,
                      "shots": [{"src": f, "src_fps": fps, "start": 0, "frames": min(max_frames, nb or max_frames)}],
                      "transitions": []})
    return specs


def save_config(config, out):
    """Write the dataset's config.yaml; warn instead of overwriting when an existing dataset used another config."""
    path = os.path.join(out, "config.yaml")
    if os.path.exists(path) and yaml.safe_load(open(path)) != config:
        print(f"warning: {path} differs from the current config; new pairs record theirs in meta.json", flush=True)
        return
    yaml.safe_dump(config, open(path, "w"), sort_keys=False, allow_unicode=True)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sequences", help="JSONL of ad specs from adup.ugc.director")
    src.add_argument("--manifest", help="CSV with a `file` column of single-shot HQ clips")
    ap.add_argument("--out", help="dataset directory (default: the directory of --sequences)")
    ap.add_argument("--gt-short", type=int, default=1440, help="GT short side (1440 = 2K, 2160 = 4K)")
    ap.add_argument("--scale", default="2", help="GT / LQ factor, or 'auto' to sample the LQ short side (degrade.lq_short)")
    ap.add_argument("--variants", type=int, default=2, help="LQ versions per GT")
    ap.add_argument("--max-frames", type=int, default=150, help="--manifest only: frames per clip")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    add_config_args(ap)
    args = ap.parse_args()
    config = load_config(args.config, args.set)
    if not (args.out or args.sequences):
        ap.error("--out is required with --manifest")
    out = args.out or os.path.dirname(args.sequences) or "."
    os.makedirs(out, exist_ok=True)
    save_config(config, out)
    args.scale = args.scale if args.scale == "auto" else float(args.scale)
    specs = clip_specs(args.manifest, args.max_frames) if args.manifest else [json.loads(l) for l in open(args.sequences)]
    if args.limit:
        specs = specs[:args.limit]
    for i, seq in enumerate(specs):
        out_dir = os.path.join(out, seq["id"])
        if os.path.exists(os.path.join(out_dir, "meta.json")):
            continue
        try:
            process_sequence(seq, out_dir, args.gt_short, args.scale, args.variants, args.seed * 100003 + i, config)
            print(f"[{i + 1}/{len(specs)}] {seq['id']}", flush=True)
        except Exception as e:
            print(f"[{i + 1}/{len(specs)}] {seq['id']} failed: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
