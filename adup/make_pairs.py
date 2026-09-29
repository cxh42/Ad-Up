"""Make (GT, LQ) pairs: render each ad spec into a UGC-style GT, then degrade it the way real ads are degraded.

Input is either ad specs from adup.ugc.director (--sequences specs.jsonl, the normal route) or a manifest of single
HQ clips (--manifest, one plain shot per clip). Parameters come from one dataset config (configs/pairs/<version>.yaml,
sections gt / ugc / degrade). Everything streams frame by frame, so 2K / 4K GT and 40 s ads fit in memory.

Per ad (random parameters are drawn once per LQ variant and held fixed over the ad):
  GT        1. geometry: aspect ratio from the spec, GT short side --gt-short (1080 = 1920x1080 / 1080x1920, never
               upscaled), LQ size from --scale (a factor, or "auto": LQ short side drawn from degrade.lq_short)
            2. gate (hq/gate.py) unless the spec's clips were pre-gated
            3. render each shot (ugc/render.py: face-aware crop, virtual handheld camera, layouts, stills, slides,
               screen recordings) and join them with transitions (ugc/sequence.py)
            4. burned-in text and graphics by the ad's style: captions, headlines, native text, prices, logo, CTA
               button, stickers, fine print, arrows (ugc/text.py), with the cards' own text, and its per-frame mask
  LQ        5. second-order degradation (degrade/second_order.py): stage 1 (blur, resize, noise, JPEG, H.264 / VP9)
               while the GT is rendered, then stage 2 and the final resize / compression from the stage-1 file
Frame count and fps are preserved end to end, so GT and LQ stay frame-aligned.

Output (data/pairs/<dataset>/): config.yaml (the config used) and one directory per ad with gt.mp4, lq_<k>.mp4 (random
mode) or lq_<short>_<codec>.mp4 (--grid, for dev / test),
mask.mkv (overlay alpha per frame, lossless FFV1 gray) and meta.json (every sampled parameter, the split, text boxes);
multi-shot ads also get shots/shot_<i>_{gt,lq_<k>}.mp4 (frame-exact, lossless) and meta.json["shots"] with each shot's
source and boundaries.

Usage (from the repo root):
  .venv-iqa/bin/python -m adup.make_pairs --sequences data/pairs/ugc_v7/specs.jsonl --scale auto --no-shot-files
  .venv-iqa/bin/python -m adup.make_pairs --sequences data/pairs/ugc_v7_test/specs.jsonl --grid
  .venv-iqa/bin/python -m adup.make_pairs --manifest data/hq/ultravideo/4k/manifest.csv --out data/pairs/plain_1080 \
      --scale auto
"""

import argparse
import json
import os
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import yaml

from adup.config import add_config_args, load_config
from adup.degrade.second_order import finish, sample_plan, stage1_writer
from adup.hq.gate import crop_filter, gate_ok, gate_stats
from adup.media import GT_H264, Writer, feasible_aspects, gt_geometry, probe, split_at
from adup.ugc.render import ShotRenderer
from adup.ugc.sequence import sequence_frames, shot_ranges
from adup.ugc.text import draw_mask, draw_text, item_meta, plan_text, shift


def pick_weighted(rng, weights):
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def variant_specs(rng, deg, gt_short, scale, variants, grid):
    """LQ variants of one ad: [(name, lq_short, forced codec or None)].
    Random mode: `variants` versions sharing one LQ size (drawn from degrade.lq_short when scale == "auto").
    Grid mode (dev / test): every combination of degrade.grid.lq_short x degrade.grid.codec, one version each."""
    if grid:
        g = deg["grid"]
        return [(f"lq_{s}_{c}", int(s), c) for s in g["lq_short"] for c in g["codec"]]
    if scale == "auto":                     # real inputs are 360-720p: sample the LQ short side, scale = GT / LQ
        lq_short = int(pick_weighted(rng, {int(k): v for k, v in deg["lq_short"].items()}))
    else:
        lq_short = int(round(gt_short / scale))
    return [(f"lq_{k}", lq_short, None) for k in range(variants)]


def process_sequence(seq, out_dir, gt_short, scale, variants, seed, config, grid=False, shot_files=True):
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    deg, ugc = config["degrade"], config["ugc"]
    vspecs = variant_specs(rng, deg, gt_short, scale, variants, grid)
    shots, transitions, fps = seq["shots"], seq["transitions"], seq["fps"]
    scale0 = gt_short / vspecs[0][1]
    if seq.get("aspect"):
        aspect = seq["aspect"]
    else:
        sizes = [probe(s["src"])[:2] for s in shots]
        aspects = feasible_aspects(sizes, gt_short, scale0, config["gt"]["aspects"])
        if not aspects:
            raise ValueError(f"no aspect ratio fits GT short side {gt_short} in sources {sizes}")
        aspect = pick_weighted(rng, aspects)
    gw, gh, _, _ = gt_geometry(aspect, gt_short, scale0)
    lq_sizes = []
    for _, lq_short, _ in vspecs:
        vw, vh, lw, lh = gt_geometry(aspect, gt_short, gt_short / lq_short)
        if (vw, vh) != (gw, gh):
            raise ValueError(f"LQ short side {lq_short} does not divide the {gw}x{gh} GT")
        lq_sizes.append((lw, lh))
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
    ranges = shot_ranges(shots, transitions)
    n = ranges[-1][1]
    # overlays follow the ad's style and theme; cards carry their own text, drawn by the same pass
    items, _ = plan_text(n, fps, gw, gh, rng, ugc["text"], style=seq.get("style", "ugc"), theme=seq.get("theme"),
                         brand=seq.get("brand"), palette=seq.get("palette"), cuts=[a for a, _ in ranges[1:]],
                         quiet=[r for s, r in zip(shots, ranges) if s.get("kind") == "slide"],
                         src_text=seq.get("src_text", False))
    for ren, (a, b) in zip(renderers, ranges):
        items += [shift(it, a, b) for it in ren.overlays]
    text_meta = item_meta(items, gw, gh)

    os.makedirs(out_dir, exist_ok=True)
    shared = len({ls for ls in lq_sizes}) == 1
    meta = {"id": seq["id"], "purpose": seq.get("purpose"), "fps": fps, "frames": n, "gt_size": [gw, gh],
            "lq_size": list(lq_sizes[0]) if shared else None, "scale": gt_short / vspecs[0][1] if shared else None,
            "aspect": aspect, "grid": grid, "gate": gates,
            "render": [r.info for r in renderers], "text": text_meta, "split": seq.get("split"),
            "config": config, "variants": []}
    names = [v[0] for v in vspecs]
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        prng = np.random.default_rng(rng.randint(0, 2 ** 31))
        pdeg = {**deg, "rbvsr_prob": deg["grid"].get("rbvsr_prob", 0.0)} if grid else deg
        plans = []
        for _, _, codec in vspecs:
            plan = sample_plan(prng, pdeg)
            if codec:                                   # grid: every encode of this version uses the given codec
                for comp in (plan["stage1"]["compress"], plan["final"]["compress"]):
                    if comp:
                        comp["codec"] = codec
            plans.append(plan)
        nprngs = [np.random.default_rng(rng.randint(0, 2 ** 31)) for _ in vspecs]
        gt_w = Writer(os.path.join(out_dir, "gt.mp4"), gw, gh, fps, GT_H264)
        mask_w = Writer(os.path.join(out_dir, "mask.mkv"), gw, gh, fps, ["-c:v", "ffv1"], pix_fmt="gray", in_fmt="gray")
        stage1 = [stage1_writer(p, os.path.join(tmp, f"s1_{k}.mkv"), (gw, gh), fps, g)
                  for k, (p, g) in enumerate(zip(plans, nprngs))]
        with ThreadPoolExecutor(max_workers=min(len(vspecs) + 1, os.cpu_count() or 4)) as pool:
            for i, (f, vel, camera) in enumerate(sequence_frames(shots, transitions, renderers, rng)):
                draw_text(f, i, items)
                mask = np.zeros((gh, gw), np.uint8)
                draw_mask(mask, i, items)
                v = vel if camera else (0.0, 0.0)   # motion blur only follows the virtual camera
                jobs = [pool.submit(st, f, v) for st, _ in stage1]
                gt_w.write(f)
                mask_w.write(mask)
                for (_, w), j in zip(stage1, jobs):
                    w.write(j.result())
        for w in [gt_w, mask_w] + [w for _, w in stage1]:
            w.close()
        # stage 2 and the final encode of every variant side by side (each has its own random stream)
        with ThreadPoolExecutor(max_workers=min(len(vspecs), 4)) as pool:
            jobs = []
            for k, ((name, _, _), plan, (st, _), nprng, lq_size) in enumerate(zip(vspecs, plans, stage1, nprngs, lq_sizes)):
                vt = os.path.join(tmp, f"v{k}")
                os.makedirs(vt)
                jobs.append(pool.submit(finish, plan, os.path.join(tmp, f"s1_{k}.mkv"), st.out_size,
                                        os.path.join(out_dir, f"{name}.mp4"), lq_size, fps, n, nprng, vt))
            for (name, lq_short, codec), plan, (st, _), lq_size, job in zip(vspecs, plans, stage1, lq_sizes, jobs):
                meta["variants"].append({"name": name, "file": os.path.join(out_dir, f"{name}.mp4"),
                                         "lq_size": list(lq_size), "scale": gt_short / lq_short, "codec": codec,
                                         "plan": plan, "stage1_size": list(st.out_size), **job.result()})

    if len(shots) > 1:
        cuts = [a for a, _ in ranges[1:]]
        sd = os.path.join(out_dir, "shots")
        if shot_files:
            os.makedirs(sd, exist_ok=True)
            for name in ["gt"] + names:
                split_at(os.path.join(out_dir, f"{name}.mp4"), os.path.join(sd, f"shot_%03d_{name}.mp4"), cuts)
        meta["cut_method"] = {"type": "composition", "note": "boundaries are exact, known from the sequence spec; "
                              "a dissolve's blended frames are the head of the following shot"}
        meta["shots"] = [{"index": i, "start_frame": a, "end_frame": b, "frames": b - a,
                          "source": {k: s[k] for k in ("kind", "src", "clip_id", "source_id", "category", "src_fps", "start",
                                                       "layout") if k in s},
                          "source_frames_used": s["frames"],
                          "transition_in": transitions[i - 1] if i else None,
                          "transition_out": transitions[i] if i < len(shots) - 1 else None,
                          "files": {name: os.path.join(sd, f"shot_{i:03d}_{name}.mp4") for name in ["gt"] + names}
                                   if shot_files else None}
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
    ap.add_argument("--gt-short", type=int, default=1080, help="GT short side (1080 = 1920x1080 / 1080x1920)")
    ap.add_argument("--scale", default="2", help="GT / LQ factor, or 'auto' to sample the LQ short side (degrade.lq_short)")
    ap.add_argument("--variants", type=int, default=2, help="LQ versions per GT (random mode)")
    ap.add_argument("--grid", action="store_true",
                    help="dev / test: one LQ per degrade.grid.lq_short x degrade.grid.codec (preset calibrated "
                         "unless degrade.grid.rbvsr_prob is set), named lq_<short>_<codec>")
    ap.add_argument("--no-shot-files", action="store_true",
                    help="do not write shots/ (boundaries stay in meta.json); saves most of the space for training sets")
    ap.add_argument("--max-frames", type=int, default=150, help="--manifest only: frames per clip")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n: only specs with index %% n == i, to run n processes side by side")
    ap.add_argument("--seed", type=int, default=0)
    add_config_args(ap)
    args = ap.parse_args()
    cv2.setNumThreads(4)          # variants already run side by side; 32 OpenCV threads each only add contention
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
    shard, n_shards = map(int, args.shard.split("/"))
    for i, seq in enumerate(specs):
        if i % n_shards != shard:
            continue
        out_dir = os.path.join(out, seq["id"])
        if os.path.exists(os.path.join(out_dir, "meta.json")):
            continue
        try:
            process_sequence(seq, out_dir, args.gt_short, args.scale, args.variants, args.seed * 100003 + i, config,
                             grid=args.grid, shot_files=not args.no_shot_files)
            print(f"[{i + 1}/{len(specs)}] {seq['id']}", flush=True)
        except Exception as e:
            print(f"[{i + 1}/{len(specs)}] {seq['id']} failed: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
