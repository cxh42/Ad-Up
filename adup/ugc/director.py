"""Turn HQ sources into UGC-ad edit specs (JSONL) for adup.make_pairs: one ad per HQ clip, edited like a creator.

For each HQ clip the director plans a short vertical (or square / landscape) ad the way UGC ads are cut:
  - target aspect: 9:16 when the source can supply it at the GT size, otherwise 9:16 through a layout (landscape clip
    in a blurred frame, or split / duet of two moments), or 16:9 / 4:5 / 1:1 (config weights, never upscaling)
  - shots: the clip is cut into sub-shots with lengths drawn from real TikTok shots; consecutive sub-shots are joined
    by jump cuts (a few frames of the source are skipped, as creators cut pauses) and alternate between normal
    framing and punch-in zooms (1.12-1.3x, taken from real source pixels)
  - handheld shake: a target shake is drawn from real ads (adup.analysis.ugc_look on TikTok / Meta) and only the
    part the source does not already have (gate CSV src_shake) is added by the virtual camera
  - optional extras: an opening text slide (hook), a product still of the same theme, a picture-in-picture reaction,
    phone screen recordings (app ads: ~15% of real ad frames), alone or with the creator's face in a corner
  - transitions mostly hard cuts, otherwise dissolve / whip / dip (rendered by adup.ugc.sequence)
All knobs are in config section ugc.director (configs/pairs/<version>.yaml). Text overlays and the phone look are
sampled later by adup.make_pairs. Only clips that passed the GT gate (adup.hq.gate, --gate) are used.

With --n-ads, clips are drawn with probability proportional to (share of the clip's theme in real ads) / (number of
clips of that theme), so the dataset's content follows real UGC ads (data/stats/hq/content_coverage.csv) instead of
the HQ pool (UltraVideo is mostly food and scenery); clips of non-ad themes get a small weight.

Usage (from the repo root):
  .venv-iqa/bin/python -m adup.ugc.director --clips data/hq/ultravideo/4k/manifest.csv \
      --gate data/hq/ultravideo/4k/gate_1440.csv --stills data/stats/hq/content_unsplash.csv \
      --screens data/hq/ui_screens/manifest.csv --gt-short 1440 --n-ads 5000 --out data/pairs/ugc_v5_2k/specs.jsonl
Then: .venv-iqa/bin/python -m adup.make_pairs --sequences data/pairs/ugc_v5_2k/specs.jsonl --scale auto
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from adup.config import add_config_args, load_config
from adup.media import gt_geometry, max_crop, probe
from adup.paths import CLIPS_TABLE, COVERAGE_TABLE, HQ, LOOK_TABLE, POOL_TABLE, SHOTS_TABLE

SEQ_FPS = 30
HOOKS = ["wait for it", "POV: you finally found it", "3 reasons you need this", "I was today years old", "don't skip this",
         "this changed everything", "honest review", "run don't walk", "day 1 vs day 30", "the viral one", "before vs after",
         "is it worth it?", "ok but why is no one talking about this", "watch till the end", "my holy grail"]

SHAKE_GAIN = 1.07      # measured shake_rms per unit of planned camera shake (calibrated on a static still)


def real_shake():
    """Measured shake of real ad shots (non-static shots of TikTok / Meta ads)."""
    if LOOK_TABLE.exists():
        d = pd.read_csv(LOOK_TABLE)
        d = d[d.group.isin(["tiktok", "meta"]) & (d.static_frac < 0.5)]
        return d.shake_rms.dropna().to_numpy()
    return np.exp(np.random.default_rng(0).normal(np.log(0.4), 1.2, 1000))


def shot_lengths():
    s = pd.read_csv(SHOTS_TABLE)
    return (s.frames * SEQ_FPS / s.fps).round().astype(int).clip(lower=10).tolist()


def feasible(sw, sh, aspect, gt_short, scale):
    gw, gh, _, _ = gt_geometry(aspect, gt_short, scale)
    return max_crop(sw, sh, gw, gh)[0] >= gw


# stills that fit next to a clip of each UltraVideo category (themes from adup.analysis.ugc_content)
STILL_THEMES = {
    "food": ["food_drink"], "p_food": ["food_drink"], "beauty": ["beauty_product", "beauty_applying", "hair"],
    "p_cosmetics": ["beauty_product"], "p_texture": ["beauty_product"], "fashion": ["fashion", "accessories"],
    "p_fashion": ["accessories", "fashion"], "p_jewelry": ["accessories"], "pets": ["pets"], "p_electronics": ["tech"],
    "hands_product": ["beauty_product", "accessories", "tech"], "p_packaging": ["beauty_product", "food_drink"],
    "talking_head": ["beauty_product", "accessories", "food_drink", "tech"], "lifestyle": ["home", "lifestyle_outdoor"],
}


# UltraVideo pool themes (adup.analysis.ugc_content) -> ad themes of content_coverage.csv
POOL_TO_AD = {"beauty_applying": "beauty", "beauty_product": "beauty", "hair": "beauty", "fashion": "fashion",
              "accessories": "fashion", "food_drink": "food", "cooking": "food", "eating": "food", "home": "home",
              "cleaning": "home", "kitchen_appliance": "home", "health": "health", "pets": "pets", "tech": "tech_app",
              "baby_kids": "baby_kids", "fitness": "fitness_sports", "sports": "fitness_sports", "car": "car",
              "talking_head": "talking_head", "hands_product": "talking_head", "unboxing": "talking_head",
              "travel": "travel", "office_work": "education", "lifestyle_outdoor": "other", "shopping": "other"}


def theme_weights(clips, other_share):
    """Sampling weights and ad theme per clip. The theme comes from CLIP on the clip itself (content_clips.csv) when
    available, else from the text description (content_pool.csv). Clips of no ad theme share `other_share` in total."""
    cov, pool, vis = COVERAGE_TABLE, POOL_TABLE, CLIPS_TABLE
    if not cov.exists():
        return np.ones(len(clips)) / len(clips), None
    share = pd.read_csv(cov).set_index("theme")["real_share_%"] / 100
    t = pd.Series("none", index=clips.index)
    if pool.exists():
        themes = pd.read_csv(pool, usecols=["clip_id", "theme"]).set_index("clip_id")["theme"]
        t = clips.clip_id.map(themes).map(POOL_TO_AD).fillna("none")
    if vis.exists():
        v = pd.read_csv(vis).set_index("file")["theme"]
        t = clips.file.map(v).fillna(t)
    # talking heads appear across all ad categories (45% of real shots have a face): give them the "other" share too
    s = t.map(lambda x: share.get(x, 0.0) + (share.get("other", 0) if x == "talking_head" else 0))
    w = (s / t.map(t.value_counts())).where(t != "none", 0.0)
    none = t == "none"
    if none.any() and w.sum() > 0:                  # clips of no ad theme: other_share of the total, whatever the pool
        w[none] = other_share / (1 - other_share) * w.sum() / none.sum()
    return np.array(w / w.sum(), dtype=float), t


def is_static(clip):
    cm = str(getattr(clip, "camera_movement", "")).lower()
    return any(w in cm for w in ("stationary", "static", "fixed", "minimal", "still"))


def other_clip(rng, clips, clip, prefer_category=None):
    """Another clip for a split / pip part: same source video first (another moment of the shoot), else same category."""
    pools = [clips[(clips.youtube_id == clip.youtube_id) & (clips.clip_id != clip.clip_id)],
             clips[(clips.category == (prefer_category or clip.category)) & (clips.clip_id != clip.clip_id)]]
    for pool in pools:
        if len(pool):
            return pool.iloc[int(rng.integers(len(pool)))]
    return None


def plan_ad(rng, clip, clips, stills, lengths, cfg, gt_short, scale, max_frames, screens=None, theme=None):
    sw, sh, fps, nb = probe(clip.file)
    avail = int((nb or clip.frames) * SEQ_FPS / fps)
    total = min(max_frames, int(avail * 0.85))
    ok = {a: w for a, w in cfg["aspects"].items() if feasible(sw, sh, a, gt_short, scale)}
    want = rng.choice(list(cfg["aspects"]), p=np.array(list(cfg["aspects"].values())) / sum(cfg["aspects"].values()))
    layout = None
    if want not in ok:
        if want == "9:16" and sw > sh and rng.random() < cfg["portrait_layout_prob"]:
            lay = cfg["portrait_layouts"]
            layout = rng.choice(list(lay), p=np.array(list(lay.values())) / sum(lay.values()))
        elif ok:
            want = rng.choice(list(ok), p=np.array(list(ok.values())) / sum(ok.values()))
        else:
            return None
    src_shake = getattr(clip, "src_shake", np.nan)
    src_pan = getattr(clip, "src_pan", np.nan)
    if np.isnan(src_shake):                                   # no gate measurement: fall back to the metadata
        src_shake, src_pan = (0.05, 1.0) if is_static(clip) else (1.0, 10.0)
    target = float(rng.choice(cfg["_real_shake"]))
    amp = np.sqrt(max(target ** 2 - src_shake ** 2, 0)) / SHAKE_GAIN
    cam = {"shake_prob": 1.0, "shake_rms": [amp, amp]} if amp > 0.03 else {"shake_prob": 0.0}
    if src_pan > 3:
        cam["drift_prob"] = 0.0
    shots, transitions, used, src_pos = [], [], 0, 0.0

    def video_shot(start, n, zoom, c=clip, c_fps=None):
        s = {"kind": "video", "src": c.file, "clip_id": c.clip_id, "source_id": c.youtube_id,
             "category": c.category, "src_fps": float(c_fps or fps), "start": int(start), "frames": int(n), "camera": cam}
        if zoom > 1:
            s["zoom"] = float(zoom)
        return s

    if rng.random() < cfg["hook_slide_prob"]:
        n = int(rng.integers(20, 45))
        shots.append({"kind": "slide", "frames": n, "text": str(rng.choice(HOOKS))})
        used += n
    zoomed = rng.random() < 0.5
    p_screen = cfg["screen_ad_prob"] * (4 if theme == "tech_app" else 1)          # app ads are mostly screen recordings
    screen_ad = want == "9:16" and screens is not None and len(screens) and rng.random() < p_screen
    n_exp = max(int(total / np.median(lengths)), 2)                # expected number of shots
    screen_at = set(rng.choice(n_exp, size=min(int(rng.integers(1, 3)), n_exp), replace=False).tolist()) if screen_ad else set()
    while used < total:
        n = int(min(rng.choice(lengths), total - used))
        if n < 10:
            break
        src_frames = n * fps / SEQ_FPS
        if src_pos + src_frames > (nb or clip.frames) - 2:
            break
        zoom = float(rng.uniform(*cfg["punch_in_zoom"])) if (zoomed and rng.random() < cfg["punch_in_prob"] * 2) else 1.0
        if len(shots) in screen_at:
            sc = {"kind": "screen", "src": str(screens[int(rng.integers(len(screens)))]), "frames": n}
            s = ({"kind": "layout", "layout": "pip", "frames": n, "parts": [sc, video_shot(src_pos, n, 1.0)]}
                 if clip.category == "talking_head" and rng.random() < cfg["screen_pip_prob"] else sc)
        elif layout:
            other = max(0.0, (nb or clip.frames) - src_pos - 2 * src_frames)
            parts = [video_shot(src_pos, n, zoom)]
            if layout in ("split", "duet"):
                oc = other_clip(rng, clips, clip)
                if oc is not None:
                    o_fps = probe(oc.file)[2]
                    o_room = max(oc.frames - n * o_fps / SEQ_FPS - 2, 0)
                    parts.append(video_shot(rng.uniform(0, o_room), n, 1.0, oc, o_fps))
                else:
                    parts.append(video_shot(src_pos + rng.uniform(0, other) if other else src_pos, n, 1.0))
            s = {"kind": "layout", "layout": str(layout), "frames": n, "parts": parts}
        elif (stills is not None and rng.random() < cfg["still_prob"] and shots
              and len(pool := stills[stills.theme.isin(STILL_THEMES.get(clip.category, []))])):
            st = pool.iloc[int(rng.integers(len(pool)))]
            s = {"kind": "still", "src": f"unsplash:{st.photo_id}|{st.photo_image_url}", "frames": n}
        elif rng.random() < cfg["pip_prob"] and shots and (oc := other_clip(rng, clips, clip, "talking_head")) is not None:
            o_fps = probe(oc.file)[2]
            s = {"kind": "layout", "layout": "pip", "frames": n,
                 "parts": [video_shot(src_pos, n, zoom),
                           video_shot(rng.uniform(0, max(oc.frames - n * o_fps / SEQ_FPS - 2, 0)), n, 1.0, oc, o_fps)]}
        else:
            s = video_shot(src_pos, n, zoom)
        if shots:
            if rng.random() < cfg["jump_cut_prob"]:
                kind, k = "cut", 0
            else:
                tr = cfg["transitions"]
                kind = rng.choice(list(tr), p=np.array(list(tr.values())) / sum(tr.values()))
                k = int(rng.integers(*cfg["transition_frames"][kind])) if kind != "cut" else 0
                k = min(k, n - 4, shots[-1]["frames"] - 4) if k else 0
            transitions.append({"type": str(kind) if k > 0 else "cut", "frames": max(k, 0)})
            used -= transitions[-1]["frames"] if kind == "dissolve" else 0
        shots.append(s)
        used += n
        src_pos += src_frames + (rng.uniform(*cfg["jump_skip_s"]) * fps if rng.random() < cfg["jump_cut_prob"] else 0)
        zoomed = not zoomed
    if not shots or sum(1 for s in shots if s["kind"] != "slide") == 0:
        return None
    return {"aspect": str(want), "shots": shots, "transitions": transitions, "frames": used, "portrait_layout": layout,
            "shake_target": round(target, 3), "src_shake": round(float(src_shake), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True, help="HQ clip manifests (data/hq/*/manifest.csv)")
    ap.add_argument("--gate", nargs="*", default=[], help="gate CSVs (adup.hq.gate); clips must pass")
    ap.add_argument("--stills", help="stills table (data/stats/hq/content_unsplash.csv)")
    ap.add_argument("--screens", help="UI screen manifest (data/hq/ui_screens/manifest.csv), for screen-recording shots")
    ap.add_argument("--gt-short", type=int, default=1440)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--per-clip", type=int, default=1, help="ads planned per HQ clip (without --n-ads)")
    ap.add_argument("--n-ads", type=int, default=0, help="draw this many ads, clips weighted towards real ad themes")
    ap.add_argument("--max-per-clip", type=int, default=3, help="with --n-ads: at most this many ads per HQ clip")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="specs file, normally data/pairs/<dataset>/specs.jsonl")
    add_config_args(ap)
    args = ap.parse_args()
    cfg = {**load_config(args.config, args.set)["ugc"]["director"], "_real_shake": real_shake()}
    rng = np.random.default_rng(args.seed)

    clips = pd.concat([pd.read_csv(m) for m in args.clips], ignore_index=True)
    clips = clips[clips.file.map(os.path.exists)]
    if args.gate:
        g = pd.concat([pd.read_csv(x) for x in args.gate])
        ok = g.groupby("file")["pass"].any()
        clips = clips[clips.file.map(ok).fillna(False).astype(bool)]
        mo = g[g["pass"]].groupby("file")[["src_shake", "src_pan"]].median() if "src_shake" in g else None
        if mo is not None:
            clips = clips.merge(mo, left_on="file", right_index=True, how="left")
    short = HQ / "ultravideo" / "short.csv"
    if short.exists():
        cm = pd.read_csv(short, usecols=["clip_id", "Camera Movement"]).rename(columns={"Camera Movement": "camera_movement"})
        clips = clips.merge(cm, on="clip_id", how="left")
    stills = None
    if args.stills:
        stills = pd.read_csv(args.stills)
        stills = stills[stills.theme != "other"]
    if args.limit:
        clips = clips.sample(min(args.limit, len(clips)), random_state=args.seed)
    lengths = shot_lengths()
    screens = None
    if args.screens:
        sm = pd.read_csv(args.screens)
        need = gt_geometry("9:16", args.gt_short, args.scale)[0]
        screens = [f for f in sm.file if os.path.exists(f) and __import__("cv2").imread(f).shape[1] >= need] if len(sm) else None
    n_out = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    clips = clips.reset_index(drop=True)
    if args.n_ads:
        w, themes = theme_weights(clips, cfg["other_theme_share"])
        picks, used_n = [], np.zeros(len(clips), int)
        while len(picks) < args.n_ads and (w > 0).any():
            i = int(rng.choice(len(clips), p=w / w.sum()))
            picks.append(i)
            used_n[i] += 1
            if used_n[i] >= args.max_per_clip:
                w[i] = 0.0
        picks = np.array(picks)
        if themes is not None:
            print("ad themes drawn:", themes.iloc[picks].value_counts().to_dict())
        order = [(clips.iloc[i], int((picks[:j] == i).sum()), themes.iloc[i] if themes is not None else None)
                 for j, i in enumerate(picks)]
    else:
        order = [(c, k, None) for c in clips.itertuples() for k in range(args.per_clip)]
    with open(args.out, "w") as f:
        for clip, k, theme in order:
            plan = plan_ad(rng, clip, clips, stills, lengths, cfg, args.gt_short, args.scale, args.max_frames, screens, theme)
            if plan is None:
                continue
            spec = {"id": f"ugc_{clip.clip_id.replace('.mp4', '')}_{k}", "purpose": "ugc", "fps": SEQ_FPS,
                    "pregated": bool(args.gate), **plan}
            f.write(json.dumps(spec) + "\n")
            n_out += 1
    print(f"{n_out} ad specs from {len(clips)} clips -> {args.out}")


if __name__ == "__main__":
    main()
