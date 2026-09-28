"""Turn HQ sources into UGC-ad edit specs (JSONL) for adup.make_pairs: one ad per HQ clip, edited like a real ad.

Every ad first gets a style, at the rate seen in real ads (manual annotation of 48 ads, data/stats/real_ads/ad_anatomy.csv,
docs/ugc_dataset.md §1.3; knobs in ugc.director.style.<style>):
  ugc    creator-made (about two thirds of real ads): handheld phone camera, jump cuts, punch-ins, captions; half of them
         cut between scenes (the creator talking, then B-roll of the product)
  brand  made by the brand or an agency (about a third): steady camera, more dissolves, several scenes, a headline and a
         logo, motion-graphics cards (product photo on a brand colour) and an end card with logo and CTA button
Then, for the ad's main HQ clip:
  - target aspect from config weights (half 9:16, half 16:9), never upscaling (9:16 at 1080x1920 needs a 4K landscape
    or a native portrait source); landscape clips become a portrait layout (clip in a blurred frame or on solid bars,
    or a split of two moments) at the rate layouts appear in real ads
  - frame rate drawn from real ads (30 fps for most, 24 / 25 for about a fifth)
  - scenes: the main clip alone, or with 1-3 more clips: other moments of the same source video (same creator and set)
    or clips of the same ad theme; shots switch scene every 1-3 shots
  - shots: lengths drawn from real TikTok shots; consecutive sub-shots of a scene are joined by jump cuts (a few frames
    of the source are skipped, as creators cut pauses) and alternate between normal framing and punch-in zooms
    (1.12-1.3x, taken from real source pixels)
  - handheld shake: a target shake is drawn from real ads (adup.analysis.ugc_look on TikTok / Meta) and only the part
    the source does not already have (gate CSV src_shake) is added by the virtual camera; brand ads mostly stay steady
  - optional extras: an opening text slide (hook), a motion-graphics card, an end card, a product still of the same
    theme, a picture-in-picture reaction, a 2 x 2 collage, phone screen recordings (app ads), alone, with the creator's
    face in a corner or inside a phone frame
  - transitions mostly hard cuts, otherwise dissolve / whip / dip (rendered by adup.ugc.sequence)
Each ad also gets a made-up brand name and a brand palette, shared by its logo, cards and CTA button. Text overlays are
sampled later by adup.make_pairs from the ad's style and theme. Only clips that passed the GT gate (adup.hq.gate,
--gate) are used. All knobs are in config section ugc.director (configs/pairs/<version>.yaml).

Train / dev / test are separated by source before composition (ugc.director.split): every clip gets the split of its
source video (UltraVideo's YouTube id, hashed), stills and screens get one from their own id, and an ad only uses
material of its own split. Each spec carries its "split".

With --n-ads, clips are drawn with probability proportional to (share of the clip's theme in real ads) / (number of
clips of that theme), so the dataset's content follows real UGC ads (data/stats/hq/content_coverage.csv) instead of
the HQ pool (UltraVideo is mostly food and scenery); clips of non-ad themes get a small weight. Clips are only ~5 s
long, so another ad from the same clip mostly repeats its pixels: a clip is the main clip of at most --max-per-clip
ads and appears in at most --max-uses ads in any role (extra scene, split / collage / pip part).

Usage (from the repo root):
  .venv-iqa/bin/python -m adup.ugc.director --clips data/hq/ultravideo/{4k,8k}/manifest.csv \\
      --gate data/hq/ultravideo/{4k,8k}/gate_1080.csv --stills data/stats/hq/content_unsplash.csv \\
      --screens data/hq/ui_screens/manifest.csv --source-text data/hq/kwaivir/source_text.csv \\
      --n-ads 5000 --out data/pairs/ugc_v7/specs.jsonl
Then: .venv-iqa/bin/python -m adup.make_pairs --sequences data/pairs/ugc_v7/specs.jsonl --scale auto
"""

import argparse
import collections
import hashlib
import json
import os
import random

import numpy as np
import pandas as pd

from adup.config import add_config_args, load_config
from adup.media import fits, gt_geometry, probe
from adup.paths import (
    CLIPS_TABLE,
    COVERAGE_TABLE,
    HQ,
    LOOK_TABLE,
    POOL_TABLE,
    SHOTS_TABLE,
)
from adup.ugc.text import CTAS, HOOKS, PALETTES, brand_name

SHAKE_GAIN = 1.07      # measured shake_rms per unit of planned camera shake (calibrated on a static still)


def pick(rng, weights):
    """Key of a {key: weight} dict, drawn by weight."""
    keys = list(weights)
    return keys[int(rng.choice(len(keys), p=np.array(list(weights.values()), float) / sum(weights.values())))]


def real_shake():
    """Measured shake of real ad shots (non-static shots of TikTok / Meta ads)."""
    if LOOK_TABLE.exists():
        d = pd.read_csv(LOOK_TABLE)
        d = d[d.group.isin(["tiktok", "meta"]) & (d.static_frac < 0.5)]
        return d.shake_rms.dropna().to_numpy()
    return np.exp(np.random.default_rng(0).normal(np.log(0.4), 1.2, 1000))


def shot_lengths(fps):
    """Real TikTok shot lengths, in frames at the ad's frame rate."""
    s = pd.read_csv(SHOTS_TABLE)
    return (s.frames * fps / s.fps).round().astype(int).clip(lower=10).tolist()


# stills that fit next to a clip of each UltraVideo category (themes from adup.analysis.ugc_content)
STILL_THEMES = {
    "food": ["food_drink"], "p_food": ["food_drink"], "beauty": ["beauty_product", "beauty_applying", "hair"],
    "p_cosmetics": ["beauty_product"], "p_texture": ["beauty_product"], "fashion": ["fashion", "accessories"],
    "p_fashion": ["accessories", "fashion"], "p_jewelry": ["accessories"], "pets": ["pets"], "p_electronics": ["tech"],
    "hands_product": ["beauty_product", "accessories", "tech"], "p_packaging": ["beauty_product", "food_drink"],
    "talking_head": ["beauty_product", "accessories", "food_drink", "tech"], "lifestyle": ["home", "lifestyle_outdoor"],
}


# stills for an ad theme (motion-graphics cards, and still shots of clips whose category has none above)
THEME_STILLS = {"beauty": ["beauty_product", "beauty_applying", "hair"], "fashion": ["fashion", "accessories"],
                "food": ["food_drink", "cooking"], "home": ["home", "kitchen_appliance", "cleaning"], "tech_app": ["tech"],
                "pets": ["pets"], "health": ["health", "beauty_product"], "baby_kids": ["baby_kids"],
                "fitness_sports": ["sports", "fitness"], "car": ["car"], "travel": ["travel"]}
PRODUCT_STILLS = ["beauty_product", "accessories", "food_drink", "tech", "home"]


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


def split_of(key, ratios):
    """Deterministic train / dev / test assignment of a source id by hash, with the given {split: share} ratios."""
    u = int(hashlib.md5(str(key).encode()).hexdigest()[:8], 16) / 0x100000000
    acc = 0.0
    for name, share in ratios.items():
        acc += share
        if u < acc:
            return name
    return name


def other_clip(rng, clips, clip, prefer_category=None):
    """Another clip for a split / pip part: same source video first (another moment of the shoot), else same category."""
    pools = [clips[(clips.youtube_id == clip.youtube_id) & (clips.clip_id != clip.clip_id)],
             clips[(clips.category == (prefer_category or clip.category)) & (clips.clip_id != clip.clip_id)]]
    for pool in pools:
        if len(pool):
            return pool.iloc[int(rng.integers(len(pool)))]
    return None


def extra_scenes(rng, clips, clip, k, fill, same_source_prob):
    """k more clips for a multi-scene ad: another moment of the same source video (same creator and set, the A-roll /
    B-roll of one shoot) with probability same_source_prob, else a clip of the same ad theme. fill(w, h) says whether
    a source of that size can fill the ad's frame."""
    out, seen = [], {clip.clip_id}
    same = clips[(clips.youtube_id == clip.youtube_id) & (clips.clip_id != clip.clip_id)]
    theme = clips[(clips.theme == getattr(clip, "theme", None)) & (clips.youtube_id != clip.youtube_id)] \
        if "theme" in clips else clips.iloc[:0]
    for _ in range(4 * k):
        if len(out) >= k:
            break
        pool = same if len(same) and (rng.random() < same_source_prob or not len(theme)) else theme
        if not len(pool):
            break
        c = pool.iloc[int(rng.integers(len(pool)))]
        if c.clip_id in seen:
            continue
        seen.add(c.clip_id)
        if fill(*probe(c.file)[:2]):
            out.append(c)
    return out


def camera_for(rng, clip, cfg):
    """Virtual-camera overrides for the shots of one clip, (overrides, target shake, source shake). Handheld: add only
    the shake the clip lacks to reach a shake drawn from real ads. Otherwise (steady brand footage) add none."""
    src_shake = getattr(clip, "src_shake", np.nan)
    src_pan = getattr(clip, "src_pan", np.nan)
    if src_shake is None or np.isnan(src_shake):              # no gate measurement: fall back to the metadata
        src_shake, src_pan = (0.05, 1.0) if is_static(clip) else (1.0, 10.0)
    if rng.random() < cfg["handheld_prob"]:
        target = float(rng.choice(cfg["_real_shake"]))
        amp = np.sqrt(max(target ** 2 - src_shake ** 2, 0)) / SHAKE_GAIN
        cam = {"shake_prob": 1.0, "shake_rms": [amp, amp]} if amp > 0.03 else {"shake_prob": 0.0}
    else:
        target, cam = float(src_shake), {"shake_prob": 0.0}
    if src_pan > 3:
        cam["drift_prob"] = 0.0
    return cam, target, float(src_shake)


def plan_ad(rng, clip, clips, stills, cfg, gt_short, scale, max_frames, fps, screens=None, theme=None, brand=None, palette=0,
            uses=None, max_uses=None):
    split = getattr(clip, "split", None)
    if split is not None:                          # only material of the ad's own split
        clips = clips[clips.split == split]
        stills = stills[stills.split == split] if stills is not None else None
        screens = (screens or {}).get(split)
    if uses is not None:                           # extra scenes and layout parts only from clips not used up yet
        clips = clips[clips.clip_id.map(lambda c: uses.get(c, 0) < max_uses)]
    lengths = shot_lengths(fps)
    sw, sh, src_fps, nb = probe(clip.file)
    ok = {a: w for a, w in cfg["aspects"].items() if fits(sw, sh, a, gt_short, scale)}
    want = pick(rng, cfg["aspects"])
    layout = None
    if want == "9:16" and sw > sh and rng.random() < cfg["portrait_layout_prob"]:
        layout = pick(rng, cfg["portrait_layouts"])         # a landscape clip in a portrait layout, at the real rate
    elif want not in ok:
        if not ok:
            return None
        want = pick(rng, ok)
    fill = (lambda w, h: w > h) if layout else (lambda w, h: fits(w, h, want, gt_short, scale))
    extras = []
    if rng.random() < cfg["multi_scene_prob"]:
        extras = extra_scenes(rng, clips, clip, int(rng.integers(cfg["scenes"][0], cfg["scenes"][1] + 1)) - 1, fill,
                              cfg["same_source_prob"])
    scenes = []
    for i, c in enumerate([clip] + extras):
        c_fps, c_nb = (src_fps, nb) if i == 0 else probe(c.file)[2:]
        c_nb = c_nb or c.frames
        cam, target, src_shake = camera_for(rng, c, cfg)
        scenes.append({"clip": c, "fps": c_fps, "nb": c_nb, "pos": 0.0 if i == 0 else float(rng.uniform(0, 0.3) * c_nb),
                       "cam": cam, "target": target, "src_shake": src_shake, "zoomed": rng.random() < 0.5, "done": False})
    total = min(max_frames, int(0.85 * sum(sc["nb"] * fps / sc["fps"] for sc in scenes)))
    shots, transitions, used = [], [], 0
    still_pool = card_pool = None
    if stills is not None:
        still_pool = stills[stills.theme.isin(STILL_THEMES.get(clip.category) or THEME_STILLS.get(theme, []))]
        card_pool = stills[stills.theme.isin(THEME_STILLS.get(theme, PRODUCT_STILLS))]
        card_pool = card_pool if len(card_pool) else stills[stills.theme.isin(PRODUCT_STILLS)]

    def video_shot(sc, start, n, zoom, c=None, c_fps=None):
        c = sc["clip"] if c is None else c
        s = {"kind": "video", "src": c.file, "clip_id": c.clip_id, "source_id": c.youtube_id, "category": c.category,
             "src_fps": float(c_fps or sc["fps"]), "start": int(start), "frames": int(n), "camera": sc["cam"]}
        if zoom > 1:
            s["zoom"] = float(zoom)
        return s

    def still_src(pool=None):
        pool = still_pool if pool is None else pool
        if pool is None or not len(pool):
            return None
        st = pool.iloc[int(rng.integers(len(pool)))]
        return f"unsplash:{st.photo_id}|{st.photo_image_url}|{st.photo_width}x{st.photo_height}"

    def card(role, n, text=None, image_prob=0.0):
        s = {"kind": "slide", "role": role, "frames": int(n), "palette": palette, "brand": brand, "theme": theme}
        if text:
            s["text"] = text
        if rng.random() < image_prob and (img := still_src(card_pool)):
            s["image"] = img
        return s

    if rng.random() < cfg["hook_slide_prob"]:
        n = int(rng.integers(20, 45))
        shots.append({"kind": "slide", "role": "hook", "frames": n, "text": str(rng.choice(HOOKS))})
        used += n
    end_card = int(rng.integers(30, 60)) if rng.random() < cfg["end_card_prob"] else 0
    total -= end_card
    p_screen = cfg["screen_ad_prob"] * (4 if theme == "tech_app" else 1)          # app ads are mostly screen recordings
    screen_ad = want == "9:16" and screens is not None and len(screens) and rng.random() < p_screen
    n_exp = max(int(total / np.median(lengths)), 2)                # expected number of shots
    screen_at = set(rng.choice(n_exp, size=min(int(rng.integers(1, 3)), n_exp), replace=False).tolist()) if screen_ad else set()
    card_at = int(rng.integers(1, n_exp)) if rng.random() < cfg["card_prob"] else -1
    collage = []
    if want == "9:16" and not layout and rng.random() < cfg["collage_prob"]:
        cw, ch = gt_geometry("9:16", gt_short, scale)[:2]
        collage = extra_scenes(rng, clips, clip, 3, lambda w, h: min(w, h) >= min(cw, ch) // 2, 0.3)
    collage_at = int(rng.integers(0, n_exp)) if len(collage) == 3 else -1
    cur, left = 0, int(rng.integers(1, 4))
    while used < total:
        live = [i for i, sc in enumerate(scenes) if not sc["done"]]
        if not live:
            break
        if cur not in live or left <= 0:                       # next scene (A-roll / B-roll), 1-3 shots each
            cur = live[(live.index(cur) + 1) % len(live)] if cur in live else live[0]
            left = int(rng.integers(1, 4))
        sc = scenes[cur]
        n = int(min(rng.choice(lengths), total - used))
        if n < 10:
            break
        src_frames = n * sc["fps"] / fps
        if sc["pos"] + src_frames > sc["nb"] - 2:
            sc["done"], left = True, 0
            continue
        zoom = float(rng.uniform(*cfg["punch_in_zoom"])) if (sc["zoomed"] and rng.random() < cfg["punch_in_prob"] * 2) else 1.0
        k = len(shots)
        if k in screen_at:
            scr = {"kind": "screen", "src": str(screens[int(rng.integers(len(screens)))]), "frames": n}
            r = rng.random()
            if clip.category == "talking_head" and r < cfg["screen_pip_prob"]:
                s = {"kind": "layout", "layout": "pip", "frames": n, "parts": [scr, video_shot(sc, sc["pos"], n, 1.0)]}
            elif r < cfg["screen_pip_prob"] + cfg["screen_phone_prob"]:
                behind = [video_shot(sc, sc["pos"], n, 1.0)] if not layout and rng.random() < 0.5 else []
                s = {"kind": "layout", "layout": "phone", "frames": n, "palette": palette, "parts": [scr] + behind}
            else:
                s = scr
        elif k == card_at and (nc := min(int(rng.uniform(1.2, 2.5) * fps), total - used, int(0.35 * total))) >= fps:
            s = card("card", nc, image_prob=0.8)          # cards last 1-2.5 s and take at most a third of the ad
            n = s["frames"]
        elif k == collage_at:
            parts = [video_shot(sc, sc["pos"], n, 1.0)]
            for c in collage:
                c_fps = probe(c.file)[2]
                parts.append(video_shot(sc, rng.uniform(0, max(c.frames - n * c_fps / fps - 2, 0)), n, 1.0, c, c_fps))
            s = {"kind": "layout", "layout": "grid4", "frames": n, "parts": parts}
        elif layout:
            parts = [video_shot(sc, sc["pos"], n, zoom)]
            if layout in ("split", "duet"):
                oc = other_clip(rng, clips, sc["clip"])
                if oc is not None:
                    o_fps = probe(oc.file)[2]
                    o_room = max(oc.frames - n * o_fps / fps - 2, 0)
                    parts.append(video_shot(sc, rng.uniform(0, o_room), n, 1.0, oc, o_fps))
                else:
                    other = max(0.0, sc["nb"] - sc["pos"] - 2 * src_frames)
                    parts.append(video_shot(sc, sc["pos"] + rng.uniform(0, other) if other else sc["pos"], n, 1.0))
            s = {"kind": "layout", "layout": str(layout), "frames": n, "parts": parts, "palette": palette}
        elif shots and rng.random() < cfg["still_prob"] and (img := still_src()):
            s = {"kind": "still", "src": img, "frames": n}
        elif rng.random() < cfg["pip_prob"] and shots and (oc := other_clip(rng, clips, sc["clip"], "talking_head")) is not None:
            o_fps = probe(oc.file)[2]
            s = {"kind": "layout", "layout": "pip", "frames": n,
                 "parts": [video_shot(sc, sc["pos"], n, zoom),
                           video_shot(sc, rng.uniform(0, max(oc.frames - n * o_fps / fps - 2, 0)), n, 1.0, oc, o_fps)]}
        else:
            s = video_shot(sc, sc["pos"], n, zoom)
        if shots:
            if s.get("kind") == "video" and shots[-1].get("clip_id") == s["clip_id"] and rng.random() < cfg["jump_cut_prob"]:
                kind, t = "cut", 0                              # jump cut within a scene
            else:
                kind = pick(rng, cfg["transitions"])
                t = int(rng.integers(*cfg["transition_frames"][kind])) if kind != "cut" else 0
                t = min(t, n - 4, shots[-1]["frames"] - 4) if t else 0
            transitions.append({"type": str(kind) if t > 0 else "cut", "frames": max(t, 0)})
            used -= transitions[-1]["frames"] if kind == "dissolve" else 0
        shots.append(s)
        used += n
        sc["pos"] += src_frames + (rng.uniform(*cfg["jump_skip_s"]) * sc["fps"] if rng.random() < cfg["jump_cut_prob"] else 0)
        sc["zoomed"] = not sc["zoomed"]
        left -= 1
    if not shots or sum(1 for s in shots if s["kind"] != "slide") == 0:
        return None
    if end_card:
        transitions.append({"type": pick(rng, {"cut": 0.7, "dissolve": 0.3}) if end_card > 20 else "cut", "frames": 0})
        if transitions[-1]["type"] == "dissolve":
            transitions[-1]["frames"] = int(min(rng.integers(4, 10), shots[-1]["frames"] - 4))
            used -= transitions[-1]["frames"]
        shots.append(card("end", end_card, text=str(rng.choice(CTAS)), image_prob=0.4))
        used += end_card
    if sum(s["frames"] for s in shots if s["kind"] == "slide") > 0.4 * used:
        return None                                # mostly cards: the footage ran out (real ads: ~7% of frames are cards)
    return {"aspect": str(want), "shots": shots, "transitions": transitions, "frames": used, "portrait_layout": layout,
            "scenes": len(scenes), "shake_target": round(scenes[0]["target"], 3), "src_shake": round(scenes[0]["src_shake"], 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True, help="HQ clip manifests (data/hq/*/manifest.csv)")
    ap.add_argument("--gate", nargs="*", default=[], help="gate CSVs (adup.hq.gate); clips must pass")
    ap.add_argument("--stills", help="stills table (data/stats/hq/content_unsplash.csv)")
    ap.add_argument("--screens", help="UI screen manifest (data/hq/ui_screens/manifest.csv), for screen-recording shots")
    ap.add_argument("--source-text", nargs="*", default=[],
                    help="source_text.csv of adup.hq.source_text: ads using clips with burned-in text get no captions")
    ap.add_argument("--gt-short", type=int, default=1080)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--per-clip", type=int, default=1, help="ads planned per HQ clip (without --n-ads)")
    ap.add_argument("--n-ads", type=int, default=0, help="draw this many ads, clips weighted towards real ad themes")
    ap.add_argument("--max-per-clip", type=int, default=2,
                    help="with --n-ads: at most this many ads per HQ clip (clips are ~5 s, so a third ad mostly repeats pixels)")
    ap.add_argument("--max-uses", type=int, default=3,
                    help="at most this many ads per HQ clip in any role (main clip, extra scene, split / collage / pip part)")
    ap.add_argument("--only-split", choices=["train", "dev", "test"], help="plan ads only from sources of this split")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="specs file, normally data/pairs/<dataset>/specs.jsonl")
    add_config_args(ap)
    args = ap.parse_args()
    config = load_config(args.config, args.set)
    dcfg = config["ugc"]["director"]
    rng = np.random.default_rng(args.seed)
    shake = real_shake()

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
    ratios = dcfg["split"]
    clips["split"] = clips.youtube_id.map(lambda k: split_of(k, ratios))
    if stills is not None:
        stills["split"] = stills.photo_id.map(lambda k: split_of(k, ratios))
    if args.only_split:
        clips = clips[clips.split == args.only_split]
    if args.limit:
        clips = clips.sample(min(args.limit, len(clips)), random_state=args.seed)
    screens = None
    if args.screens:
        sm = pd.read_csv(args.screens)
        need = gt_geometry("9:16", args.gt_short, args.scale)[0]
        ok = [f for f in sm.file if os.path.exists(f) and __import__("cv2").imread(f).shape[1] >= need]
        screens = {}
        for f in ok:
            screens.setdefault(split_of(os.path.basename(f), ratios), []).append(f)
    n_out = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    clips = clips.reset_index(drop=True)
    w, themes = theme_weights(clips, dcfg["other_theme_share"])
    clips["theme"] = themes if themes is not None else "none"
    with_text = set()
    for t in args.source_text:
        d = pd.read_csv(t)
        with_text |= set(d.file[d.has_text.astype(bool)])
    # clips are drawn one ad at a time, so that every use (main clip or part of another ad) counts towards the caps
    uses, mains, styles, drawn = collections.Counter(), collections.Counter(), {}, []
    row = {c: i for i, c in reversed(list(enumerate(clips.clip_id)))}

    def next_clip():
        if args.n_ads:
            while n_out < args.n_ads and (w > 0).any():
                yield clips.iloc[int(rng.choice(len(clips), p=w / w.sum()))]
        else:
            for c in clips.itertuples():
                yield from [c] * args.per_clip

    with open(args.out, "w") as f:
        for clip in next_clip():
            k = mains[clip.clip_id]
            mains[clip.clip_id] += 1
            drawn.append(clip.theme)
            style = pick(rng, dcfg["styles"])
            cfg = {**dcfg, **dcfg["style"][style], "_real_shake": shake}
            fps = int(pick(rng, dcfg["fps"]))
            theme = clip.theme if clip.theme not in ("none", "other", "talking_head") else None
            brand = brand_name(random.Random(int(rng.integers(2 ** 31))))
            palette = int(rng.integers(len(PALETTES)))
            plan = plan_ad(rng, clip, clips, stills, cfg, args.gt_short, args.scale, args.max_frames, fps, screens, theme,
                           brand, palette, uses, args.max_uses)
            used = set()
            if plan is not None:
                parts = [p for s in plan["shots"] for p in [s, *s.get("parts", [])]]
                used = {p["clip_id"] for p in parts if p.get("clip_id")}
                uses.update(used)
                srcs = {p.get("src") for p in parts}
                spec = {"id": f"ugc_{clip.clip_id.replace('.mp4', '')}_{k}", "purpose": "ugc", "split": clip.split,
                        "fps": fps, "style": style, "theme": theme, "brand": brand, "palette": palette,
                        "src_text": bool(srcs & with_text), "pregated": bool(args.gate), **plan}
                f.write(json.dumps(spec) + "\n")
                n_out += 1
                styles[style] = styles.get(style, 0) + 1
            if args.n_ads:
                for c in used | {clip.clip_id}:
                    if mains[c] >= args.max_per_clip or uses[c] >= args.max_uses:
                        w[row[c]] = 0.0
    print("ad themes drawn:", pd.Series(drawn).value_counts().to_dict())
    print(f"clips used: {len(uses)} of {len(clips)}; uses per clip {dict(sorted(collections.Counter(uses.values()).items()))}")
    print(f"{n_out} ad specs from {len(clips)} clips -> {args.out}; styles {styles}")


if __name__ == "__main__":
    main()
