"""Burn realistic on-screen text into GT frames: running captions, hook titles, stickers and fine print.

Text is drawn on the GT, before any degradation, because in real ads it is burned in by the editing app or at upload
and then goes through every later encode. Distributions follow docs/ugc_degradations.md §2 (OCR on 110 real TikTok and
Meta ads): ~95% of ads carry text in almost every frame, OCR box height is 1.6–6.6% of the frame height, and captions
cluster in the lower-middle of the frame. Styles are the families seen in those ads (and in CapCut / TikTok / caption
apps): plain white sans, rounded plates, bold caps with a highlighted current word, and karaoke-style plates.

Everything is drawn from masks and composited with straight alpha, so anti-aliased edges carry no dark fringes.
"""

import glob
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

from adup.degrade import fonts
from adup.paths import ADS

FALLBACK_COPY = [
    "I was today years old when I found this", "ok but why is nobody talking about this", "this changed my whole routine",
    "honestly obsessed with how this turned out", "three reasons you need this in your life", "run don't walk to get this",
    "POV you finally found the one that works", "day seven update and I am shocked", "wait for the before and after",
    "this is your sign to try it", "I tested it so you don't have to", "my new favorite thing this month",
    "the texture is actually insane", "no filter no edits just this", "everyone keeps asking me what I use",
]
EMOJIS = list("😍🔥✨💯😭🙌👀💕😂🤯✅⭐🛒💪🥰😱👇🎉💖🤩")
STICKERS = ["50% OFF", "SALE", "FREE SHIPPING", "Shop Now >", "Link in bio", "#ad", "Paid partnership", "NEW", "BEST SELLER",
            "Limited time only", "RETAIL: $27", "$29.99", "BOGO", "Only $19", "4.8 stars", "Use code SAVE20",
            "Buy 1 Get 1 FREE", "TikTok Made Me Buy It", "As seen on TV", "Get yours now", "Sign up today"]
FINE_PRINT = ["Results may vary. Individual results not guaranteed.", "Paid partnership. #ad",
              "These statements have not been evaluated by the Food and Drug Administration.",
              "Terms apply. See site for details.", "Offer valid while supplies last.", "Dramatization. Not actual results."]

WHITE, BLACK, GREY = (255, 255, 255), (0, 0, 0), (150, 150, 150)
HIGHLIGHTS = [(247, 194, 4), (255, 230, 0), (2, 251, 35), (46, 204, 113), (0, 200, 255), (254, 44, 85), (255, 140, 0),
              (255, 64, 160), (160, 90, 255)]
PLATES = [(WHITE, BLACK), (WHITE, BLACK), (WHITE, BLACK), (BLACK, WHITE), ((254, 44, 85), WHITE), ((46, 170, 90), WHITE),
          ((255, 221, 0), BLACK), ((30, 30, 30), WHITE), ((230, 230, 255), (40, 40, 120))]


# ---------------------------------------------------------------- copy
@lru_cache(maxsize=1)
def ad_copy():
    """Sentences from the scraped ads' own copy (TikTok titles, Meta bodies), falling back to generic lines."""
    lines = []
    for f in glob.glob(str(ADS / "*" / "*" / "summary.csv")):
        df = pd.read_csv(f)
        for col in ("ad_title", "body"):
            if col in df:
                lines += df[col].dropna().astype(str).tolist()
    sents = []
    for t in lines:
        t = re.sub(r"https?://\S+|www\.\S+|@\w+|#\w+|\{\{.*?\}\}", " ", t)
        t = t.translate(str.maketrans("‘’“”–—…", "''\"\"--."))
        t = re.sub(r"[^\x00-\x7F]+", " ", t)              # emoji are drawn separately; fonts are Latin-only
        for s in re.split(r"(?<=[.!?])\s+|\n+", t):
            words = s.split()
            if 3 <= len(words) <= 30:
                sents.append(" ".join(words))
    return sents + FALLBACK_COPY


def word_stream(rng, n_words):
    corpus = ad_copy()
    words = []
    while len(words) < n_words:
        words += rng.choice(corpus).split()
    return words[:n_words]


# ---------------------------------------------------------------- drawing
@lru_cache(maxsize=64)
def emoji_image(ch, size):
    im = Image.new("RGBA", (fonts.EMOJI_SIZE * 2, fonts.EMOJI_SIZE * 2), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((0, 0), ch, font=fonts.emoji_font(), embedded_color=True)
    im = im.crop(im.getbbox() or (0, 0, 1, 1))
    return im.resize((size, max(int(size * im.height / max(im.width, 1)), 1)), Image.LANCZOS)


def colored(mask, color):
    layer = Image.new("RGBA", mask.size, tuple(color) + (0,))
    layer.putalpha(mask)
    return layer


def draw_block(lines, font, size, stroke=0, stroke_color=BLACK, shadow=0.0, plate=None, glow=None):
    """Render lines of (token, rgb) pairs as one RGBA image. A token is a word, or ("emoji", ch)."""
    space = font.getlength(" ")
    asc, desc = font.getmetrics()
    lh = asc + desc
    widths = [[size * 1.1 if isinstance(t, tuple) else font.getlength(t) for t, _ in line] for line in lines]
    line_w = [sum(ws) + space * (len(ws) - 1) for ws in widths]
    pad_x, pad_y = (int(size * 0.35), int(size * 0.18)) if plate else (stroke + int(size * 0.2), stroke + int(size * 0.2))
    line_step = lh + (2 * pad_y if plate else int(size * 0.1))
    w = int(max(line_w) + 2 * pad_x + 2 * size * 0.2)
    h = int(line_step * len(lines) + (0 if plate else 2 * pad_y) + size * 0.3)
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    if plate:
        m = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(m)
        for i, lw in enumerate(line_w):
            x0 = (w - lw) / 2 - pad_x
            y0 = i * line_step + size * 0.1
            d.rounded_rectangle([x0, y0, x0 + lw + 2 * pad_x, y0 + lh + 2 * pad_y + 1], radius=int(size * 0.28), fill=255)
        color, alpha = plate
        canvas = Image.alpha_composite(canvas, colored(m.point(lambda v: v * alpha // 255), color))

    text_masks = {}           # rgb -> mask, so each colour is composited once
    stroke_mask = Image.new("L", (w, h), 0)
    emojis = []
    for i, (line, ws) in enumerate(zip(lines, widths)):
        x = (w - line_w[i]) / 2
        y = i * line_step + (size * 0.1 + pad_y if plate else pad_y)
        for (tok, rgb), tw in zip(line, ws):
            if isinstance(tok, tuple):
                emojis.append((tok[1], int(x), int(y + (lh - size * 1.1) / 2)))
            else:
                m = text_masks.setdefault(rgb, Image.new("L", (w, h), 0))
                ImageDraw.Draw(m).text((x, y), tok, font=font, fill=255)
                if stroke:
                    ImageDraw.Draw(stroke_mask).text((x, y), tok, font=font, fill=255, stroke_width=stroke, stroke_fill=255)
            x += tw + space
    all_text = Image.new("L", (w, h), 0)
    for m in text_masks.values():
        all_text = Image.fromarray(np.maximum(np.asarray(all_text), np.asarray(m)))
    if glow:
        g = (stroke_mask if stroke else all_text).filter(ImageFilter.GaussianBlur(size * 0.18))
        canvas = Image.alpha_composite(canvas, colored(g.point(lambda v: min(255, v * 3)), glow))
    if shadow:
        s = (stroke_mask if stroke else all_text).filter(ImageFilter.GaussianBlur(max(size * 0.06, 1)))
        s = s.point(lambda v: int(v * shadow))
        off = Image.new("L", (w, h), 0)
        off.paste(s, (max(int(size * 0.04), 1), max(int(size * 0.05), 1)))
        canvas = Image.alpha_composite(canvas, colored(off, BLACK))
    if stroke:
        canvas = Image.alpha_composite(canvas, colored(stroke_mask, stroke_color))
    for rgb, m in text_masks.items():
        canvas = Image.alpha_composite(canvas, colored(m, rgb))
    for ch, x, y in emojis:
        e = emoji_image(ch, int(size * 1.1))
        canvas.alpha_composite(e, (max(x, 0), max(y, 0)))
    return canvas


def wrap(tokens, font, size, max_w):
    space = font.getlength(" ")
    lines, cur, cur_w = [], [], 0.0
    for tok in tokens:
        tw = size * 1.1 if isinstance(tok[0], tuple) else font.getlength(tok[0])
        if cur and cur_w + space + tw > max_w:
            lines.append(cur)
            cur, cur_w = [], 0.0
        cur.append(tok)
        cur_w += (space if len(cur) > 1 else 0) + tw
    if cur:
        lines.append(cur)
    return lines


def pick_font(rng, family, size):
    path, weight = rng.choice(fonts.FONTS[family])
    return fonts.load(path, weight, size), f"{path.split('/')[-1]}@{weight}"


def loguniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


# ---------------------------------------------------------------- timeline items
# Each item is a list of segments (f0, f1, image, cx, cy, anim); frames f0 <= i < f1 show the image centred at (cx, cy).
def caption_y(rng):
    r = rng.random()
    y = rng.gauss(0.70, 0.07) if r < 0.55 else rng.gauss(0.50, 0.09) if r < 0.85 else rng.gauss(0.25, 0.07)
    return min(max(y, 0.12), 0.88)


def caption_track(rng, n, fps, W, H, y=None):
    style = rng.choices(["plain", "plate", "bold_pop", "karaoke"], weights=[0.3, 0.25, 0.3, 0.15])[0]
    family = {"plain": rng.choice(["sans", "sans", "sans_bold"]), "plate": rng.choice(["sans", "sans_bold"]),
              "bold_pop": "heavy", "karaoke": "sans_bold"}[style]
    box = loguniform(rng, 0.028, 0.07) if style == "bold_pop" else loguniform(rng, 0.018, 0.045)
    size = max(int(box * H), 12)
    font, font_name = pick_font(rng, family, size)
    per_chunk = {"plain": (3, 7), "plate": (2, 6), "bold_pop": (1, 3), "karaoke": (4, 8)}[style]
    upper = rng.random() < (0.8 if style == "bold_pop" else 0.08)
    rate = rng.uniform(2.2, 3.6)                                   # spoken words per second
    y = (caption_y(rng) if y is None else y) * H
    max_w = W * rng.uniform(0.72, 0.9)
    base = WHITE if rng.random() < 0.85 or style != "bold_pop" else (255, 230, 0)
    hl = rng.choice(HIGHLIGHTS)
    highlight = style == "bold_pop" and rng.random() < 0.7
    plate_bg, plate_fg = rng.choice(PLATES) if style == "plate" else (WHITE, BLACK)
    plate_alpha = 255 if plate_bg != BLACK else rng.choice([150, 200, 255])
    stroke = int(size * rng.uniform(0.08, 0.14)) if style == "bold_pop" else (int(size * 0.05) if style == "plain" and rng.random() < 0.4 else 0)
    shadow = rng.uniform(0.5, 0.9) if style == "plain" else 0.0
    anim = "pop" if style == "bold_pop" and rng.random() < 0.7 else ("fade" if rng.random() < 0.3 else None)

    start = int(rng.uniform(0, 0.6) * fps)
    end = n if rng.random() < 0.8 else int(n * rng.uniform(0.4, 1.0))
    words = word_stream(rng, int((end - start) / fps * rate) + 10)
    segs, f, wi = [], start, 0
    while f < end and wi < len(words):
        k = rng.randint(*per_chunk)
        chunk = [w.upper() if upper else w for w in words[wi:wi + k]]
        wi += k
        dur = max(int(len(chunk) / rate * fps), 2)
        emoji = rng.choice(EMOJIS) if rng.random() < 0.12 else None
        # sub-states: per word for highlight / karaoke, otherwise one state per chunk
        states = range(len(chunk)) if (highlight or style == "karaoke") else [None]
        sub = max(dur // len(states), 1)
        for si, cur in enumerate(states):
            if style == "karaoke":
                toks = [(w, plate_fg if j <= cur else GREY) for j, w in enumerate(chunk)]
            elif highlight:
                toks = [(w, hl if j == cur else base) for j, w in enumerate(chunk)]
            else:
                toks = [(w, plate_fg if style == "plate" else base) for w in chunk]
            if emoji:
                toks.append((("emoji", emoji), base))
            img = draw_block(wrap(toks, font, size, max_w), font, size, stroke=stroke, shadow=shadow,
                             plate=(plate_bg, plate_alpha) if style in ("plate", "karaoke") else None)
            f0 = f + si * sub
            f1 = min(f + (si + 1) * sub if si < len(states) - 1 else f + dur, end)
            if f0 < f1:
                segs.append((f0, f1, img, W / 2, y, anim if si == 0 else None))
        f += dur
        if rng.random() < 0.15:
            f += int(rng.uniform(0.1, 0.6) * fps)
    return {"type": "caption", "style": style, "font": font_name, "size_px": size, "segments": segs}


def hook_title(rng, n, fps, W, H):
    family = rng.choices(["sans_bold", "heavy", "serif", "script", "mono"], weights=[0.3, 0.3, 0.2, 0.12, 0.08])[0]
    size = max(int(loguniform(rng, 0.03, 0.065) * H), 14)
    font, font_name = pick_font(rng, family, size)
    style = rng.choices(["plate", "stroke", "shadow", "glow"], weights=[0.35, 0.3, 0.25, 0.1])[0]
    words = rng.choice(ad_copy()).split()[:rng.randint(2, 8)]
    if family == "heavy" or rng.random() < 0.2:
        words = [w.upper() for w in words]
    fg = rng.choice([WHITE, WHITE, (255, 230, 0)]) if style != "plate" else None
    bg, pfg = rng.choice(PLATES)
    toks = [(w, pfg if style == "plate" else fg) for w in words]
    img = draw_block(wrap(toks, font, size, W * rng.uniform(0.7, 0.88)), font, size,
                     stroke=int(size * 0.09) if style == "stroke" else 0, shadow=0.8 if style == "shadow" else 0.0,
                     plate=(bg, 255) if style == "plate" else None, glow=rng.choice(HIGHLIGHTS) if style == "glow" else None)
    f1 = n if rng.random() < 0.5 else min(int(rng.uniform(2, 4) * fps), n)
    return {"type": "hook", "style": style, "font": font_name, "size_px": size,
            "segments": [(0, f1, img, W / 2, rng.uniform(0.1, 0.3) * H, "pop" if rng.random() < 0.3 else None)]}


def band(item, H):
    """Vertical extent (y0, y1) of an item over all its segments, in units of the frame height."""
    ys = [(cy - img.height / 2, cy + img.height / 2) for _, _, img, _, cy, _ in item["segments"]]
    return min(a for a, _ in ys) / H, max(b for _, b in ys) / H


def sticker(rng, n, fps, W, H, avoid=()):
    if rng.random() < 0.25:
        size = int(loguniform(rng, 0.045, 0.09) * H)
        img = emoji_image(rng.choice(EMOJIS), size)
        kind, font_name = "emoji", "NotoColorEmoji"
    else:
        size = max(int(loguniform(rng, 0.022, 0.045) * H), 12)
        font, font_name = pick_font(rng, rng.choice(["sans_bold", "heavy"]), size)
        bg, fg = rng.choice(PLATES[3:] + [(rng.choice(HIGHLIGHTS), WHITE)])
        text = rng.choice(STICKERS)
        toks = [(text, fg)] + ([(("emoji", "⭐"), fg)] if "stars" in text else [])     # symbols via the emoji font
        img = draw_block([toks], font, size, plate=(bg, 255))
        kind = "text"
    if rng.random() < 0.3:
        img = img.rotate(rng.uniform(-10, 10), resample=Image.BICUBIC, expand=True)
    if rng.random() < 0.5:
        f0, f1 = 0, n
    else:
        f0 = rng.randint(0, max(n - int(1.5 * fps), 0))
        f1 = min(f0 + int(rng.uniform(1.5, 5) * fps), n)
    # stickers sit towards the corners (faces and products are usually central) and off other text
    hh = img.height / H / 2
    for _ in range(30):
        cx = (rng.uniform(0.12, 0.35) if rng.random() < 0.5 else rng.uniform(0.65, 0.88)) * W
        cy = rng.uniform(0.08, 0.3) if rng.random() < 0.5 else rng.uniform(0.6, 0.9)
        if all(cy + hh < a or cy - hh > b for a, b in avoid):
            break
    cy *= H
    return {"type": "sticker", "style": kind, "font": font_name, "size_px": size,
            "segments": [(f0, f1, img, cx, cy, "pop" if rng.random() < 0.5 else None)]}


def fine_print(rng, n, fps, W, H):
    size = max(int(loguniform(rng, 0.011, 0.016) * H), 10)
    font, font_name = pick_font(rng, "sans", size)
    toks = [(w, (235, 235, 235)) for w in rng.choice(FINE_PRINT).split()]
    img = draw_block(wrap(toks, font, size, W * 0.85), font, size, shadow=0.7)
    return {"type": "fine_print", "style": "plain", "font": font_name, "size_px": size,
            "segments": [(0, n, img, W / 2, rng.uniform(0.9, 0.95) * H, None)]}


# ---------------------------------------------------------------- compositing
def place(w, h, cx, cy, W, H):
    """Top-left corner of a w x h block centred at (cx, cy), kept inside the W x H frame."""
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    return min(max(x0, 0), max(W - w, 0)), min(max(y0, 0), max(H - h, 0))


def paste(frame, img, cx, cy, alpha_mul=1.0):
    a = np.asarray(img)
    h, w = a.shape[:2]
    H, W = frame.shape[:2]
    x0, y0 = place(w, h, cx, cy, W, H)
    x1, y1 = min(x0 + w, W), min(y0 + h, H)
    a = a[:y1 - y0, :x1 - x0]
    alpha = a[..., 3:4].astype(np.float32) / 255 * alpha_mul
    region = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (region * (1 - alpha) + a[..., :3] * alpha + 0.5).astype(np.uint8)
    return [x0, y0, x1, y1]


def animated(img, anim, k):
    """k = frames since the segment started. Pop: 118% -> 100% over 3 frames. Fade: 3-frame fade-in."""
    if anim == "pop" and k < 3:
        s = [1.18, 1.08, 1.0][k]
        return img.resize((max(int(img.width * s), 1), max(int(img.height * s), 1)), Image.BILINEAR), 1.0
    if anim == "fade" and k < 3:
        return img, (k + 1) / 4
    return img, 1.0


def plan_text(n, fps, W, H, rng, config):
    """Decide and pre-render every text item for an n-frame W x H clip. Returns (items, meta for meta.json)."""
    items = []
    if rng.random() < config["text_prob"]:
        cap = rng.random() < config["caption_prob"]
        if rng.random() < config["hook_prob"] or not cap:
            items.append(hook_title(rng, n, fps, W, H))
        if cap:                                   # captions stay out of the title's band
            bands = [band(it, H) for it in items]
            y = caption_y(rng)
            for _ in range(30):
                if all(y + 0.06 < a or y - 0.06 > b for a, b in bands):
                    break
                y = caption_y(rng)
            items.append(caption_track(rng, n, fps, W, H, y=y))
        for _ in range(rng.choices([0, 1, 2], weights=[1 - config["sticker_prob"], config["sticker_prob"] * 0.7,
                                                         config["sticker_prob"] * 0.3])[0]):
            items.append(sticker(rng, n, fps, W, H, avoid=[band(it, H) for it in items]))
        if rng.random() < config["fine_print_prob"]:
            items.append(fine_print(rng, n, fps, W, H))
    meta = []
    for it in items:
        boxes = []
        for f0, f1, img, cx, cy, _ in it["segments"]:
            x0, y0 = place(img.width, img.height, cx, cy, W, H)
            boxes.append([f0, f1, x0, y0, min(x0 + img.width, W), min(y0 + img.height, H)])
        meta.append({k: v for k, v in it.items() if k != "segments"} | {"boxes": boxes})
    return items, meta


def draw_text(frame, i, items):
    """Burn every item active at frame index i into frame (H x W x 3 uint8, modified in place)."""
    for it in items:
        for f0, f1, img, cx, cy, anim in it["segments"]:
            if f0 <= i < f1:
                im, a = animated(img, anim, i - f0)
                paste(frame, im, cx, cy, a)


def burn_text(frames, fps, rng, config):
    """Array version of plan_text + draw_text. Returns (new frames, meta)."""
    n, H, W = frames.shape[:3]
    items, meta = plan_text(n, fps, W, H, rng, config)
    out = frames.copy()
    for i in range(n):
        draw_text(out[i], i, items)
    return out, meta
