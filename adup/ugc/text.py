"""Burn realistic on-screen text and graphics into GT frames: captions, headlines, TikTok-style native text, prices and
promo codes, brand logos, CTA buttons, stickers, fine print and annotation shapes.

Part of UGC-ification (config section ugc.text). Text is drawn on the GT, before any degradation, because in real ads
it is burned in by the editing app or at upload and then goes through every later encode, so GT and LQ both carry it.

Which items an ad gets depends on its style (ugc.director.styles). The per-style probabilities (ugc.text.by_style) come
from a manual annotation of 48 real ads (data/stats/real_ads/ad_anatomy.csv, docs/ugc_dataset.md §1.3): creator (UGC)
ads mostly carry running captions, brand ads a headline and a logo. Geometry follows OCR on 106 portrait ads
(docs/ugc_degradations.md §2): line heights 1.6–6% of the frame height, text over the whole height with a bias to the
lower middle. Styles are the families seen in those ads (and in CapCut / TikTok / caption apps). Captions and headlines
use the scraped ads' own copy, from ads of the same theme when there are enough.

Everything is drawn from masks and composited with straight alpha, so anti-aliased edges carry no dark fringes. Shapes
(pills, icons, arrows) are drawn at 4x and downsampled. Items do not overlap: each takes the first free position drawn
from its own placement distribution. Sizes are fractions of the frame's long side (the OCR statistics are on portrait ads),
so landscape ads get type of the same pixel size as portrait ones.
"""

import glob
import os
import re
from functools import lru_cache
from itertools import pairwise

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

from adup.paths import REAL_ADS
from adup.ugc import fonts

FALLBACK_COPY = [
    "I was today years old when I found this", "ok but why is nobody talking about this", "this changed my whole routine",
    "honestly obsessed with how this turned out", "three reasons you need this in your life", "run don't walk to get this",
    "POV you finally found the one that works", "day seven update and I am shocked", "wait for the before and after",
    "this is your sign to try it", "I tested it so you don't have to", "my new favorite thing this month",
    "the texture is actually insane", "no filter no edits just this", "everyone keeps asking me what I use",
]
HOOKS = ["wait for it", "POV: you finally found it", "3 reasons you need this", "I was today years old", "don't skip this",
         "this changed everything", "honest review", "run don't walk", "day 1 vs day 30", "the viral one", "before vs after",
         "is it worth it?", "ok but why is no one talking about this", "watch till the end", "my holy grail"]
CTAS = ["Shop now", "Link in bio", "Tap to shop", "Get yours today", "Download now", "Order now", "Try it risk-free",
        "Learn more", "Sign up today", "Get offer", "Install now", "Book now", "Try now", "Shop the sale"]
EMOJIS = list("😍🔥✨💯😭🙌👀💕😂🤯✅⭐🛒💪🥰😱👇🎉💖🤩")
STICKERS = ["50% OFF", "SALE", "FREE SHIPPING", "Shop Now >", "Link in bio", "#ad", "Paid partnership", "NEW", "BEST SELLER",
            "Limited time only", "RETAIL: $27", "$29.99", "BOGO", "Only $19", "4.8 stars", "Use code SAVE20",
            "Buy 1 Get 1 FREE", "TikTok Made Me Buy It", "As seen on TV", "Get yours now", "Sign up today"]
FINE_PRINT = ["Results may vary. Individual results not guaranteed.", "Paid partnership. #ad",
              "These statements have not been evaluated by the Food and Drug Administration.",
              "Terms apply. See site for details.", "Offer valid while supplies last.", "Dramatization. Not actual results.",
              "Actor portrayal based on real user reviews.", "Some parts of this video were generated with AI.",
              "Members only. Savings vary.", "*Based on a consumer study of 120 participants."]
PRICES = ["$9", "$12", "$15", "$19", "$21", "$24", "$29", "$36", "$44", "$49", "$59", "$79", "$99", "$19.99", "$29.99",
          "$4.95", "50% OFF", "30% OFF", "20% OFF", "70% OFF", "-40%", "2 for $30", "BOGO", "Only $19", "Under $25",
          "FREE SHIPPING", "SALE"]
CODE_WORDS = ["SAVE", "GLOW", "FRESH", "HELLO", "WELCOME", "SHOP", "NEW", "VIP", "SUMMER", "TRY", "LOVE", "FIRST"]
SYLLABLES = ["lu", "ma", "no", "va", "ri", "ka", "zo", "be", "la", "vi", "sol", "mi", "ra", "ne", "to", "ly", "o", "a",
             "fi", "den", "kin", "pur", "glo", "ver", "tal", "mo", "sen", "cor", "bel", "nu", "ze", "hal", "qui", "ro"]

WHITE, BLACK, GREY = (255, 255, 255), (0, 0, 0), (150, 150, 150)
HIGHLIGHTS = [(247, 194, 4), (255, 230, 0), (2, 251, 35), (46, 204, 113), (0, 200, 255), (254, 44, 85), (255, 140, 0),
              (255, 64, 160), (160, 90, 255)]
PLATES = [(WHITE, BLACK), (WHITE, BLACK), (WHITE, BLACK), (BLACK, WHITE), ((254, 44, 85), WHITE), ((46, 170, 90), WHITE),
          ((255, 221, 0), BLACK), ((30, 30, 30), WHITE), ((230, 230, 255), (40, 40, 120))]
# brand palettes (background, accent, text on the background), as on real brand cards and end cards
PALETTES = [((225, 20, 40), WHITE, WHITE), ((40, 200, 210), WHITE, WHITE), ((245, 240, 228), (200, 30, 50), (200, 30, 50)),
            (WHITE, (20, 20, 20), (20, 20, 20)), ((30, 60, 230), WHITE, WHITE), ((245, 170, 200), WHITE, (120, 20, 70)),
            ((18, 18, 18), (220, 190, 120), WHITE), ((20, 110, 70), (255, 230, 120), WHITE), ((255, 215, 0), (20, 20, 20), (20, 20, 20)),
            ((15, 30, 70), (80, 200, 255), WHITE), ((235, 225, 210), (90, 60, 40), (60, 40, 30)), ((250, 250, 250), (46, 170, 90), (30, 30, 30))]

# real-ad folders (Meta search query / TikTok industry) -> ad themes of data/stats/hq/content_coverage.csv
FOLDER_THEME = {
    "app": "tech_app", "apps": "tech_app", "games": "tech_app", "mobile_game": "tech_app", "headphones": "tech_app",
    "phone_case": "tech_app", "tech_electronics": "tech_app", "makeup": "beauty", "skincare": "beauty", "haircare": "beauty",
    "perfume": "beauty", "teeth_whitening": "beauty", "beauty_personal_care": "beauty", "dress": "fashion",
    "sneakers": "fashion", "jewelry": "fashion", "apparel_accessories": "fashion", "snack": "food", "meal_kit": "food",
    "food_beverage": "food", "cleaning": "home", "home_decor": "home", "kitchen_gadget": "home", "mattress": "home",
    "vacuum": "home", "household_products": "home", "home_improvement": "home", "appliances": "home", "dog_food": "pets",
    "pets": "pets", "supplement": "health", "protein_powder": "health", "health": "health", "baby": "baby_kids",
    "baby_kids_maternity": "baby_kids", "fitness": "fitness_sports", "sports_outdoor": "fitness_sports", "car": "car",
    "vehicle_transportation": "car", "travel": "travel", "education": "education", "language_learning": "education",
}


# ---------------------------------------------------------------- copy
def clean_sentences(text):
    t = re.sub(r"https?://\S+|www\.\S+|@\w+|#\w+|\{\{.*?\}\}", " ", text)
    t = t.translate(str.maketrans("‘’“”–—…", "''\"\"--."))
    t = re.sub(r"[^\x00-\x7F]+", " ", t)              # emoji are drawn separately; fonts are Latin-only
    return [" ".join(s.split()) for s in re.split(r"(?<=[.!?])\s+|\n+", t)]


@lru_cache(maxsize=1)
def copy_by_theme():
    """{theme: sentences} from the scraped ads' own copy (TikTok titles, Meta bodies); theme None holds all of them."""
    out = {None: []}
    for f in glob.glob(str(REAL_ADS / "*" / "*" / "summary.csv")):
        df = pd.read_csv(f)
        folder = df["video_file"].astype(str).map(lambda p: os.path.basename(os.path.dirname(p))) if "video_file" in df else None
        for col in ("ad_title", "body"):
            if col not in df:
                continue
            for i, t in df[col].dropna().astype(str).items():
                theme = FOLDER_THEME.get(folder[i]) if folder is not None else None
                for s in clean_sentences(t):
                    if 3 <= len(s.split()) <= 30:
                        out[None].append(s)
                        out.setdefault(theme, []).append(s)
    return out


def ad_copy(theme=None):
    """Sentences of ads of this theme when there are at least 30, else of all ads, plus generic lines."""
    c = copy_by_theme()
    own = c.get(theme, [])
    return (own if len(own) >= 30 else c[None]) + FALLBACK_COPY


STOP_END = {"a", "an", "the", "and", "or", "of", "to", "for", "with", "your", "my", "our", "in", "on", "at", "is", "are",
            "you", "i", "it", "this", "that", "from", "by", "as", "be", "so", "but", "if", "we", "they", "was", "just"}


def short_line(rng, theme=None, lo=2, hi=7):
    """A headline: a short sentence of the theme's copy, else the start of a longer one cut after a content word."""
    pool = ad_copy(theme)
    short = [s for s in pool if lo <= len(s.split()) <= hi]
    words = (rng.choice(short) if short and rng.random() < 0.8 else rng.choice(pool)).split()[:hi]
    while len(words) > lo and words[-1].lower().strip(",;:-.") in STOP_END:
        words = words[:-1]
    words[-1] = words[-1].rstrip(",;:-.")
    return " ".join(words)


def word_stream(rng, n_words, theme=None):
    corpus = ad_copy(theme)
    words = []
    while len(words) < n_words:
        words += rng.choice(corpus).split()
    return words[:n_words]


def brand_name(rng):
    """A made-up brand name (two or three syllables), so logos and end cards never show a real brand."""
    name = "".join(rng.choice(SYLLABLES) for _ in range(rng.choice([2, 2, 3])))
    return name.capitalize() if rng.random() < 0.7 else name.upper() if rng.random() < 0.5 else name.lower()


def promo_code(rng, brand=None):
    word = brand.upper()[:8] if brand and rng.random() < 0.4 else rng.choice(CODE_WORDS)
    return f"{word}{rng.choice([10, 15, 20, 25, 30, 40, 50])}"


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


def luma(rgb):
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


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


def fitted_font(rng, family, size, texts, max_w, max_lines=3, min_size=12):
    """A font of the family at `size`, made smaller until every text wraps into at most max_lines lines at max_w."""
    path, weight = rng.choice(fonts.FONTS[family])
    while True:
        font = fonts.load(path, weight, size)
        if size <= min_size or all(len(wrap([(w, None) for w in t.split()], font, size, max_w)) <= max_lines for t in texts):
            return font, f"{path.split('/')[-1]}@{weight}", size
        size = max(int(size * 0.85), min_size)


def loguniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def supersampled(w, h, draw, k=4):
    """An anti-aliased L mask of w x h: `draw(ImageDraw, k)` paints it at k times the size."""
    m = Image.new("L", (w * k, h * k), 0)
    draw(ImageDraw.Draw(m), k)
    return m.resize((w, h), Image.LANCZOS)


def pill_image(rng, text, size, fill, fg, family=None, arrow=False):
    """A CTA button: text on a fully rounded (pill) or rounded-rectangle plate."""
    font, name = pick_font(rng, family or rng.choice(["sans_bold", "sans_bold", "heavy"]), size)
    label = draw_block([[(text + (" >" if arrow else ""), fg)]], font, size)
    lw, lh = label.size
    w, h = int(lw + size * 1.2), int(size * 2.1)
    radius = h / 2 if rng.random() < 0.7 else size * 0.35
    m = supersampled(w, h, lambda d, k: d.rounded_rectangle([0, 0, w * k - 1, h * k - 1], radius=radius * k, fill=255))
    img = colored(m, fill)
    img.alpha_composite(label, ((w - lw) // 2, (h - lh) // 2))
    return img, name


def logo_image(rng, brand, size, color, family=None):
    """A made-up wordmark, with an icon (disc, rounded square or ring holding the initial) half of the time."""
    family = family or rng.choice(["serif", "sans_bold", "heavy", "script", "sans"])
    font, name = pick_font(rng, family, size)
    text = brand.upper() if family in ("sans_bold", "heavy", "sans") and rng.random() < 0.5 else brand
    word = draw_block([[(text, color)]], font, size)
    if rng.random() >= 0.5:
        return word, text, name
    s = int(size * 1.35)
    shape = rng.choice(["disc", "square", "ring"])
    m = supersampled(s, s, lambda d, k: (d.ellipse([0, 0, s * k - 1, s * k - 1], fill=255) if shape == "disc" else
                                         d.rounded_rectangle([0, 0, s * k - 1, s * k - 1], radius=s * k * 0.25, fill=255)
                                         if shape == "square" else
                                         d.ellipse([0, 0, s * k - 1, s * k - 1], outline=255, width=max(int(s * k * 0.1), 1))))
    icon = colored(m, color)
    ifont, _ = pick_font(rng, "sans_bold", int(size * 0.8))
    initial = draw_block([[(brand[0].upper(), color if shape == "ring" else (BLACK if luma(color) > 140 else WHITE))]],
                         ifont, int(size * 0.8))
    icon.alpha_composite(initial, (max((s - initial.width) // 2, 0), max((s - initial.height) // 2, 0)))
    out = Image.new("RGBA", (s + int(size * 0.3) + word.width, max(s, word.height)), (0, 0, 0, 0))
    out.alpha_composite(icon, (0, (out.height - s) // 2))
    out.alpha_composite(word, (s + int(size * 0.3), (out.height - word.height) // 2))
    return out, text, name


def arrow_image(rng, size, color):
    """A thick arrow (with a dark outline) or a ring, drawn at a random angle."""
    kind = rng.choice(["arrow", "arrow", "ring"])
    if kind == "ring":
        s = int(size * 2.2)
        w = max(int(size * 0.18), 3)
        m = supersampled(s, s, lambda d, k: d.ellipse([w * k, w * k, (s - w) * k, (s - w) * k], outline=255, width=w * k))
        return colored(m, color), "ring"
    L, T = int(size * 2.4), int(size * 0.34)

    def body(d, k, grow=0):
        g = grow * k
        d.rectangle([0 - g + T * k, L * k // 2 - T * k // 2 - g, L * k * 0.62 + g, L * k // 2 + T * k // 2 + g], fill=255)
        d.polygon([(L * k * 0.55 - g, L * k * 0.5 - T * k * 1.6 - g), (L * k - k - g * 0.2, L * k * 0.5),
                   (L * k * 0.55 - g, L * k * 0.5 + T * k * 1.6 + g)], fill=255)
    fill = supersampled(L, L, body)
    edge = supersampled(L, L, lambda d, k: body(d, k, grow=max(int(size * 0.06), 1)))
    img = Image.alpha_composite(colored(edge, (20, 20, 20)), colored(fill, color))
    return img.rotate(rng.uniform(0, 360), resample=Image.BICUBIC, expand=True), "arrow"


# ---------------------------------------------------------------- placement
def place(w, h, cx, cy, W, H):
    """Top-left corner of a w x h block centred at (cx, cy), kept inside the W x H frame."""
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    return min(max(x0, 0), max(W - w, 0)), min(max(y0, 0), max(H - h, 0))


def free_spot(rng, w, h, W, H, spots, taken, tries=30):
    """Centre (cx, cy) in pixels for a w x h block: the first draw of spots() (relative centre) whose box does not
    overlap a box already taken; the last draw if none is free. The chosen box is added to `taken`."""
    m = H * 0.008
    for _ in range(tries):
        rx, ry = spots()
        cx, cy = rx * W, ry * H
        x0, y0 = place(w, h, cx, cy, W, H)
        box = (x0, y0, x0 + w, y0 + h)
        if all(box[2] + m < t[0] or box[0] - m > t[2] or box[3] + m < t[1] or box[1] - m > t[3] for t in taken):
            break
    taken.append(box)
    return cx, cy


def item(kind, style, font, size, segments, texts, alpha=1.0):
    """A timeline item: segments (f0, f1, image, cx, cy, anim) show the image centred at (cx, cy) on frames f0 <= i < f1;
    texts line up with segments ("" for shapes and emoji)."""
    return {"type": kind, "style": style, "font": font, "size_px": int(size), "segments": segments, "texts": texts,
            "alpha": alpha}


# ---------------------------------------------------------------- timeline items
def caption_y(rng):
    r = rng.random()
    y = rng.gauss(0.70, 0.07) if r < 0.55 else rng.gauss(0.50, 0.09) if r < 0.85 else rng.gauss(0.25, 0.07)
    return min(max(y, 0.12), 0.88)


def caption_track(rng, n, fps, W, H, taken, cfg, theme=None):
    """Running captions of the voice-over, chunk by chunk (auto-caption styles of CapCut / TikTok / caption apps)."""
    styles = cfg["caption_styles"]
    style = rng.choices(list(styles), weights=list(styles.values()))[0]
    family = {"plain": rng.choice(["sans", "sans", "sans_bold"]), "plate": rng.choice(["sans", "sans_bold"]),
              "bold_pop": "heavy", "karaoke": "sans_bold"}[style]
    box = loguniform(rng, 0.028, 0.07) if style == "bold_pop" else loguniform(rng, 0.016, 0.04)
    size = max(int(box * max(W, H)), 12)
    font, font_name = pick_font(rng, family, size)
    per_chunk = {"plain": (3, 8), "plate": (2, 6), "bold_pop": (1, 3), "karaoke": (4, 8)}[style]
    upper = rng.random() < (0.8 if style == "bold_pop" else 0.08)
    rate = rng.uniform(2.2, 3.6)                                   # spoken words per second
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
    words = word_stream(rng, int((end - start) / fps * rate) + 10, theme)
    segs, texts, f, wi = [], [], start, 0
    while f < end and wi < len(words):
        k = rng.randint(*per_chunk)
        chunk = [w.upper() if upper else w for w in words[wi:wi + k]]
        wi += k
        dur = max(int(len(chunk) / rate * fps), 2)
        emoji = rng.choice(EMOJIS) if rng.random() < cfg["emoji_prob"] else None
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
                segs.append([f0, f1, img, anim if si == 0 else None])
                texts.append(" ".join(chunk))
        f += dur
        if rng.random() < 0.15:
            f += int(rng.uniform(0.1, 0.6) * fps)
    if not segs:
        return None
    w, h = max(s[2].width for s in segs), max(s[2].height for s in segs)
    cx, cy = free_spot(rng, w, h, W, H, lambda: (0.5, caption_y(rng)), taken)
    return item("caption", style, font_name, size, [(f0, f1, img, cx, cy, a) for f0, f1, img, a in segs], texts)


def headline_y(rng):
    r = rng.random()
    return rng.uniform(0.07, 0.3) if r < 0.55 else rng.uniform(0.3, 0.55) if r < 0.85 else rng.uniform(0.55, 0.78)


def headline(rng, n, fps, W, H, taken, theme=None, cuts=(), family=None):
    """A title block: a hook for the first seconds, one line for the whole ad, or a new line on every shot (brand ads)."""
    family = family or rng.choices(["sans_bold", "heavy", "serif", "script", "mono"], weights=[0.3, 0.3, 0.25, 0.1, 0.05])[0]
    style = rng.choices(["plate", "stroke", "shadow", "glow", "plain"], weights=[0.3, 0.25, 0.25, 0.05, 0.15])[0]
    upper = family == "heavy" or rng.random() < 0.2
    fg = rng.choice([WHITE, WHITE, WHITE, (255, 230, 0), BLACK if style == "plain" else WHITE])
    bg, pfg = rng.choice(PLATES)
    max_w = W * rng.uniform(0.66, 0.88)
    per_shot = len(cuts) > 0 and rng.random() < 0.35
    bounds = [0, *cuts, n] if per_shot else [0, n if rng.random() < 0.5 else min(int(rng.uniform(2, 4) * fps), n)]
    spans = [(f0, f1) for f0, f1 in pairwise(bounds) if f1 - f0 >= 8]
    lines = [short_line(rng, theme) for _ in spans]
    lines = [t.upper() for t in lines] if upper else lines
    # big type for short lines only: at most three lines, as in real ads
    font, font_name, size = fitted_font(rng, family, max(int(loguniform(rng, 0.028, 0.065) * max(W, H)), 14), lines, max_w)
    anim = rng.choice([None, "pop", "fade", "slide"])
    segs, texts = [], []
    for (f0, f1), line in zip(spans, lines):
        words = line.split()
        toks = [(w, pfg if style == "plate" else fg) for w in words]
        img = draw_block(wrap(toks, font, size, max_w), font, size,
                         stroke=int(size * 0.09) if style == "stroke" else 0, shadow=0.8 if style == "shadow" else 0.0,
                         plate=(bg, 255) if style == "plate" else None, glow=rng.choice(HIGHLIGHTS) if style == "glow" else None)
        segs.append([f0, f1, img, anim])
        texts.append(" ".join(words))
    if not segs:
        return None
    w, h = max(s[2].width for s in segs), max(s[2].height for s in segs)
    cx, cy = free_spot(rng, w, h, W, H, lambda: (0.5, headline_y(rng)), taken)
    return item("headline", style, font_name, size, [(f0, f1, img, cx, cy, a) for f0, f1, img, a in segs], texts)


def native_text(rng, n, fps, W, H, taken, theme=None):
    """TikTok's in-app text: small TikTok Sans, plain white with a soft shadow or on a white / black plate, usually
    one static line or two above the middle for the whole video."""
    size = max(int(loguniform(rng, 0.011, 0.022) * max(W, H)), 10)
    font, font_name = pick_font(rng, "sans", size)
    words = short_line(rng, theme, 3, 12).split()
    plate = rng.random() < 0.3
    bg, fg = rng.choice([(WHITE, BLACK), (BLACK, WHITE)]) if plate else (None, WHITE)
    img = draw_block(wrap([(w, fg) for w in words], font, size, W * rng.uniform(0.5, 0.75)), font, size,
                     shadow=0.0 if plate else rng.uniform(0.3, 0.7), plate=(bg, 255) if plate else None)
    f0 = 0 if rng.random() < 0.7 else int(rng.uniform(0.3, 1.5) * fps)
    cx, cy = free_spot(rng, img.width, img.height, W, H, lambda: (rng.uniform(0.4, 0.6), rng.uniform(0.1, 0.55)), taken)
    return item("native", "plate" if plate else "plain", font_name, size, [(f0, n, img, cx, cy, None)], [" ".join(words)])


def promo(rng, n, fps, W, H, taken, cuts=(), brand=None):
    """A price or discount in big stroked type next to the product (on one or more shots), or a promo code on a plate."""
    if rng.random() < 0.6:
        size = max(int(loguniform(rng, 0.035, 0.075) * max(W, H)), 14)
        font, font_name = pick_font(rng, "heavy", size)
        fill = rng.choice([(2, 251, 35), (255, 230, 0), WHITE, (254, 44, 85)])
        starts = sorted(rng.sample([0, *cuts], min(len(cuts) + 1, rng.randint(1, 3))))
        segs, texts = [], []
        for f0 in starts:
            text = rng.choice(PRICES)
            img = draw_block([[(text, fill)]], font, size, stroke=int(size * rng.uniform(0.08, 0.13)))
            if rng.random() < 0.3:
                img = img.rotate(rng.uniform(-12, 12), resample=Image.BICUBIC, expand=True)
            segs.append([f0, min(f0 + int(rng.uniform(1.5, 4) * fps), n), img, rng.choice([None, "pop"])])
            texts.append(text)
        w, h = max(s[2].width for s in segs), max(s[2].height for s in segs)
        cx, cy = free_spot(rng, w, h, W, H, lambda: (rng.choice([rng.uniform(0.2, 0.35), rng.uniform(0.65, 0.8)]),
                                                     rng.uniform(0.3, 0.65)), taken)
        return item("promo", "price", font_name, size, [(f0, f1, img, cx, cy, a) for f0, f1, img, a in segs], texts)
    size = max(int(loguniform(rng, 0.02, 0.032) * max(W, H)), 12)
    font, font_name = pick_font(rng, "sans_bold", size)
    code = promo_code(rng, brand)
    text = rng.choice([f"Use code {code}", f"Code {code} for {code[-2:]}% off", f"Use the code {code}", f"{code} at checkout"])
    bg, fg = rng.choice(PLATES[3:] + [(rng.choice(HIGHLIGHTS), WHITE)])
    img = draw_block(wrap([(w, fg) for w in text.split()], font, size, W * 0.7), font, size, plate=(bg, 255))
    f0 = 0 if rng.random() < 0.5 else int(n * rng.uniform(0.3, 0.6))
    cx, cy = free_spot(rng, img.width, img.height, W, H,
                       lambda: (0.5, rng.uniform(0.12, 0.25) if rng.random() < 0.4 else rng.uniform(0.7, 0.86)), taken)
    return item("promo", "code", font_name, size, [(f0, n, img, cx, cy, rng.choice([None, "pop", "slide"]))], [text])


def logo(rng, n, fps, W, H, taken, brand, palette=None):
    """The brand's wordmark for the whole ad: a header at the top, a corner mark, or a semi-transparent watermark."""
    size = max(int(loguniform(rng, 0.018, 0.038) * max(W, H)), 12)
    color = WHITE if rng.random() < 0.7 or palette is None else PALETTES[palette][0]
    img, text, font_name = logo_image(rng, brand, size, color)
    wr, hr = img.width / W, img.height / H
    where = rng.choices(["top", "corner", "watermark"], weights=[0.45, 0.35, 0.2])[0]
    spots = {"top": lambda: (0.5, 0.045 + hr / 2 + rng.uniform(0, 0.04)),
             "corner": lambda: (0.05 + wr / 2 if rng.random() < 0.6 else 0.95 - wr / 2, 0.045 + hr / 2 + rng.uniform(0, 0.03)),
             "watermark": lambda: (0.95 - wr / 2, rng.uniform(0.82, 0.93))}[where]
    cx, cy = free_spot(rng, img.width, img.height, W, H, spots, taken)
    return item("logo", where, font_name, size, [(0, n, img, cx, cy, None)], [text],
                alpha=rng.uniform(0.5, 0.8) if where == "watermark" else 1.0)


def cta(rng, n, fps, W, H, taken, palette=None):
    """A call-to-action button near the bottom, for the last seconds or the whole ad."""
    size = max(int(loguniform(rng, 0.016, 0.026) * max(W, H)), 12)
    pal = PALETTES[palette] if palette is not None else rng.choice(PALETTES)
    fill = pal[0] if luma(pal[0]) < 200 else pal[1]
    text = rng.choice(CTAS)
    img, font_name = pill_image(rng, text, size, fill, WHITE if luma(fill) < 150 else BLACK, arrow=rng.random() < 0.25)
    f0 = 0 if rng.random() < 0.35 else int(n * rng.uniform(0.4, 0.8))
    cx, cy = free_spot(rng, img.width, img.height, W, H, lambda: (0.5, rng.uniform(0.66, 0.86)), taken)
    return item("cta", "pill", font_name, size, [(f0, n, img, cx, cy, rng.choice(["pop", "slide", None]))], [text])


def sticker(rng, n, fps, W, H, taken):
    if rng.random() < 0.25:
        size = int(loguniform(rng, 0.045, 0.09) * max(W, H))
        img = emoji_image(rng.choice(EMOJIS), size)
        kind, font_name, text = "emoji", "NotoColorEmoji", ""
    else:
        size = max(int(loguniform(rng, 0.022, 0.045) * max(W, H)), 12)
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
    # stickers sit towards the corners (faces and products are usually central)
    cx, cy = free_spot(rng, img.width, img.height, W, H,
                       lambda: (rng.uniform(0.12, 0.35) if rng.random() < 0.5 else rng.uniform(0.65, 0.88),
                                rng.uniform(0.08, 0.3) if rng.random() < 0.5 else rng.uniform(0.6, 0.9)), taken)
    return item("sticker", kind, font_name, size, [(f0, f1, img, cx, cy, "pop" if rng.random() < 0.5 else None)], [text])


def fine_print(rng, n, fps, W, H, taken):
    size = max(int(loguniform(rng, 0.009, 0.015) * max(W, H)), 9)
    font, font_name = pick_font(rng, "sans", size)
    line = rng.choice(FINE_PRINT)
    toks = [(w, (235, 235, 235)) for w in line.split()]
    img = draw_block(wrap(toks, font, size, W * 0.85), font, size, shadow=0.7)
    f0 = 0 if rng.random() < 0.6 else int(n * rng.uniform(0.3, 0.7))
    cx, cy = free_spot(rng, img.width, img.height, W, H,
                       lambda: (0.5, rng.uniform(0.9, 0.96) if rng.random() < 0.8 else rng.uniform(0.04, 0.08)), taken)
    return item("fine_print", "plain", font_name, size, [(f0, n, img, cx, cy, None)], [line])


def annotation(rng, n, fps, W, H, taken):
    """An arrow or a ring drawn over the picture for a second or two, pointing at the product."""
    size = max(int(loguniform(rng, 0.03, 0.06) * max(W, H)), 12)
    img, kind = arrow_image(rng, size, rng.choice([(235, 30, 40), (235, 30, 40), (255, 230, 0), WHITE]))
    f0 = rng.randint(0, max(n - int(1.5 * fps), 0))
    cx, cy = free_spot(rng, img.width, img.height, W, H, lambda: (rng.uniform(0.25, 0.75), rng.uniform(0.3, 0.7)), taken)
    return item("annotation", kind, "", size, [(f0, min(f0 + int(rng.uniform(1, 3) * fps), n), img, cx, cy, "pop")], [""])


# ---------------------------------------------------------------- planning
ORDER = ["logo", "headline", "caption", "native", "cta", "promo", "fine_print", "sticker", "annotation"]


def outside(segments, texts, quiet):
    """Segments with the frames of `quiet` ranges cut out (text slides and cards carry their own text)."""
    out_s, out_t = [], []
    for (f0, f1, *rest), t in zip(segments, texts):
        pieces = [(f0, f1)]
        for q0, q1 in quiet:
            pieces = [p for a, b in pieces for p in ((a, min(b, q0)), (max(a, q1), b)) if p[1] - p[0] > 0]
        for k, (a, b) in enumerate(pieces):
            out_s.append((a, b, *rest[:3], rest[3] if k == 0 and a == f0 else None))
            out_t.append(t)
    return out_s, out_t


def item_meta(items, W, H):
    """meta.json entries: every item without its images, with "boxes" [f0, f1, x0, y0, x1, y1] per segment lined up
    with "texts" (the words shown in that segment; "" for shapes and emoji), the ground truth for OCR-based metrics."""
    meta = []
    for it in items:
        boxes = []
        for f0, f1, img, cx, cy, _ in it["segments"]:
            x0, y0 = place(img.width, img.height, cx, cy, W, H)
            boxes.append([f0, f1, x0, y0, min(x0 + img.width, W), min(y0 + img.height, H)])
        meta.append({k: v for k, v in it.items() if k != "segments"} | {"boxes": boxes})
    return meta


SOURCE_TEXT_FREE = ("logo", "cta", "promo", "fine_print", "annotation")    # items that may join text already in the footage


def plan_text(n, fps, W, H, rng, config, style="ugc", theme=None, brand=None, palette=None, cuts=(), quiet=(),
              src_text=False):
    """Decide and pre-render every text item for an n-frame W x H ad of the given style. cuts: frames where shots
    start (per-shot headlines and prices); quiet: frame ranges without overlays; src_text: the footage already has
    burned-in captions (adup.hq.source_text), so no captions, headlines or native text are added. Returns (items, meta
    for meta.json)."""
    p = {k: v for k, v in config["by_style"][style].items() if not src_text or k in SOURCE_TEXT_FREE}
    brand = brand or brand_name(rng)
    taken, items = [], []
    for kind in ORDER:
        if rng.random() >= p.get(kind, 0.0):
            continue
        it = {"logo": lambda: logo(rng, n, fps, W, H, taken, brand, palette),
              "headline": lambda: headline(rng, n, fps, W, H, taken, theme, cuts),
              "caption": lambda: caption_track(rng, n, fps, W, H, taken, config, theme),
              "native": lambda: native_text(rng, n, fps, W, H, taken, theme),
              "cta": lambda: cta(rng, n, fps, W, H, taken, palette),
              "promo": lambda: promo(rng, n, fps, W, H, taken, cuts, brand),
              "fine_print": lambda: fine_print(rng, n, fps, W, H, taken),
              "sticker": lambda: sticker(rng, n, fps, W, H, taken),
              "annotation": lambda: annotation(rng, n, fps, W, H, taken)}[kind]()
        if it is None:
            continue
        it["segments"], it["texts"] = outside(it["segments"], it["texts"], quiet)
        if it["segments"]:
            items.append(it)
    return items, item_meta(items, W, H)


# ---------------------------------------------------------------- compositing
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


def animated(img, anim, k, H):
    """k = frames since the segment started -> (image, alpha, vertical offset in pixels).
    pop: 118% -> 100% over 3 frames; fade: 3-frame fade-in; slide: rises 4% of the frame height while fading in over 5."""
    if anim == "pop" and k < 3:
        s = [1.18, 1.08, 1.0][k]
        return img.resize((max(int(img.width * s), 1), max(int(img.height * s), 1)), Image.BILINEAR), 1.0, 0
    if anim == "fade" and k < 3:
        return img, (k + 1) / 4, 0
    if anim == "slide" and k < 5:
        e = 1 - (1 - (k + 1) / 6) ** 3
        return img, e, round((1 - e) * 0.04 * H)
    return img, 1.0, 0


def draw_text(frame, i, items):
    """Burn every item active at frame index i into frame (H x W x 3 uint8, modified in place)."""
    H = frame.shape[0]
    for it in items:
        for f0, f1, img, cx, cy, anim in it["segments"]:
            if f0 <= i < f1:
                im, a, dy = animated(img, anim, i - f0, H)
                paste(frame, im, cx, cy + dy, a * it.get("alpha", 1.0))


def draw_mask(mask, i, items):
    """Alpha (0-255) of every item active at frame index i, max-combined into mask (H x W uint8, modified in place)."""
    H, W = mask.shape
    for it in items:
        for f0, f1, img, cx, cy, anim in it["segments"]:
            if f0 <= i < f1:
                im, a, dy = animated(img, anim, i - f0, H)
                arr = np.asarray(im)
                h, w = arr.shape[:2]
                x0, y0 = place(w, h, cx, cy + dy, W, H)
                x1, y1 = min(x0 + w, W), min(y0 + h, H)
                alpha = (arr[:y1 - y0, :x1 - x0, 3].astype(np.float32) * a * it.get("alpha", 1.0)).astype(np.uint8)
                np.maximum(mask[y0:y1, x0:x1], alpha, out=mask[y0:y1, x0:x1])


def shift(it, offset, end):
    """The item with its segments moved by `offset` frames and cut at `end` (for overlays planned per shot)."""
    segs = [(f0 + offset, min(f1 + offset, end), *rest) for f0, f1, *rest in it["segments"] if f0 + offset < end]
    return {**it, "segments": segs, "texts": it["texts"][:len(segs)]}
