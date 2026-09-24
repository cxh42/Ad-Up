"""Tag what UGC ads show (format / scene and product type) with zero-shot CLIP, and map the HQ pool onto the same tags.

  ads   one frame every --every seconds of each real ad -> CLIP image embedding -> best format and product label
  pool  UltraVideo clip descriptions (all ~42k clips, no download needed) -> CLIP text embedding -> same labels,
        so we can see which ad themes the HQ pool covers and how many 4K / 8K clips each theme has

  clips downloaded HQ clips: CLIP on the middle frame -> ad theme (product first, then format), with extra
        "not an ad" scene prompts; more reliable than the text descriptions, used by the UGC director

Usage (from the repo root):
  .venv-iqa/bin/python -m adup.analysis.ugc_content ads   outputs/analysis/content_ads.csv <videos...>
  .venv-iqa/bin/python -m adup.analysis.ugc_content pool  outputs/analysis/content_pool.csv
  .venv-iqa/bin/python -m adup.analysis.ugc_content clips outputs/analysis/content_clips.csv <manifest.csv...>
Needs ALL_PROXY unset for the first weight download (httpx rejects socks:// proxies).
"""

import argparse
import os
import subprocess

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image

from adup.paths import HQ

FORMATS = {
    "talking_head": "a person talking to the camera in a selfie-style phone video",
    "hands_demo": "close-up of hands holding and demonstrating a product",
    "product_closeup": "a close-up product shot of a product standing on a table",
    "unboxing": "hands opening a package and unboxing a product",
    "applying": "a person applying makeup, skincare or cream on their face",
    "before_after": "a split-screen before and after comparison",
    "screen_ui": "a screen recording of a smartphone app interface",
    "game": "gameplay footage of a video game",
    "text_slide": "a slide with large text on a plain colored background",
    "graphics": "an animated motion graphic or cartoon illustration",
    "cooking": "a person cooking food in a kitchen",
    "food": "a close-up of food or a drink",
    "pet": "a dog or a cat",
    "fitness": "a person exercising or working out",
    "fashion": "a person showing an outfit or trying on clothes",
    "car": "a car, driving, or a car interior",
    "outdoor_vlog": "a person walking and filming outdoors in a city or nature",
    "home": "a home interior such as a living room, bedroom or bathroom",
    "cleaning": "cleaning a floor, a surface or a bathroom",
    "baby": "a baby or a young child with a parent",
    "travel": "a travel destination, a hotel, a beach or an airplane",
    "people_group": "a group of people or friends together",
    "studio_brand": "a polished studio commercial with professional lighting",
    "green_screen": "a person in front of a screenshot or image background, green screen style",
}
PRODUCTS = {
    "skincare": "a skincare product such as a serum bottle or a cream jar",
    "makeup": "makeup such as lipstick, foundation or an eyeshadow palette",
    "perfume": "a perfume bottle",
    "hair": "a hair care product or a hair styling tool",
    "oral_care": "a toothbrush or teeth whitening product",
    "clothing": "clothing such as a dress, a shirt or pants",
    "shoes": "shoes or sneakers",
    "jewelry": "jewelry or a wristwatch",
    "bag": "a handbag or a backpack",
    "phone": "a smartphone or a phone case",
    "audio": "headphones or earbuds",
    "computer": "a laptop or a computer",
    "kitchen_appliance": "a kitchen appliance such as a blender or an air fryer",
    "cookware": "pots, pans or kitchen utensils",
    "cleaning_product": "a cleaning product such as a spray bottle or detergent",
    "vacuum": "a vacuum cleaner or a floor cleaning machine",
    "bedding": "a mattress, a pillow or bedding",
    "home_decor": "furniture or home decor",
    "supplement": "vitamins, supplements or pills",
    "snack": "a snack or packaged food",
    "beverage": "a drink can or a bottle of beverage",
    "coffee": "a cup of coffee",
    "meal": "a plated meal or restaurant food",
    "pet_product": "pet food or a pet toy",
    "toy": "a toy",
    "baby_product": "a baby product such as a stroller or diapers",
    "sports_gear": "sports or fitness equipment",
    "vehicle": "a car or a motorcycle",
    "finance": "a credit card, money or a banking app",
    "book": "a book or study materials",
    "none": "no product, just people or a scene",
}


# product / format labels -> ad themes of content_coverage.csv
P2T = {"makeup": "beauty", "skincare": "beauty", "perfume": "beauty", "hair": "beauty", "oral_care": "beauty",
       "clothing": "fashion", "shoes": "fashion", "jewelry": "fashion", "bag": "fashion", "snack": "food", "beverage": "food",
       "coffee": "food", "meal": "food", "cleaning_product": "home", "vacuum": "home", "bedding": "home", "home_decor": "home",
       "kitchen_appliance": "home", "cookware": "home", "supplement": "health", "pet_product": "pets", "phone": "tech_app",
       "audio": "tech_app", "computer": "tech_app", "finance": "tech_app", "toy": "baby_kids", "baby_product": "baby_kids",
       "sports_gear": "fitness_sports", "vehicle": "car", "book": "education"}
F2T = {"applying": "beauty", "fashion": "fashion", "food": "food", "cooking": "food", "cleaning": "home", "home": "home",
       "pet": "pets", "screen_ui": "tech_app", "game": "tech_app", "baby": "baby_kids", "fitness": "fitness_sports",
       "car": "car", "travel": "travel", "talking_head": "talking_head", "hands_demo": "talking_head",
       "unboxing": "talking_head", "product_closeup": "talking_head"}
NOT_AD = ["a landscape or nature scene without people", "an antique object or a historical scene",
          "an abandoned or industrial place", "wild animals in nature", "an artwork or a museum exhibit",
          "a fantasy or costume scene", "a city skyline or architecture", "a sports match in a stadium"]


def load_clip():
    model, _, pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    tok = open_clip.get_tokenizer("ViT-L-14")
    return model.eval().cuda(), pre, tok


@torch.no_grad()
def text_emb(model, tok, texts, bs=256):
    out = []
    for i in range(0, len(texts), bs):
        e = model.encode_text(tok(texts[i:i + bs]).cuda())
        out.append(torch.nn.functional.normalize(e.float(), dim=-1))
    return torch.cat(out)


def label(sim, names):
    p = (100 * sim).softmax(-1)
    best = p.argmax(-1)
    return [names[i] for i in best.tolist()], p.max(-1).values.tolist()


def frames(path, every):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps=1/{every},scale=336:-2", "-f", "image2pipe",
                          "-vcodec", "png", "-"], capture_output=True).stdout
    imgs, i = [], 0
    while True:
        j = raw.find(b"\x89PNG", i + 1)
        chunk = raw[i:j] if j != -1 else raw[i:]
        if chunk:
            import io
            imgs.append(Image.open(io.BytesIO(chunk)).convert("RGB"))
        if j == -1:
            break
        i = j
    return imgs


@torch.no_grad()
def tag_ads(videos, out, every):
    model, pre, tok = load_clip()
    tf = text_emb(model, tok, [f"a frame from a social media video ad: {t}" for t in FORMATS.values()])
    tp = text_emb(model, tok, [f"a frame showing {t}" for t in PRODUCTS.values()])
    rows = []
    for v in videos:
        imgs = frames(v, every)
        if not imgs:
            continue
        e = torch.nn.functional.normalize(model.encode_image(torch.stack([pre(i) for i in imgs]).cuda()).float(), dim=-1)
        fl, fp = label(e @ tf.T, list(FORMATS))
        pl, pp = label(e @ tp.T, list(PRODUCTS))
        rows += [{"video": v, "t": k * every, "format": a, "format_p": b, "product": c, "product_p": d}
                 for k, (a, b, c, d) in enumerate(zip(fl, fp, pl, pp))]
    pd.DataFrame(rows).to_csv(out, index=False)


# UGC-ad themes for the HQ pool, as example sentences in the style of UltraVideo's brief descriptions.
POOL_THEMES = {
    "talking_head": ["A woman talks directly to the camera in a close-up shot.", "A man speaks to the camera while sitting in his room."],
    "beauty_applying": ["A woman applies makeup to her face in front of a mirror.", "A person applies skincare cream to their face."],
    "beauty_product": ["Close-up of a cosmetic product such as a lipstick or a serum bottle.", "A perfume bottle displayed on a table."],
    "hair": ["A woman brushes and styles her long hair.", "A hairdresser cuts a client's hair in a salon."],
    "fashion": ["A woman shows her outfit and poses for the camera.", "A person tries on clothes in a clothing store."],
    "accessories": ["Close-up of sneakers on someone's feet.", "A person wearing a wristwatch and jewelry.", "A woman carries a handbag."],
    "hands_product": ["Hands hold a product and show it to the camera.", "A person demonstrates how to use a small gadget."],
    "unboxing": ["A person opens a package and unboxes a new product."],
    "tech": ["A person uses a smartphone, scrolling on the screen.", "A person types on a laptop at a desk.", "A person wearing headphones listens to music."],
    "cooking": ["A person cooks food in a kitchen, stirring a pan.", "Someone chops vegetables on a cutting board."],
    "food_drink": ["Close-up of a delicious meal on a plate.", "A person pours a drink into a glass.", "A cup of coffee on a table."],
    "eating": ["A person eats food and enjoys the taste."],
    "kitchen_appliance": ["A blender mixes ingredients in a kitchen.", "A person uses a coffee machine."],
    "cleaning": ["A person cleans the floor with a mop or a vacuum cleaner.", "Someone wipes a kitchen counter clean."],
    "home": ["A cozy living room with a sofa and decorations.", "A person relaxes in a bedroom.", "A modern bathroom interior."],
    "baby_kids": ["A mother plays with her baby.", "A child plays with toys on the floor."],
    "pets": ["A dog plays with its owner at home.", "A cat sits on a couch."],
    "fitness": ["A person exercises at the gym lifting weights.", "A woman does yoga at home."],
    "health": ["A person takes vitamins or pills with water.", "A doctor talks to a patient."],
    "car": ["A person drives a car.", "Interior view of a car dashboard."],
    "travel": ["A person walks through a hotel room.", "A tourist relaxes at a beach resort."],
    "lifestyle_outdoor": ["Friends laugh together in a park.", "A person walks down a city street filming."],
    "office_work": ["A person works at an office desk.", "People have a meeting in an office."],
    "shopping": ["A person shops in a store picking items from the shelves."],
    "sports": ["A person plays tennis on a court.", "A person rides a bicycle outdoors."],
}
DISTRACTORS = ["A bird perched on a branch in nature.", "Wild animals roam the savanna.", "Aerial view of mountains and forests.",
               "A city skyline at night.", "An underwater scene with fish and coral.", "A medieval knight in armor.",
               "A factory with large industrial machines.", "A crowd at a concert in front of a stage.",
               "Scientific laboratory equipment.", "A historical reenactment in period costumes.", "Planets and stars in outer space.",
               "A cartoon animation.", "A professional sports match in a stadium.", "Soldiers in military uniforms.",
               "A religious ceremony in a temple.", "A close-up of an insect on a leaf.", "A waterfall in a forest."]
POOL_MIN_SIM = 0.45


@torch.no_grad()
def mpnet_encoder():
    from transformers import AutoModel, AutoTokenizer
    name = "sentence-transformers/all-mpnet-base-v2"
    tok, model = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).cuda().eval()

    @torch.no_grad()
    def enc(texts, bs=256):
        out = []
        for i in range(0, len(texts), bs):
            b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
            h = model(**b).last_hidden_state
            e = (h * b.attention_mask[..., None]).sum(1) / b.attention_mask.sum(1, keepdim=True)
            out.append(torch.nn.functional.normalize(e, dim=-1))
        return torch.cat(out)
    return enc


@torch.no_grad()
def tag_pool(out):
    enc = mpnet_encoder()
    s = pd.read_csv(HQ / "ultravideo" / "short.csv", usecols=["clip_id", "url", "frame_width", "frame_height",
                                                             "total_frames", "fps", "Brief Description"])
    s = s[s.frame_width >= 3800].reset_index(drop=True)
    e = enc(s["Brief Description"].fillna("").tolist())
    names = [k for k, v in POOL_THEMES.items() for _ in v]
    theme_sim = e @ enc([x for v in POOL_THEMES.values() for x in v]).T
    best = torch.stack([theme_sim[:, [i for i, n in enumerate(names) if n == k]].max(1).values for k in POOL_THEMES], 1)
    dis = (e @ enc(DISTRACTORS).T).max(1).values
    sim, idx = best.max(1)
    keep = (sim >= POOL_MIN_SIM) & (sim > dis)
    s["theme"] = [list(POOL_THEMES)[i] if k else "other" for i, k in zip(idx.tolist(), keep.tolist())]
    s["theme_sim"] = sim.cpu().numpy().round(3)
    s["res"] = np.where(s.frame_width >= 7680, "8K", "4K")
    s.drop(columns=["Brief Description"]).to_csv(out, index=False)


@torch.no_grad()
def tag_clips(manifests, out):
    model, pre, tok = load_clip()
    clips = pd.concat([pd.read_csv(m) for m in manifests], ignore_index=True)
    fmt = list(FORMATS) + [f"not_ad_{i}" for i in range(len(NOT_AD))]
    tf = text_emb(model, tok, [f"a frame from a video: {t}" for t in list(FORMATS.values()) + NOT_AD])
    tp = text_emb(model, tok, [f"a frame showing {t}" for t in PRODUCTS.values()])
    rows = []
    for f in clips.file:
        if not os.path.exists(f):
            continue
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", "1.5", "-i", f, "-frames:v", "1", "-vf", "scale=448:-2", "-f",
                              "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
        if not raw:
            continue
        import io
        e = torch.nn.functional.normalize(model.encode_image(pre(Image.open(io.BytesIO(raw)).convert("RGB"))[None].cuda()).float(), dim=-1)
        (fl,), (fp,) = label(e @ tf.T, fmt)
        (pl,), (pp,) = label(e @ tp.T, list(PRODUCTS))
        theme = "none" if fl.startswith("not_ad") else (P2T.get(pl) if pl != "none" and pp > 0.3 else None) or F2T.get(fl, "other")
        rows.append({"file": f, "format": fl, "format_p": round(fp, 3), "product": pl, "product_p": round(pp, 3), "theme": theme})
    pd.DataFrame(rows).to_csv(out, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ads", "pool", "clips"])
    ap.add_argument("out")
    ap.add_argument("videos", nargs="*", help="ads: video files; clips: clip manifests")
    ap.add_argument("--every", type=float, default=2.0)
    args = ap.parse_args()
    os.environ.pop("ALL_PROXY", None)
    os.environ.pop("all_proxy", None)
    if args.mode == "ads":
        tag_ads(args.videos, args.out, args.every)
    elif args.mode == "clips":
        tag_clips(args.videos, args.out)
    else:
        tag_pool(args.out)


if __name__ == "__main__":
    main()
