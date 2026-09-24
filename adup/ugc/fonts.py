"""Open-licensed (OFL / Apache) fonts for burned-in captions, grouped by the caption styles seen in real ads.

The files are downloaded from github.com/google/fonts into data/assets/fonts/ (not committed). TikTok Sans is TikTok's own
"Classic" text font; Montserrat / Poppins stand in for Proxima Nova and "The Bold Font" used by caption apps.

Usage (from the repo root): .venv-iqa/bin/python -m adup.ugc.fonts     # download everything once
"""

import os
from functools import lru_cache

import requests
from PIL import ImageFont

from adup.paths import FONTS as FONT_DIR
from adup.paths import PROXIES

GOOGLE_FONTS = "https://raw.githubusercontent.com/google/fonts/main"

# (repo path, weight for variable fonts or None)
FONTS = {
    "sans": [("ofl/tiktoksans/TikTokSans[opsz,slnt,wdth,wght].ttf", 500), ("ofl/tiktoksans/TikTokSans[opsz,slnt,wdth,wght].ttf", 600),
             ("ofl/inter/Inter[opsz,wght].ttf", 500), ("ofl/montserrat/Montserrat[wght].ttf", 600),
             ("ofl/poppins/Poppins-Medium.ttf", None), ("ofl/poppins/Poppins-SemiBold.ttf", None),
             ("ofl/dmsans/DMSans[opsz,wght].ttf", 500)],
    "sans_bold": [("ofl/tiktoksans/TikTokSans[opsz,slnt,wdth,wght].ttf", 700), ("ofl/montserrat/Montserrat[wght].ttf", 800),
                  ("ofl/poppins/Poppins-Bold.ttf", None), ("ofl/inter/Inter[opsz,wght].ttf", 700),
                  ("ofl/leaguespartan/LeagueSpartan[wght].ttf", 700), ("ofl/dmsans/DMSans[opsz,wght].ttf", 700)],
    "heavy": [("ofl/montserrat/Montserrat[wght].ttf", 900), ("ofl/poppins/Poppins-Black.ttf", None),
              ("ofl/anton/Anton-Regular.ttf", None), ("ofl/bebasneue/BebasNeue-Regular.ttf", None),
              ("ofl/archivoblack/ArchivoBlack-Regular.ttf", None), ("ofl/bangers/Bangers-Regular.ttf", None),
              ("ofl/oswald/Oswald[wght].ttf", 700), ("ofl/leaguespartan/LeagueSpartan[wght].ttf", 900),
              ("ofl/tiktoksans/TikTokSans[opsz,slnt,wdth,wght].ttf", 900), ("ofl/poppins/Poppins-BlackItalic.ttf", None)],
    "serif": [("ofl/playfairdisplay/PlayfairDisplay[wght].ttf", 600), ("ofl/playfairdisplay/PlayfairDisplay-Italic[wght].ttf", 500),
              ("ofl/dmserifdisplay/DMSerifDisplay-Regular.ttf", None), ("ofl/dmserifdisplay/DMSerifDisplay-Italic.ttf", None)],
    "script": [("ofl/pacifico/Pacifico-Regular.ttf", None), ("ofl/dancingscript/DancingScript[wght].ttf", 600),
               ("ofl/lobster/Lobster-Regular.ttf", None), ("ofl/caveat/Caveat[wght].ttf", 600),
               ("apache/permanentmarker/PermanentMarker-Regular.ttf", None)],
    "mono": [("ofl/courierprime/CourierPrime-Regular.ttf", None), ("apache/specialelite/SpecialElite-Regular.ttf", None)],
}
# The CBDT (bitmap) build: Pillow cannot draw the COLRv1 build that google/fonts ships. Bitmap size is fixed at 109.
EMOJI_URL = "https://raw.githubusercontent.com/googlefonts/noto-emoji/main/2D/fonts/NotoColorEmoji.ttf"
EMOJI_SIZE = 109


def local_path(repo_path):
    return FONT_DIR / os.path.basename(repo_path)


def fetch():
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    urls = {local_path(p): f"{GOOGLE_FONTS}/{p}" for group in FONTS.values() for p, _ in group}
    urls[local_path(EMOJI_URL)] = EMOJI_URL
    for dst, url in sorted(urls.items()):
        if not dst.exists():
            r = requests.get(url, proxies=PROXIES, timeout=120)
            r.raise_for_status()
            dst.write_bytes(r.content)
        print(f"ok  {dst.name}")


@lru_cache(maxsize=512)
def load(repo_path, weight, size):
    font = ImageFont.truetype(str(local_path(repo_path)), size)
    if weight is not None:
        axes = font.get_variation_axes()
        values = []
        for a in axes:
            name = a["name"].decode() if isinstance(a["name"], bytes) else a["name"]
            values.append(weight if name.lower() == "weight" else a["default"])
        font.set_variation_by_axes(values)
    return font


@lru_cache(maxsize=1)
def emoji_font():
    return ImageFont.truetype(str(local_path(EMOJI_URL)), EMOJI_SIZE)


if __name__ == "__main__":
    fetch()
