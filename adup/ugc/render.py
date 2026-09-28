"""Render one shot of a UGC sequence into GT frames of a given box size: video, still, layout or text slide.

Shot kinds (spec dicts from adup.ugc.director, or plain {"src", "start", "frames", "src_fps"} = video):
  video   crop of a source clip (face-aware placement) + virtual phone camera (adup.ugc.camera)
  still   a high-resolution photo animated by the virtual camera (Ken Burns / handheld), e.g. product photos
  layout  a composition of sub-shots, as made in CapCut / TikTok:
            fit_blur  landscape clip inside a portrait frame over a blurred, darkened copy of itself
            fit_solid the same over solid bars (letterbox), often with the text in the bars
            split     two clips stacked (before/after, two angles)       duet  two clips side by side
            grid4     a 2 x 2 collage of four clips (several creators, several shades)
            pip       an inset clip (rounded corners, border) over a full-frame clip
            phone     a screen recording inside a phone frame, over a blurred clip or a brand gradient
  slide   a card made in an editor or a brand template (motion graphics): a solid / gradient background, optionally a
          product photo in a shape, and text elements animating in; role hook (one big line), card (photo + headline)
          or end (logo, tagline, photo, CTA button, fine print)
  screen  a phone screen recording: a tall app page (adup.hq.ui_screens) scrolled by swipes and pauses;
          offsets are whole pixels, so frames are exact copies of the rendered UI
Text on slides is returned as `overlays` (text items, frames relative to the shot) for the frame-level text pass, so it
is in mask.mkv and meta.json like every other overlay. Nothing is ever upscaled except the blurred backgrounds of
fit_blur and phone, which carry no detail by design.
"""

import os
import random
import urllib.request

import cv2
import numpy as np

from adup.media import max_crop, place_crop, probe, read_frames, stream_frames
from adup.paths import HQ, PROXY
from adup.ugc.camera import plan_camera, warp
from adup.ugc.text import (
    FINE_PRINT,
    PALETTES,
    WHITE,
    brand_name,
    draw_block,
    fitted_font,
    free_spot,
    item,
    logo_image,
    luma,
    pick_font,
    pill_image,
    short_line,
    supersampled,
    wrap,
)

STILLS = HQ / "unsplash_lite" / "images"          # Unsplash Lite photos, fetched on first use


def even(x):
    return max(int(round(x / 2)) * 2, 2)


def still_path(src, min_w, min_h=0):
    """Local file for a still: a path, or 'unsplash:<id>|<image_url>|<width>x<height>' fetched once, just wide enough
    to cover min_w x min_h at the photo's aspect ratio (never above its original size)."""
    if not src.startswith("unsplash:"):
        return src
    pid, url, *size = src[len("unsplash:"):].split("|")
    pw, ph = map(int, size[0].split("x")) if size else (10 ** 5, 10 ** 5 * 2 // 3)    # unknown: assume 3:2 landscape
    w = min(int(max(min_w, min_h * pw / ph)) + 2, pw)
    dst = STILLS / f"{pid}_w{w}.jpg"
    if not dst.exists():
        STILLS.mkdir(parents=True, exist_ok=True)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({"https": PROXY, "http": PROXY}))
        with opener.open(f"{url}?w={w}&q=95&fm=jpg", timeout=120) as r, open(dst, "wb") as f:
            f.write(r.read())
    return str(dst)


class ShotRenderer:
    """Frames of one shot at bw x bh. `info` describes crop, camera and layout for meta.json."""

    def __init__(self, shot, bw, bh, fps, rng, cfg):
        self.shot, self.bw, self.bh, self.fps, self.rng, self.cfg = shot, bw, bh, fps, rng, cfg
        self.kind = shot.get("kind", "video")
        self.n = shot["frames"]
        self.info = {"kind": self.kind, "box": [bw, bh]}
        self.overlays = []
        getattr(self, f"_setup_{self.kind}")()

    # ---------------------------------------------------------------- video
    def _camera(self, headroom, still=False):
        c = {**self.cfg["camera"], **(self.cfg["still_camera"] if still else {}), **self.shot.get("camera", {})}
        if "zoom" in self.shot:                                  # punch-in framing chosen by the director
            c["base_zoom"] = [self.shot["zoom"], self.shot["zoom"]]
        P, S, params = plan_camera(self.rng, self.n, self.fps, c, headroom)
        static = np.allclose(P[:, 1:], 0) and np.allclose(P[:, 0], P[0, 0]) and abs(S - 1) < 1e-6
        return P, S, params, static

    def _setup_video(self):
        s = self.shot
        sw, sh, fps, nb = probe(s["src"])
        cw, ch = max_crop(sw, sh, self.bw, self.bh)
        if cw < self.bw:
            raise ValueError(f"{os.path.basename(s['src'])} {sw}x{sh} cannot fill {self.bw}x{self.bh} without upscaling")
        src_n = int(self.n * s["src_fps"] / self.fps) + 1
        mid = read_frames(s["src"], f"select=eq(n\\,{s['start'] + src_n // 2})", sw, sh, 1)[0]
        x, y, how = place_crop(mid, cw, ch, self.bw, self.bh, self.rng)
        self.preview = cv2.resize(mid[y:y + ch, x:x + cw], (even(self.bw / 4), even(self.bh / 4)), interpolation=cv2.INTER_AREA)
        self.P, S, cam, self.static = self._camera(cw / self.bw)
        self.ww, self.wh = even(self.bw * S), even(self.bh * S)
        vf = f"trim=start_frame={s['start']},setpts=PTS-STARTPTS"
        if abs(s["src_fps"] - self.fps) > 0.01:
            vf += f",fps={self.fps}"
        self.vf = f"{vf},crop={cw}:{ch}:{x}:{y},scale={self.ww}:{self.wh}:flags=lanczos"
        self.info.update({"src": s["src"], "start": s["start"], "crop": [cw, ch, x, y], "placement": how, "camera": cam})

    def _frames_video(self):
        for i, f in enumerate(stream_frames(self.shot["src"], self.vf, self.ww, self.wh, self.n)):
            yield f if self.static else warp(f, *self.P[i], self.bw, self.bh)

    # ---------------------------------------------------------------- still
    def _setup_still(self):
        path = still_path(self.shot["src"], self.bw * 1.5, self.bh * 1.5)     # room for the camera's motion
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        sh, sw = img.shape[:2]
        cw, ch = max_crop(sw, sh, self.bw, self.bh)
        if cw < self.bw:
            raise ValueError(f"{path} {sw}x{sh} cannot fill {self.bw}x{self.bh} without upscaling")
        x, y, how = place_crop(img, cw, ch, self.bw, self.bh, self.rng)
        self.P, S, cam, self.static = self._camera(cw / self.bw, still=True)
        self.work = cv2.resize(img[y:y + ch, x:x + cw], (even(self.bw * S), even(self.bh * S)), interpolation=cv2.INTER_AREA)
        self.preview = cv2.resize(self.work, (even(self.bw / 4), even(self.bh / 4)), interpolation=cv2.INTER_AREA)
        self.info.update({"src": self.shot["src"], "crop": [cw, ch, x, y], "placement": how, "camera": cam})

    def _frames_still(self):
        for i in range(self.n):
            yield self.work.copy() if self.static else warp(self.work, *self.P[i], self.bw, self.bh)

    # ---------------------------------------------------------------- screen recording
    def _setup_screen(self):
        img = cv2.cvtColor(cv2.imread(self.shot["src"]), cv2.COLOR_BGR2RGB)
        if img.shape[1] < self.bw:
            raise ValueError(f"{self.shot['src']} is {img.shape[1]} px wide, cannot fill {self.bw} without upscaling")
        if img.shape[1] > self.bw:
            img = cv2.resize(img, (self.bw, int(img.shape[0] * self.bw / img.shape[1])), interpolation=cv2.INTER_AREA)
        if img.shape[0] < self.bh:
            img = np.pad(img, ((0, self.bh - img.shape[0]), (0, 0), (0, 0)), mode="edge")
        self.page = img
        room, r, y, ys = img.shape[0] - self.bh, self.rng, 0.0, []
        while len(ys) < self.n:                                  # pause, then an ease-out swipe
            ys += [y] * int(r.uniform(0.3, 1.2) * self.fps)
            dist = r.uniform(0.3, 0.9) * self.bh * (1 if r.random() < 0.85 or y <= 0 else -0.5)
            k = int(r.uniform(0.25, 0.6) * self.fps)
            ys += [float(np.clip(y + dist * (1 - (1 - (i + 1) / k) ** 3), 0, room)) for i in range(k)]
            y = ys[-1]
        self.offsets = np.round(ys[:self.n]).astype(int)
        self.preview = cv2.resize(img[:self.bh], (even(self.bw / 4), even(self.bh / 4)), interpolation=cv2.INTER_AREA)
        self.info.update({"src": self.shot["src"], "scroll_px": int(self.offsets.max())})

    def _frames_screen(self):
        for y in self.offsets:
            yield self.page[y:y + self.bh].copy()

    # ---------------------------------------------------------------- slide
    def _setup_slide(self):
        s, bw, bh = self.shot, self.bw, self.bh
        pr = random.Random(int(self.rng.integers(2 ** 31)))
        role = s.get("role", "hook")
        if s.get("palette") is not None:
            bg, accent, fg = PALETTES[s["palette"]]
        else:                                                   # an editor's random solid colour
            bg = tuple(int(v) for v in self.rng.integers(0, 256, 3))
            accent, fg = WHITE, (255, 255, 255) if luma(bg) < 140 else (20, 20, 20)
        c1 = np.array(bg, np.float32)
        c2 = c1 * pr.uniform(0.7, 1.0) if pr.random() < 0.6 else c1
        if pr.random() < 0.3:                                   # radial: lighter in the middle
            yy, xx = np.mgrid[0:bh, 0:bw].astype(np.float32)
            g = np.clip(np.hypot((xx - bw / 2) / bw, (yy - bh / 2) / bh) * 1.6, 0, 1)[..., None]
        else:
            g = np.linspace(0, 1, bh, dtype=np.float32)[:, None, None] * np.ones((1, bw, 1), np.float32)
        self.card = (c1 * (1 - g) + c2 * g).astype(np.uint8)
        taken, n = [], self.n
        self.photo = None
        wide = bw > bh                                          # landscape cards: photo on the left, text on the right
        if s.get("image"):
            self.photo = self._card_photo(s["image"], pr, 0.5 if role == "card" or wide else 0.42, 0.3 if wide else 0.5)
            if self.photo is not None:
                _, _, x, y, w, h = self.photo
                taken.append((x, y, x + w, y + h))
        brand = s.get("brand") or brand_name(pr)
        theme = s.get("theme")
        ov = self.overlays
        side = wide and self.photo is not None
        tx, tw = 0.5, 0.82                                      # text column: centre and width (fractions of bw)
        if side:                                                # the space right of the photo, with margins
            right = (self.photo[2] + self.photo[4]) / bw
            tx, tw = (right + 1) / 2, (1 - right) * 0.85

        def text_block(text, family, size, color, max_w=0.82):
            font, name = pick_font(pr, family, size)
            return draw_block(wrap([(w, color) for w in text.split()], font, size, bw * max_w), font, size), name

        if role == "hook":
            size = max(int(max(bw, bh) * pr.uniform(0.04, 0.07)), 16)
            text = s.get("text") or short_line(pr, theme)
            img, name = text_block(text, pr.choice(["heavy", "sans_bold", "serif"]), size, fg, 0.8)
            ov.append(item("slide_text", "hook", name, size, [(0, n, img, bw / 2, bh / 2, pr.choice([None, "pop"]))], [text]))
        elif role == "card":
            text = s.get("text") or short_line(pr, theme)
            text = text.upper() if pr.random() < 0.4 else text
            if self.photo is None:                              # type-only card: big text in the middle
                size = max(int(max(bw, bh) * pr.uniform(0.045, 0.075)), 16)
                spot = lambda: (0.5, pr.uniform(0.38, 0.55))
            elif side:                                          # headline next to the product photo
                size = max(int(max(bw, bh) * pr.uniform(0.032, 0.05)), 14)
                spot = lambda: (tx, pr.uniform(0.42, 0.58))
            else:                                               # headline above or below the product photo
                size = max(int(max(bw, bh) * pr.uniform(0.032, 0.055)), 14)
                above = pr.random() < 0.6
                spot = lambda: (0.5, pr.uniform(0.1, 0.22) if above else pr.uniform(0.75, 0.86))
            font, name, size = fitted_font(pr, pr.choice(["heavy", "sans_bold", "serif"]), size, [text], bw * tw)
            img = draw_block(wrap([(w, fg) for w in text.split()], font, size, bw * tw), font, size)
            cx, cy = free_spot(pr, img.width, img.height, bw, bh, spot, taken)
            ov.append(item("slide_text", "headline", name, size, [(3, n, img, cx, cy, pr.choice(["slide", "pop", "fade"]))], [text]))
            if pr.random() < 0.4:
                lsize = max(int(max(bw, bh) * pr.uniform(0.018, 0.028)), 12)
                limg, ltext, lname = logo_image(pr, brand, lsize, fg)
                cx, cy = free_spot(pr, limg.width, limg.height, bw, bh, lambda: (tx, pr.choice([0.05, 0.93])), taken)
                ov.append(item("slide_text", "logo", lname, lsize, [(0, n, limg, cx, cy, None)], [ltext]))
        else:                                                   # end card
            lsize = max(int(max(bw, bh) * pr.uniform(0.035, 0.07)), 14)
            limg, ltext, lname = logo_image(pr, brand, lsize, accent if luma(bg) > 200 and luma(accent) < 200 else fg)
            ly = 0.35 if side else 0.14 if self.photo is not None else pr.uniform(0.3, 0.42)
            cx, cy = free_spot(pr, limg.width, limg.height, bw, bh, lambda: (tx, ly), taken)
            ov.append(item("slide_text", "logo", lname, lsize, [(0, n, limg, cx, cy, pr.choice([None, "pop", "fade"]))], [ltext]))
            if pr.random() < 0.5:
                tsize = max(int(max(bw, bh) * pr.uniform(0.015, 0.024)), 11)
                tag = short_line(pr, theme, 3, 7)
                timg, tname = text_block(tag, pr.choice(["sans", "serif"]), tsize, fg, min(tw, 0.7))
                cx, cy = free_spot(pr, timg.width, timg.height, bw, bh,
                                   lambda: (tx, ly + pr.uniform(0.05, 0.08) * (1.6 if wide else 1)), taken)
                ov.append(item("slide_text", "tagline", tname, tsize, [(4, n, timg, cx, cy, "fade")], [tag]))
            if pr.random() < 0.7:
                csize = max(int(max(bw, bh) * pr.uniform(0.016, 0.025)), 12)
                text = s.get("text") or pr.choice(["Shop now", "Learn more", "Download now", "Get yours today"])
                pimg, pname = pill_image(pr, text, csize, accent, bg if luma(bg) != luma(accent) else (0, 0, 0))
                cy0 = 0.66 if side else 0.83 if self.photo is not None else pr.uniform(0.6, 0.72)
                cx, cy = free_spot(pr, pimg.width, pimg.height, bw, bh, lambda: (tx, cy0), taken)
                ov.append(item("slide_text", "cta", pname, csize, [(8, n, pimg, cx, cy, pr.choice(["pop", "slide"]))], [text]))
            if pr.random() < 0.3:
                fsize = max(int(max(bw, bh) * 0.011), 9)
                line = pr.choice(FINE_PRINT)
                fimg, fname = text_block(line, "sans", fsize, fg, 0.85)
                cx, cy = free_spot(pr, fimg.width, fimg.height, bw, bh, lambda: (0.5, 0.95), taken)
                ov.append(item("slide_text", "fine_print", fname, fsize, [(0, n, fimg, cx, cy, None)], [line]))
        self.preview = cv2.resize(self.card, (even(bw / 4), even(bh / 4)), interpolation=cv2.INTER_AREA)
        self.info.update({"role": role, "palette": s.get("palette"), "photo": s.get("image") if self.photo is not None else None,
                          "text": [t for it in ov for t in it["texts"]]})

    def _card_photo(self, src, pr, cy, cx=0.5):
        """A product photo for a card, cut to a circle, a rounded rectangle or a full-width band, with a soft shadow.
        Returns (rgb, alpha, x, y, w, h), or None when the photo is too small to fill the shape without upscaling."""
        bw, bh = self.bw, self.bh
        shape = pr.choice(["circle", "round", "round"] + (["band"] if bh >= bw else []))
        base = min(bw, bh)                                      # shapes follow the short side (landscape cards too)
        if shape == "circle":
            w = h = even(base * pr.uniform(0.55, 0.75))
        elif shape == "round":
            w = even(base * pr.uniform(0.68, 0.86))
            h = even(w * pr.choice([1.0, 1.25]))
        else:
            w, h = bw, even(bh * pr.uniform(0.38, 0.5))
        if h > 0.8 * bh:                                        # leave room for the headline
            w, h = even(w * 0.8 * bh / h), even(0.8 * bh)
        img = cv2.cvtColor(cv2.imread(still_path(src, w, h)), cv2.COLOR_BGR2RGB)
        cw, ch = max_crop(img.shape[1], img.shape[0], w, h)
        if cw < w:
            return None
        x0, y0 = (img.shape[1] - cw) // 2, (img.shape[0] - ch) // 2
        rgb = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_AREA)
        if shape == "circle":
            m = supersampled(w, h, lambda d, k: d.ellipse([0, 0, w * k - 1, h * k - 1], fill=255))
        elif shape == "round":
            m = supersampled(w, h, lambda d, k: d.rounded_rectangle([0, 0, w * k - 1, h * k - 1], radius=w * k * 0.07, fill=255))
        else:
            m = supersampled(w, h, lambda d, k: d.rectangle([0, 0, w * k, h * k], fill=255))
        alpha = np.asarray(m, np.float32)[..., None] / 255
        x, y = int(np.clip(cx * bw - w / 2, 0, bw - w)), int(np.clip(cy * bh - h / 2, 0, bh - h))
        if shape != "band":                                     # drop shadow under the shape
            sh = cv2.GaussianBlur(alpha[..., 0], (0, 0), w * 0.03) * 0.45
            dy = int(w * 0.02)
            region = self.card[y + dy:y + dy + h, x:x + w].astype(np.float32)
            self.card[y + dy:y + dy + h, x:x + w] = (region * (1 - sh[:region.shape[0], :, None])).astype(np.uint8)
        return rgb, alpha, x, y, w, h

    def _frames_slide(self):
        for i in range(self.n):
            f = self.card.copy()
            if self.photo is not None:
                rgb, alpha, x, y, w, h = self.photo
                if i < 8:                                       # scale in 90% -> 100% and fade in
                    e = 1 - (1 - (i + 1) / 9) ** 3
                    sw, sh = even(w * (0.9 + 0.1 * e)), even(h * (0.9 + 0.1 * e))
                    rgb_i = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
                    a_i = cv2.resize(alpha, (sw, sh), interpolation=cv2.INTER_AREA)[..., None] * e
                    xi, yi = x + (w - sw) // 2, y + (h - sh) // 2
                else:
                    rgb_i, a_i, xi, yi, sw, sh = rgb, alpha, x, y, w, h
                region = f[yi:yi + sh, xi:xi + sw].astype(np.float32)
                f[yi:yi + sh, xi:xi + sw] = (region * (1 - a_i) + rgb_i * a_i + 0.5).astype(np.uint8)
            yield f

    # ---------------------------------------------------------------- layout
    def _setup_layout(self):
        s, bw, bh, r = self.shot, self.bw, self.bh, self.rng
        lay = s["layout"]
        mk = lambda part, w, h: ShotRenderer({**part, "frames": self.n}, even(w), even(h), self.fps, r, self.cfg)
        if lay in ("fit_blur", "fit_solid"):
            ch = even(bw * 9 / 16)
            self.parts = [mk(s["parts"][0], bw, ch)]
            self.fg_y = int((bh - ch) * r.uniform(0.3, 0.5)) // 2 * 2
            self.darken = float(r.uniform(0.55, 0.85))
            if lay == "fit_solid":                              # black bars mostly, else white or the brand colour
                c = r.random()
                self.solid = (0, 0, 0) if c < 0.6 else (255, 255, 255) if c < 0.8 or s.get("palette") is None \
                    else PALETTES[s["palette"]][0]
        elif lay == "split":
            gap = even(r.choice([0, 0, 4, 8]))
            self.gap, self.gap_col = gap, (255, 255, 255) if r.random() < 0.5 else (0, 0, 0)
            self.parts = [mk(p, bw, (bh - gap) / 2) for p in s["parts"][:2]]
        elif lay == "duet":
            self.parts = [mk(p, bw / 2, bh) for p in s["parts"][:2]]
        elif lay == "grid4":
            gap = even(r.choice([0, 4, 8]))
            self.gap, self.gap_col = gap, (255, 255, 255) if r.random() < 0.5 else (0, 0, 0)
            self.parts = [mk(p, (bw - gap) / 2, (bh - gap) / 2) for p in s["parts"][:4]]
        elif lay == "phone":
            ph = even(bh * r.uniform(0.66, 0.8))
            pw = even(ph * 9 / 19.5)
            bez = max(even(pw * 0.03), 4)
            iw, ih = pw - 2 * bez, ph - 2 * bez
            self.parts = [mk(s["parts"][0], iw, ih)] + [mk(p, bw, bh) for p in s["parts"][1:2]]
            self.phone_xy = (even((bw - pw) / 2), even((bh - ph) * r.uniform(0.3, 0.6)))
            self.bez = bez
            body = supersampled(pw, ph, lambda d, k: d.rounded_rectangle([0, 0, pw * k - 1, ph * k - 1], radius=pw * k * 0.14, fill=255))
            screen = supersampled(iw, ih, lambda d, k: d.rounded_rectangle([0, 0, iw * k - 1, ih * k - 1], radius=iw * k * 0.12, fill=255))
            iw2, ih2 = even(iw * 0.3), max(even(iw * 0.085), 4)
            island = supersampled(iw2, ih2, lambda d, k: d.rounded_rectangle([0, 0, iw2 * k - 1, ih2 * k - 1], radius=ih2 * k / 2, fill=255))
            self.body_mask = np.asarray(body, np.float32)[..., None] / 255
            self.screen_mask = np.asarray(screen, np.float32)[..., None] / 255
            self.island = (np.asarray(island, np.float32)[..., None] / 255, (iw - iw2) // 2, int(ih * 0.015))
            if len(self.parts) == 1:                            # no clip behind the phone: a brand gradient
                c1 = np.array(PALETTES[s["palette"]][0] if s.get("palette") is not None
                              else r.integers(0, 256, 3), np.float32)
                g = np.linspace(0, 1, bh, dtype=np.float32)[:, None, None]
                self.phone_bg = (c1 * (1 - g) + c1 * float(r.uniform(0.6, 0.9)) * g).repeat(bw, 1).astype(np.uint8)
        elif lay == "pip":
            iw = even(min(bw, bh) * r.uniform(0.3, 0.42))          # the short side: landscape insets fit too
            ih = even(iw * (16 / 9 if r.random() < 0.6 else 1.0))
            self.parts = [mk(s["parts"][0], bw, bh), mk(s["parts"][1], iw, ih)]
            x = even(bw * r.uniform(0.04, 0.1)) if r.random() < 0.5 else even(bw * 0.96 - iw)
            y = even(bh * r.uniform(0.08, 0.18)) if r.random() < 0.5 else even(bh * 0.72 - ih)
            self.inset_xy = (int(np.clip(x, 0, bw - iw)), int(np.clip(y, 0, bh - ih)))
            m = np.zeros((ih, iw), np.uint8)
            rad = int(iw * 0.08)
            cv2.rectangle(m, (rad, 0), (iw - rad, ih), 255, -1)
            cv2.rectangle(m, (0, rad), (iw, ih - rad), 255, -1)
            for cx, cy in [(rad, rad), (iw - rad - 1, rad), (rad, ih - rad - 1), (iw - rad - 1, ih - rad - 1)]:
                cv2.circle(m, (cx, cy), rad, 255, -1, lineType=cv2.LINE_AA)
            self.inset_mask = cv2.GaussianBlur(m, (3, 3), 0)[..., None].astype(np.float32) / 255
        else:
            raise ValueError(lay)
        self.preview = self.parts[0].preview
        self.info.update({"layout": lay, "parts": [p.info for p in self.parts]})

    def _frames_layout(self):
        lay, bw, bh = self.shot["layout"], self.bw, self.bh
        for frames in zip(*[p.frames() for p in self.parts]):
            if lay == "fit_blur":
                fg = frames[0]
                small = cv2.resize(fg, (bw // 16, bh // 16), interpolation=cv2.INTER_AREA)     # cover the frame
                bg = cv2.GaussianBlur(cv2.resize(small, (bw, bh), interpolation=cv2.INTER_LINEAR), (0, 0), bw / 60)
                out = (bg.astype(np.float32) * self.darken).astype(np.uint8)
                out[self.fg_y:self.fg_y + fg.shape[0]] = fg
            elif lay == "split":
                out = np.empty((bh, bw, 3), np.uint8)
                out[:] = self.gap_col
                out[:frames[0].shape[0]] = frames[0]
                out[bh - frames[1].shape[0]:] = frames[1]
            elif lay == "duet":
                out = np.concatenate(frames, 1)
            elif lay == "fit_solid":
                fg = frames[0]
                out = np.empty((bh, bw, 3), np.uint8)
                out[:] = self.solid
                out[self.fg_y:self.fg_y + fg.shape[0]] = fg
            elif lay == "grid4":
                out = np.empty((bh, bw, 3), np.uint8)
                out[:] = self.gap_col
                for k, f in enumerate(frames):
                    h, w = f.shape[:2]
                    x, y = (0 if k % 2 == 0 else bw - w), (0 if k < 2 else bh - h)
                    out[y:y + h, x:x + w] = f
            elif lay == "phone":
                if len(frames) > 1:                             # the clip behind, blurred and darkened
                    small = cv2.resize(frames[1], (bw // 16, bh // 16), interpolation=cv2.INTER_AREA)
                    out = (cv2.GaussianBlur(cv2.resize(small, (bw, bh), interpolation=cv2.INTER_LINEAR), (0, 0), bw / 60)
                           .astype(np.float32) * 0.7).astype(np.uint8)
                else:
                    out = self.phone_bg.copy()
                x, y = self.phone_xy
                ph, pw = self.body_mask.shape[:2]
                reg = out[y:y + ph, x:x + pw].astype(np.float32)
                reg = reg * (1 - self.body_mask) + np.float32(18) * self.body_mask
                scr, (ix, iy) = frames[0], (self.bez, self.bez)
                sub = reg[iy:iy + scr.shape[0], ix:ix + scr.shape[1]]
                sub[:] = sub * (1 - self.screen_mask) + scr * self.screen_mask
                im, dx, dy = self.island
                isl = sub[dy:dy + im.shape[0], dx:dx + im.shape[1]]
                isl[:] = isl * (1 - im)
                out[y:y + ph, x:x + pw] = (reg + 0.5).astype(np.uint8)
            else:
                out = frames[0].copy()
                ins, (x, y) = frames[1], self.inset_xy
                reg = out[y:y + ins.shape[0], x:x + ins.shape[1]].astype(np.float32)
                out[y:y + ins.shape[0], x:x + ins.shape[1]] = (reg * (1 - self.inset_mask) + ins * self.inset_mask).astype(np.uint8)
            yield out

    def frames(self):
        return getattr(self, f"_frames_{self.kind}")()

    @property
    def camera_captured(self):
        """False for digital content (screen recordings, text slides): no camera motion blur."""
        if self.kind == "layout":
            return all(p.camera_captured for p in self.parts)
        return self.kind in ("video", "still")

    def velocity(self):
        """Per-frame camera velocity (vx, vy) in output pixels / frame, for motion blur in the capture stage."""
        if self.kind in ("video", "still") and not self.static:
            xy = self.P[:, 1:3] / 100 * np.array([self.bw, self.bh])
            return np.vstack([np.zeros((1, 2)), np.diff(xy, axis=0)])
        if self.kind == "layout":
            return self.parts[0].velocity()
        return np.zeros((self.n, 2))
