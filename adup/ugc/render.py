"""Render one shot of a UGC sequence into GT frames of a given box size: video, still, layout or text slide.

Shot kinds (spec dicts from adup.ugc.director, or plain {"src", "start", "frames", "src_fps"} = video):
  video   crop of a source clip (face-aware placement) + virtual phone camera (adup.ugc.camera)
  still   a high-resolution photo animated by the virtual camera (Ken Burns / handheld), e.g. product photos
  layout  a composition of sub-shots, as made in CapCut / TikTok:
            fit_blur  landscape clip inside a portrait frame over a blurred, darkened copy of itself
            split     two clips stacked (before/after, two angles)       duet  two clips side by side
            pip       an inset clip (rounded corners, border) over a full-frame clip
  slide   a solid or gradient card with large text (hooks, CTAs)
  screen  a phone screen recording: a tall app page (adup.sources.ui_screens) scrolled by swipes and pauses;
          offsets are whole pixels, so frames are exact copies of the rendered UI
Nothing is ever upscaled except the blurred background of fit_blur, which carries no detail by design.
"""

import os
import urllib.request

import cv2
import numpy as np

from adup.degrade.overlays import draw_block, pick_font, wrap
from adup.media import max_crop, place_crop, probe, read_frames, stream_frames
from adup.paths import HQ, PROXY
from adup.ugc.camera import plan_camera, warp

STILLS = HQ / "unsplash_lite" / "images"


def even(x):
    return max(int(round(x / 2)) * 2, 2)


def still_path(src, min_width):
    """Local file for a still: a path, or 'unsplash:<id>|<image_url>' fetched once at a width >= min_width."""
    if not src.startswith("unsplash:"):
        return src
    pid, url = src[len("unsplash:"):].split("|", 1)
    dst = STILLS / f"{pid}.jpg"
    if not dst.exists():
        STILLS.mkdir(parents=True, exist_ok=True)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({"https": PROXY, "http": PROXY}))
        with opener.open(f"{url}?w={int(min_width)}&q=95&fm=jpg", timeout=120) as r, open(dst, "wb") as f:
            f.write(r.read())
    return str(dst)


class ShotRenderer:
    """Frames of one shot at bw x bh. `info` describes crop, camera and layout for meta.json."""

    def __init__(self, shot, bw, bh, fps, rng, cfg):
        self.shot, self.bw, self.bh, self.fps, self.rng, self.cfg = shot, bw, bh, fps, rng, cfg
        self.kind = shot.get("kind", "video")
        self.n = shot["frames"]
        self.info = {"kind": self.kind, "box": [bw, bh]}
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
        path = still_path(self.shot["src"], self.bw * 2)
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
        room, r, y, ys, t = img.shape[0] - self.bh, self.rng, 0.0, [], 0
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
        s, r = self.shot, self.rng
        c1 = np.array(s.get("bg", r.integers(0, 256, 3)), np.float32)
        c2 = c1 * r.uniform(0.6, 1.0) if r.random() < 0.5 else c1
        g = np.linspace(0, 1, self.bh, dtype=np.float32)[:, None, None]
        bg = (c1 * (1 - g) + c2 * g).repeat(self.bw, 1).astype(np.uint8)
        size = max(int(self.bh * r.uniform(0.04, 0.07)), 16)
        font, name = pick_font(_PyRng(r), r.choice(["heavy", "sans_bold", "serif"]), size)
        fg = (255, 255, 255) if c1.mean() < 140 else (20, 20, 20)
        block = draw_block(wrap([(w, fg) for w in s.get("text", "wait for it").split()], font, size, self.bw * 0.8),
                           font, size)
        a = np.asarray(block)
        y0, x0 = (self.bh - a.shape[0]) // 2, (self.bw - a.shape[1]) // 2
        alpha = a[..., 3:4] / 255
        region = bg[y0:y0 + a.shape[0], x0:x0 + a.shape[1]].astype(np.float32)
        bg[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = (region * (1 - alpha) + a[..., :3] * alpha).astype(np.uint8)
        self.card = bg
        self.preview = cv2.resize(bg, (even(self.bw / 4), even(self.bh / 4)), interpolation=cv2.INTER_AREA)
        self.info.update({"text": s.get("text"), "font": name})

    def _frames_slide(self):
        for _ in range(self.n):
            yield self.card.copy()

    # ---------------------------------------------------------------- layout
    def _setup_layout(self):
        s, bw, bh, r = self.shot, self.bw, self.bh, self.rng
        lay = s["layout"]
        mk = lambda part, w, h: ShotRenderer({**part, "frames": self.n}, even(w), even(h), self.fps, r, self.cfg)
        if lay == "fit_blur":
            ch = even(bw * 9 / 16)
            self.parts = [mk(s["parts"][0], bw, ch)]
            self.fg_y = int((bh - ch) * r.uniform(0.3, 0.5)) // 2 * 2
            self.darken = float(r.uniform(0.55, 0.85))
        elif lay == "split":
            gap = even(r.choice([0, 0, 4, 8]))
            self.gap, self.gap_col = gap, (255, 255, 255) if r.random() < 0.5 else (0, 0, 0)
            self.parts = [mk(p, bw, (bh - gap) / 2) for p in s["parts"][:2]]
        elif lay == "duet":
            self.parts = [mk(p, bw / 2, bh) for p in s["parts"][:2]]
        elif lay == "pip":
            iw = even(bw * r.uniform(0.3, 0.42))
            ih = even(iw * (16 / 9 if r.random() < 0.6 else 1.0))
            self.parts = [mk(s["parts"][0], bw, bh), mk(s["parts"][1], iw, ih)]
            self.inset_xy = (even(bw * r.uniform(0.04, 0.1)) if r.random() < 0.5 else even(bw * 0.96 - iw),
                             even(bh * r.uniform(0.08, 0.18)) if r.random() < 0.5 else even(bh * 0.72 - ih))
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
        """False for digital content (screen recordings, text slides): no phone look, no camera / ISP degradations."""
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


class _PyRng:
    """random.Random-like view of a numpy Generator, for helpers written against the stdlib API."""

    def __init__(self, g):
        self.g = g

    def choice(self, seq):
        return seq[int(self.g.integers(len(seq)))]

    def random(self):
        return float(self.g.random())
