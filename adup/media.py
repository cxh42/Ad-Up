"""Shared media helpers: ffmpeg frame IO, GT geometry (aspect ratios, crops) and subject-aware crop placement."""

import json
import subprocess
from functools import lru_cache

import cv2
import numpy as np

from adup.paths import FACE_MODEL

# height / width of each GT aspect ratio
ASPECTS = {"9:16": 16 / 9, "4:5": 5 / 4, "1:1": 1.0, "16:9": 9 / 16}
LOSSLESS_H264 = ["-c:v", "libx264", "-preset", "veryfast", "-qp", "0"]
GT_H264 = ["-c:v", "libx264", "-preset", "medium", "-crf", "10"]


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_frames", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(num) / float(den), int(s.get("nb_frames") or 0)


def read_frames(path, vf, w, h, max_frames):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", vf, "-frames:v", str(max_frames), "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def stream_frames(path, vf, w, h, n):
    """Yield exactly n frames (the last one is repeated if the source runs short)."""
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-vf", vf, "-frames:v", str(n), "-fps_mode", "passthrough",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=w * h * 3)
    size, count, last = w * h * 3, 0, None
    while count < n:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        last = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
        count += 1
        yield last
    p.stdout.close()
    p.wait()
    while count < n and last is not None:
        count += 1
        yield last.copy()


class Writer:
    """Pipe RGB frames into an ffmpeg encoder."""

    def __init__(self, path, w, h, fps, codec_args):
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", *codec_args, "-pix_fmt", "yuv420p",
                                   "-fps_mode", "passthrough", path], stdin=subprocess.PIPE)
        self.path = path

    def write(self, frame):
        self.p.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait():
            raise RuntimeError(f"ffmpeg failed writing {self.path}")


def transcode(src, dst, vf, codec_args):
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", vf, *codec_args,
           "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-an", dst]
    subprocess.run(cmd, check=True)


def split_at(src, pattern, cuts):
    """Split src at frame indices `cuts` into pattern % i files, frame-exact and lossless, in one pass."""
    if not cuts:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, *LOSSLESS_H264, "-pix_fmt", "yuv420p",
                        "-fps_mode", "passthrough", "-an", pattern % 0], check=True)
        return
    keys = "+".join(f"eq(n,{c})" for c in cuts)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, *LOSSLESS_H264, "-pix_fmt", "yuv420p",
                    "-force_key_frames", f"expr:{keys}", "-fps_mode", "passthrough", "-an", "-f", "segment",
                    "-segment_frames", ",".join(map(str, cuts)), "-reset_timestamps", "1", pattern], check=True)


def gt_geometry(aspect, gt_short, scale):
    """(gt_w, gt_h, lq_w, lq_h) with the given short side; LQ sides are even and GT = LQ * scale."""
    r = ASPECTS[aspect]
    lq_short = int(round(gt_short / scale)) // 2 * 2
    lq_w, lq_h = (lq_short, int(round(lq_short * r)) // 2 * 2) if r >= 1 else (int(round(lq_short / r)) // 2 * 2, lq_short)
    return int(round(lq_w * scale)) // 2 * 2, int(round(lq_h * scale)) // 2 * 2, lq_w, lq_h


def feasible_aspects(sources, gt_short, scale, weights):
    """Aspects whose GT can be cut from every source (sw, sh) without upscaling."""
    ok = {}
    for a, wgt in weights.items():
        gw, gh, _, _ = gt_geometry(a, gt_short, scale)
        if all(max_crop(sw, sh, gw, gh)[0] >= gw for sw, sh in sources):
            ok[a] = wgt
    return ok


def max_crop(sw, sh, gw, gh):
    """Largest crop of the gw:gh aspect ratio that fits in an sw x sh source."""
    return (int(sh * gw / gh) // 2 * 2, sh) if sw / sh > gw / gh else (sw, int(sw * gh / gw) // 2 * 2)


def detail(gray):
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


@lru_cache(maxsize=1)
def face_detector():
    return cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), 0.75)


def largest_face(rgb):
    """(cx, cy, area_fraction) of the largest face in relative coordinates, or None."""
    h, w = rgb.shape[:2]
    s = 640 / max(h, w)
    img = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (int(w * s), int(h * s)))
    det = face_detector()
    det.setInputSize((img.shape[1], img.shape[0]))
    _, f = det.detect(img)
    if f is None:
        return None
    i = int((f[:, 2] * f[:, 3]).argmax())
    x, y, fw, fh = f[i, :4]
    return (x + fw / 2) / img.shape[1], (y + fh / 2) / img.shape[0], fw * fh / (img.shape[0] * img.shape[1])


def place_crop(frame, cw, ch, bw, bh, rng, candidates=7):
    """Top-left (x, y) of a cw x ch crop in a source frame: put the largest face on the horizontal centre line and
    in the upper third (selfie framing); without a face, use the position with the most texture at the output size."""
    sh, sw = frame.shape[:2]
    face = largest_face(frame)
    if face is not None:
        fx, fy = face[0] * sw, face[1] * sh
        x = int(np.clip(fx - cw / 2, 0, sw - cw)) // 2 * 2
        y = int(np.clip(fy - ch * 0.38, 0, sh - ch)) // 2 * 2
        return int(x), int(y), "face"
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if cw < sw:
        y = (sh - ch) // 2 // 2 * 2
        xs = sorted({int(v) // 2 * 2 for v in np.linspace(0, sw - cw, candidates)})
        best = max((detail(cv2.resize(gray[y:y + ch, x:x + cw], (bw, bh), interpolation=cv2.INTER_AREA)), x) for x in xs)[1]
        return int(min(max(best + rng.integers(-cw // 20, cw // 20 + 1) // 2 * 2, 0), sw - cw)), int(y), "detail"
    x = 0
    ys = sorted({int(v) // 2 * 2 for v in np.linspace(0, sh - ch, candidates)})
    y = max((detail(cv2.resize(gray[yy:yy + ch, :cw], (bw, bh), interpolation=cv2.INTER_AREA)), yy) for yy in ys)[1]
    return int(x), int(y), "detail"
