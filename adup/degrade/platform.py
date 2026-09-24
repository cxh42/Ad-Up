"""Degradation stages 2-4: editing-app export, platform transcode and re-upload.

  2. edit:      export from an editing app (x264, moderate CRF); done by make_pairs while writing the frames
  3. platform:  resize to LQ size -> optional pre-processing (sharpen / denoise) -> H.264 (TikTok / Meta-like x264
                High profile), VP9 or AV1 (YouTube-like) or HEVC
  4. re-upload: optional extra generation (fake-HD upscale -> re-encode with any codec -> back to LQ size)
VP9 / AV1 LQ is decoded and stored as lossless H.264, because the pip builds of OpenCV and decord cannot decode AV1;
the pixels are exactly the decoded ones.
"""

import os

from adup.media import LOSSLESS_H264, transcode

os.environ.setdefault("SVT_LOG", "1")   # SVT-AV1 prints its config banner to stderr unless told otherwise


def pick_weighted(rng, weights):
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def platform_codec(codec, crf, maxrate_k, keyint):
    """Encoder settings per platform family. H.264 mirrors the BVC / x264 streams measured on TikTok and Meta; VP9 and
    AV1 CRF ranges were chosen to reproduce the bits-per-pixel of YouTube Shorts' 720p / 1080p rungs."""
    if codec == "h264":
        return ["-c:v", "libx264", "-preset", "medium", "-profile:v", "high", "-crf", f"{crf}",
                "-maxrate", f"{maxrate_k}k", "-bufsize", f"{2 * maxrate_k}k", "-g", f"{keyint}", "-bf", "3", "-refs", "4"]
    if codec == "hevc":
        return ["-c:v", "libx265", "-preset", "medium", "-crf", f"{crf}", "-tag:v", "hvc1", "-x265-params",
                f"log-level=error:keyint={keyint}:vbv-maxrate={maxrate_k}:vbv-bufsize={2 * maxrate_k}"]
    if codec == "vp9":
        return ["-c:v", "libvpx-vp9", "-crf", f"{crf}", "-b:v", "0", "-deadline", "good", "-cpu-used", "4",
                "-row-mt", "1", "-g", f"{keyint}"]
    if codec == "av1":
        return ["-c:v", "libsvtav1", "-preset", "8", "-crf", f"{crf}", "-g", f"{keyint}", "-svtav1-params", "tune=0"]
    raise ValueError(codec)


def codec_crf(rng, c, codec):
    return rng.randint(*c["platform_crf" if codec == "h264" else f"{codec}_crf"])


def sample_chain(rng, c):
    codec, reupload_codec = pick_weighted(rng, c["platform_codecs"]), pick_weighted(rng, c["platform_codecs"])
    return {
        "edit_export": rng.random() < c["edit_prob"],
        "edit_crf": rng.randint(*c["edit_crf"]),
        "resize_flags": rng.choice(c["resize_flags"]),
        "platform_codec": codec,
        "platform_crf": codec_crf(rng, c, codec),
        "platform_maxrate_k": rng.choice(c["platform_maxrate_k"]),
        "keyint": rng.choice(c["keyint"]),
        "reupload": rng.random() < c["reupload_prob"],
        "reupload_up": rng.choice(c["reupload_up"]),
        "platform_pre": pick_weighted(rng, c["platform_pre"]),
        "pre_amount": rng.uniform(*c["pre_sharpen"]),
        "pre_denoise": rng.uniform(*c["pre_denoise"]),
        "reupload_codec": reupload_codec,
        "reupload_crf": rng.randint(*c["reupload_crf"]) if reupload_codec == "h264" else codec_crf(rng, c, reupload_codec),
    }


def edit_codec(chain):
    """Encoder args of the editing-app export (or a near-lossless intermediate when there is no export)."""
    return ["-c:v", "libx264", "-preset", "fast", "-crf", str(chain["edit_crf"] if chain["edit_export"] else 8)]


def finish_variant(edit_path, out_path, chain, lq_w, lq_h, seconds, tmpdir):
    """Platform transcode (+ optional re-upload) of the edit export -> LQ file."""
    platform = os.path.join(tmpdir, "platform.mp4")
    rate = (chain["platform_maxrate_k"], chain["keyint"])
    vf = f"scale={lq_w}:{lq_h}:flags={chain['resize_flags']}"
    pre = chain.get("platform_pre", "none")
    if pre == "sharpen":
        vf += f",unsharp=5:5:{chain['pre_amount']:.2f}:5:5:0"
    elif pre == "denoise":
        d = chain["pre_denoise"]
        vf += f",hqdn3d={d:.2f}:{d * 0.75:.2f}:{d * 1.5:.2f}:{d * 1.1:.2f}"
    transcode(edit_path, platform, vf, platform_codec(chain["platform_codec"], chain["platform_crf"], *rate))
    platform_kbps = os.path.getsize(platform) * 8 / 1000 / seconds
    final, final_codec = platform, chain["platform_codec"]
    if chain["reupload"]:
        up_w, up_h = int(lq_w * chain["reupload_up"]) // 2 * 2, int(lq_h * chain["reupload_up"]) // 2 * 2
        mid = os.path.join(tmpdir, "reupload_mid.mp4")
        transcode(platform, mid, f"scale={up_w}:{up_h}:flags=bicubic",
                  platform_codec(chain["reupload_codec"], chain["reupload_crf"], *rate))
        final = os.path.join(tmpdir, "reupload.mp4")
        transcode(mid, final, f"scale={lq_w}:{lq_h}:flags={chain['resize_flags']}",
                  platform_codec(chain["reupload_codec"], chain["reupload_crf"], *rate))
        final_codec = chain["reupload_codec"]
    stored_as = final_codec
    if final_codec in ("vp9", "av1"):
        decoded = os.path.join(tmpdir, "decoded.mp4")
        transcode(final, decoded, "null", LOSSLESS_H264)
        final, stored_as = decoded, "h264_lossless"
    os.replace(final, out_path)
    return {"stored_as": stored_as, "platform_kbps": round(platform_kbps, 1)}
