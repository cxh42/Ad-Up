# Real-world degradations in UGC ad videos

This note maps the degradations that UGC ads pick up between the phone and the viewer to
`adup/degrade/pipeline.py`. It is the basis for synthesizing (GT, LQ) training pairs.

Evidence tags used below:
- **measured** — measured on our 46 scraped TikTok Top Ads (`outputs/analysis/`).
- **lit** — from the literature listed at the bottom.
- **obs** — seen when browsing the ads by eye.

## 1. What we measured on real TikTok ads (2026-09)

| Property | Finding |
|---|---|
| Encoder | 23/55 carry an SEI `bvc0ot v2.2.1.3-20250220` (ByteDance BVC, which reuses the x264 SEI UUID). 2/55 carry `x264 core 148, crf=24, vbv_maxrate=4000`. 30/55 have no SEI. |
| Bitstream | H.264 High profile, level 3.1, 4–5 reference frames, B-frames with reorder depth 2, CABAC, 8x8 transform. Limited (tv) range, BT.709. |
| GOP | Median keyframe interval ≈ 110 frames. About 71% of frames are B-frames. |
| Resolution / bitrate | 720x1280 or 576x1024 (one clip at 360x640). Bitrate 0.3–2.3 Mbps, median 1.3 Mbps. bpp ≈ 0.06. |
| "1080p" versions | These are upscales. After field-of-view matching they look the same as the 720p version, MUSIQ drops from 65 to 58, and the down-up PSNR rises. |
| Effective resolution | A 1.5x down-up round trip keeps a median 38 dB PSNR (HQ-VSR: 35.7 dB), so the real detail is about 480p inside a 720p frame. |
| Noise | Lower than HQ-VSR (σ 0.34 vs 0.42). The footage is smoothed (denoised or beautified, then encoded), not noisy. |
| Blocking | Slightly above HQ-VSR (1.037 vs 1.021 boundary/interior gradient ratio). Mild. |
| Shots | Median 50 frames per shot and 8.5 shots per ad. Shots of ≥35 frames cover 91% of all frames. |
| Content | About 70% UGC style, 100% 9:16. Burned-in captions on almost every ad. Stickers, price tags, split screens, screen recordings. |

## 2. Degradation taxonomy by stage

### 2.1 Capture (phone camera and ISP)
- **Sensor noise in low light.** Luma and chroma noise, which varies over time. [lit, obs]
- **ISP denoise.** Produces waxy, over-smoothed textures. [lit, measured: low noise σ]
- **Beauty filters.** Skin smoothing, face reshaping. [obs]
- **ISP sharpening.** Unsharp-mask halos around edges. [lit]
- **Blur.** Defocus from phone portrait mode, motion blur from handheld shooting and fast gestures, rolling shutter. [obs]
- **Digital zoom.** An upscaled crop. The front camera is also often lower quality than the rear camera. [obs]
- **Other artifacts.** Stabilization warping, auto-exposure and white-balance flicker, variable frame rate. [lit]

### 2.2 Editing app (CapCut / InShot / in-app editor)
- **Export encode.** One extra compression generation. [lit]
- **Mixed sources in one timeline.** Stock clips, screen recordings, green-screen composites with keying edges, re-used downloads that are already compressed and sometimes watermarked. [obs]
- **Overlays.** Burned-in captions, stickers, emoji, price tags, picture-in-picture. [obs, measured]
- **Aspect-ratio fill.** Landscape clips placed into 9:16 with letterboxing or a blurred background. [obs]
- **Speed ramps.** Frame blending, duplicated frames, or interpolated frames. [obs]
- **Filters and LUTs.** Color grading. [obs]

### 2.3 Platform ingest and transcode
- **Downscale to the ladder.** 540, 576, or 720 short side. [measured]
- **Low-bitrate H.264.** BVC or x264 at about 0.3–2.3 Mbps. [measured]
- **Coding artifacts.** Deblocking smoothness, blocking in flat areas and fast motion, and banding in gradients such as skies and studio backdrops. [lit, measured]
- **4:2:0 chroma subsampling.** Causes color bleeding on saturated text and sticker edges. [lit]
- **"Fake HD".** The platform upscales to 1080p. [measured]
- **Platform enhancement.** Pre-processing such as sharpening or denoising, and enhancement workflows. KVQ lists pre-processing, transcoding, and enhancement as the practical short-video workflows. [lit]

### 2.4 Redistribution
- **Re-upload generations.** Download, re-edit, re-upload, which adds another rescale and another encode. [lit, obs]
- **Screen recordings of other apps.** Scaling, UI chrome, moiré. [obs]

## 3. Coverage in `adup/degrade/pipeline.py`

| Stage | Implemented | Not yet |
|---|---|---|
| GT | Portrait crop placed on the most detailed region. GT detail and luma gate. Burned-in captions and stickers on the GT. | IQA gating (CLIP-IQA / DOVER), motion-area crops |
| Capture | Gaussian defocus blur. Luma and chroma noise. Bilateral or skin-masked smoothing. Unsharp-mask sharpening. | Motion blur, rolling shutter, digital-zoom upscale, exposure flicker |
| Edit | x264 export (CRF 16–22, preset fast) | Letterbox or blurred-background fill, screen-recording look, speed-ramp frame blending, green-screen composites |
| Platform | Random-kernel downscale to GT/scale. x264 High profile with CRF 20–28, maxrate 1.2–4 Mbps, keyint 60–250, 3 B-frames, 4 refs (v3). | Platform sharpening or denoise pre-filter. An HEVC or BVC-like encoder mix. |
| Re-upload | 20% chance: upscale 1–2x, re-encode, downscale, re-encode | Watermarks |

Degradations excluded on purpose:
- **Color and LUT changes.** The model should keep the advertiser's grade, not undo it.
- **Beauty-filter reversal.** Whether the model should restore skin texture that was smoothed on purpose is a product decision. Smoothing is only applied to the LQ with a moderate probability.

## 4. Calibration against real ads

The synthetic LQ is compared with the 46 real TikTok ads at the same size (GT 1080x1920, scale 1.5, LQ 720x1280) on
19 UltraVideo clips. Each cell is W/IQR: the 1-D Wasserstein distance divided by the real ads' interquartile range.
Lower is closer; 0.5 or more counts as off. Script: `adup/analysis/compare.py`.

| Metric | v1 (first guess) | v2 | v3 (default) | clean GT, no degradation |
|---|---|---|---|---|
| DOVER | 0.32 | 0.17 | **0.15** | 0.36 |
| DOVER technical | 0.40 | 0.21 | **0.13** | 0.26 |
| noise σ | 0.42 | 0.36 | **0.29** | 0.31 |
| sharpening overshoot | 0.68 | 0.46 | **0.37** | 0.51 |
| blockiness | 0.49 | 0.53 | **0.39** | 1.72 |
| bitrate | 0.41 | 0.48 | 0.56 | 10.65 |
| down-up PSNR ×1.5 | 1.01 | 0.74 | 0.55 | 0.45 |
| MUSIQ | 1.51 | 1.09 | 0.82 | 0.42 |
| CLIP-IQA | 0.96 | 0.73 | 0.57 | 0.25 |

v3 matches the degradation-sensitive metrics. The residual gap in MUSIQ, CLIP-IQA and effective resolution is about
the same as the gap of the undegraded GT itself (median MUSIQ: clean GT 62.2, v3 61.4, real ads 66.9). The cause is
the source domain: cinematic, shallow depth-of-field YouTube 4K footage is softer than ISP-sharpened phone footage.
Closing that gap needs phone-shot HQ sources, not stronger degradation.

## 5. References

The TikTok measurements come from the scripts in `adup/analysis/`.

- Real-ESRGAN, high-order degradation model: https://arxiv.org/abs/2107.10833
- RealBasicVSR / VideoLQ, video degradations with codecs; DOVE uses this pipeline: https://arxiv.org/abs/2111.12704
- VCISR, synthetic data built with real video codecs: https://arxiv.org/abs/2311.00996
- KVQ, Kwai short-form video quality with pre-processing, transcoding, and enhancement workflows (CVPR 2024): https://openaccess.thecvf.com/content/CVPR2024/html/Lu_KVQ_Kwai_Video_Quality_Assessment_for_Short-form_Videos_CVPR_2024_paper.html
- KwaiSR, short-form UGC super-resolution with synthetic and wild data (NTIRE 2025): https://arxiv.org/abs/2504.15003
- KwaiVIR, short-form UGC video restoration (NTIRE 2026): https://arxiv.org/abs/2604.10551
- YouTube UGC dataset for compression research: https://arxiv.org/abs/1904.06457
