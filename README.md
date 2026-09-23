# Ad-Up

Ad-Up does real-world video super-resolution for UGC ads: it upscales 576p–720p ads to 2K/4K. The repo covers three
parts:
- **Data collection:** real ads, used as the target LQ domain.
- **Paired training data:** HQ footage, plus synthetic LQ calibrated to match real ads.
- **Model training:** DOVE fine-tuning.

## Layout

```
adup/                     project code (a Python package); run from the repo root with `python -m adup.<module>`
  paths.py                all repo paths + the network proxy (ADUP_PROXY, default http://127.0.0.1:7897)
  collect/                real ads = real-world LQ domain (eval only)
    tiktok_topads.py      TikTok Creative Center Top Ads -> data/ads/tiktok_topads/<date>/
    meta_adlib.py         Meta Ad Library via AdDownloader (needs META_TOKEN) -> data/ads/meta_adlib/
    contact_sheets.py     per-industry frame contact sheets of a scraped set
  sources/                HQ footage used as GT
    ultravideo.py         human-centric 4K subset of UltraVideo, fetched clip-by-clip from the remote zips
  degrade/
    pipeline.py           HQ -> (GT, LQ) pairs: portrait crop, captions, capture/ISP, edit export, platform
                          transcode, re-upload; DEFAULT_CONFIG = calibrated v3
  analysis/
    quality.py            DOVER / CLIP-IQA / MUSIQ / bitrate / effective resolution
    degradation_stats.py  blockiness, noise, sharpening overshoot, GOP, duplicate frames, black bars
    compare.py            real-vs-synthetic distribution distance (calibration report)
configs/degradation/      earlier degradation configs (v1); pass one with --config
training/dove/            our DOVE training scripts/configs (upstream code stays in third_party/DOVE)
third_party/              upstream repos as git submodules, unmodified: DOVE (training/inference), DOVER (video
                          quality metric); weights live inside them untracked (DOVE/pretrained_models/,
                          DOVER/pretrained_weights/DOVER.pth)
docs/                     research notes (docs/ugc_degradations.md)
requirements/             collect.txt (.venv), ml.txt (.venv-iqa)
data/        (gitignored) ads/  hq/  public/{HQ-VSR,VideoLQ,_archives}  pairs/{<set>,calibration/}
outputs/     (gitignored) analysis/ (metric CSVs, figures, logs)  runs/ (training)
```

## Setup

```bash
git clone --recurse-submodules <this repo>      # or, in an existing clone: git submodule update --init
```

## Environments

| venv | Python | Used for |
|---|---|---|
| `.venv` | 3.11 | Ad collection. AdDownloader needs Python < 3.12. See `requirements/collect.txt`. |
| `.venv-iqa` | 3.11 | Everything with torch: HQ sourcing, degradation, metrics, and later training. See `requirements/ml.txt`. |

## Pipeline

```bash
# 1. real ads (target LQ domain)
.venv/bin/python -m adup.collect.tiktok_topads
.venv/bin/python -m adup.collect.contact_sheets data/ads/tiktok_topads/<date>

# 2. HQ sources
.venv-iqa/bin/python -m adup.sources.ultravideo 25          # clips per category

# 3. synthetic pairs (GT 1080x1920, LQ 540x960, 2 LQ variants per clip)
.venv-iqa/bin/python -m adup.degrade.pipeline --manifest data/hq/ultravideo/manifest.csv \
    --out data/pairs/uv_p1080_x2 --gt-size 1080x1920 --scale 2 --variants 2

# 4. calibration: score a synthetic set and the real ads, then compare the distributions
.venv-iqa/bin/python -m adup.analysis.degradation_stats <group> outputs/analysis/degradation.csv <videos...>
.venv-iqa/bin/python -m adup.analysis.quality           <group> outputs/analysis/quality.csv     <videos...>
.venv-iqa/bin/python -m adup.analysis.compare tiktok_720p <group>
```

## Data sources and licenses

| Data | Location | License / terms |
|---|---|---|
| TikTok Top Ads | `data/ads/tiktok_topads/` | Advertisers' creatives. Use for analysis and evaluation only. |
| UltraVideo | `data/hq/ultravideo/` | CC-BY-4.0 + **non-commercial research only** (sources are from YouTube). |
| HQ-VSR (DOVE) | `data/public/HQ-VSR/` | Derived from OpenVid-1M. Research use. |
| VideoLQ | `data/public/VideoLQ/` | Research benchmark. |

Pexels, Pixabay and Mixkit forbid scripted or bulk downloading. Pexels and Pixabay also name ML use explicitly.
Commercial training needs self-shot, creator-licensed, or purchased footage.
