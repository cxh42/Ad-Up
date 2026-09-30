"""Every directory and file location the code uses, in one place, plus network settings.

The layout follows the pipeline (see README.md and data/README.md):

  data/sources     GT sources (sharp at 1080p): UltraVideo, KwaiVIR's HQ clips, photos, UI screens; each clip
                   directory holds its manifest.csv, gate_<gt_short>.csv and source_text.csv
  data/real_ads    real ads (TikTok / Meta): the real-world LQ domain; measured and evaluated on, never trained on
  data/pairs       generated (GT, LQ) datasets, one directory per dataset (v7_train, v7_dev, ..., examples/)
  data/stats       measurement tables the pair pipeline reads (real-ad targets, source content tags)
  data/eval_sets   public evaluation sets, evaluated on only (KwaiVIR's LQ videos, VideoLQ)
  data/assets      fonts and small models
  outputs/         calibration reports, model runs, figures, logs
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- data (not in git)
DATA = ROOT / "data"
REAL_ADS = DATA / "real_ads"          # <scraper>/<date>/{videos/, summary.csv, shots/}
SOURCES = DATA / "sources"            # ultravideo/{4k,8k}/  kwaivir/clips/  unsplash_lite/  ui_screens/
EVAL_SETS = DATA / "eval_sets"        # KwaiVIR/{wild, val_input, test_data, shots}  VideoLQ/
PAIRS = DATA / "pairs"                # <dataset>/{specs.jsonl, config.yaml, <ad id>/{gt.mp4, lq_<k>.mp4, meta.json, shots/}}
ASSETS = DATA / "assets"
FONTS = ASSETS / "fonts"              # OFL caption fonts + Noto Color Emoji (adup.ugc.fonts)
FACE_MODEL = ASSETS / "models" / "face_detection_yunet_2023mar.onnx"

STATS = DATA / "stats"
REAL_STATS = STATS / "real_ads"       # measured on real ads (adup.analysis.*): the targets synthetic data is matched to
SOURCE_STATS = STATS / "sources"      # content tags of the source pool (adup.analysis.ugc_content)
EVAL_STATS = STATS / "eval_sets"      # degradation / quality of public sets and of HQ-VSR (DOVE's training set), for reference
# tables read while making pairs
LOOK_TABLE = REAL_STATS / "ugc_look.csv"             # per-shot shake / colour / faces of real ads -> camera shake targets
SHOTS_TABLE = REAL_STATS / "shots.csv"               # shot lengths of real ads -> the director's cut rhythm
COVERAGE_TABLE = SOURCE_STATS / "content_coverage.csv"   # theme shares of real ads vs HQ pool -> theme-balanced sampling
POOL_TABLE = SOURCE_STATS / "content_pool.csv"           # UltraVideo catalogue themes (from text descriptions)
CLIPS_TABLE = SOURCE_STATS / "content_clips.csv"         # downloaded clips' themes (CLIP on the clip itself)
STILLS_TABLE = SOURCE_STATS / "content_unsplash.csv"     # Unsplash Lite photos with themes -> stills and UI-screen pictures

# ---------------------------------------------------------------- outputs (not in git)
OUTPUTS = ROOT / "outputs"
CALIBRATION = OUTPUTS / "calibration"  # metric tables of synthetic sets (compared with data/stats) and reports
RUNS = OUTPUTS / "runs"                # model inference / training outputs
FIGURES = OUTPUTS / "figures"
LOGS = OUTPUTS / "logs"

# ---------------------------------------------------------------- in git
CONFIGS = ROOT / "configs"
PAIR_CONFIG = CONFIGS / "pairs" / "v7.yaml"          # current dataset config (GT gate, UGC-ification, degradation)
THIRD_PARTY = ROOT / "third_party"

# All outbound traffic goes through the local proxy; override with ADUP_PROXY="" to disable.
PROXY = os.environ.get("ADUP_PROXY", "http://127.0.0.1:7897")
PROXIES = {"http": PROXY, "https": PROXY} if PROXY else None
# The shell may export ALL_PROXY=socks://...; httpx (huggingface_hub, open_clip) rejects that scheme. Use the HTTP proxy.
for _k in ("ALL_PROXY", "all_proxy"):
    if os.environ.get(_k, "").startswith("socks") and PROXY:
        os.environ[_k] = PROXY
