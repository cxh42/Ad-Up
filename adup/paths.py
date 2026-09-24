"""Every directory and file location the code uses, in one place, plus network settings.

The layout follows the pipeline (see README.md and data/README.md):

  data/real_ads    real ads (TikTok / Meta): the real-world LQ domain; measured, never trained on
  data/hq          >= 2K GT sources; each clip directory holds its manifest.csv and gate_<gt_short>.csv
  data/benchmarks  public datasets (KwaiVIR, VideoLQ, HQ-VSR)
  data/stats       measurement tables the pair pipeline reads (real-ad targets, HQ-pool content tags)
  data/pairs       generated (GT, LQ) datasets, one directory per dataset
  data/assets      fonts and small models
  outputs/         calibration reports, model runs, figures, logs
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- data (not in git)
DATA = ROOT / "data"
REAL_ADS = DATA / "real_ads"          # <scraper>/<date>/{videos/, summary.csv, sheets/, shots/}
HQ = DATA / "hq"                      # ultravideo/{4k,8k}/  unsplash_lite/  ui_screens/
BENCHMARKS = DATA / "benchmarks"      # KwaiVIR/  VideoLQ/  HQ-VSR/
PAIRS = DATA / "pairs"                # <dataset>/{specs.jsonl, config.yaml, <ad id>/{gt.mp4, lq_<k>.mp4, meta.json, shots/}}
ASSETS = DATA / "assets"
FONTS = ASSETS / "fonts"              # OFL caption fonts + Noto Color Emoji (adup.ugc.fonts)
FACE_MODEL = ASSETS / "models" / "face_detection_yunet_2023mar.onnx"

STATS = DATA / "stats"
REAL_STATS = STATS / "real_ads"       # measured on real ads (adup.analysis.*): the targets synthetic data is matched to
HQ_STATS = STATS / "hq"               # content tags of the HQ pool (adup.analysis.ugc_content)
BENCHMARK_STATS = STATS / "benchmarks"
# tables read while making pairs
LOOK_TABLE = REAL_STATS / "ugc_look.csv"             # per-shot shake / colour / faces of real ads -> camera and look targets
SHOTS_TABLE = REAL_STATS / "shots.csv"               # shot lengths of real ads -> the director's cut rhythm
COVERAGE_TABLE = HQ_STATS / "content_coverage.csv"   # theme shares of real ads vs HQ pool -> theme-balanced sampling
POOL_TABLE = HQ_STATS / "content_pool.csv"           # UltraVideo catalogue themes (from text descriptions)
CLIPS_TABLE = HQ_STATS / "content_clips.csv"         # downloaded clips' themes (CLIP on the clip itself)
STILLS_TABLE = HQ_STATS / "content_unsplash.csv"     # Unsplash Lite photos with themes -> stills and UI-screen pictures

# ---------------------------------------------------------------- outputs (not in git)
OUTPUTS = ROOT / "outputs"
CALIBRATION = OUTPUTS / "calibration"  # metric tables of synthetic sets (compared with data/stats) and reports
RUNS = OUTPUTS / "runs"                # model inference / training outputs
FIGURES = OUTPUTS / "figures"
LOGS = OUTPUTS / "logs"

# ---------------------------------------------------------------- in git
CONFIGS = ROOT / "configs"
PAIR_CONFIG = CONFIGS / "pairs" / "v6.yaml"          # current dataset config (GT gate, UGC-ification, degradation)
THIRD_PARTY = ROOT / "third_party"

# All outbound traffic goes through the local proxy; override with ADUP_PROXY="" to disable.
PROXY = os.environ.get("ADUP_PROXY", "http://127.0.0.1:7897")
PROXIES = {"http": PROXY, "https": PROXY} if PROXY else None
# The shell may export ALL_PROXY=socks://...; httpx (huggingface_hub, open_clip) rejects that scheme. Use the HTTP proxy.
for _k in ("ALL_PROXY", "all_proxy"):
    if os.environ.get(_k, "").startswith("socks") and PROXY:
        os.environ[_k] = PROXY
