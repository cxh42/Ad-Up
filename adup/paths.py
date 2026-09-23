"""Repository-wide paths and network settings, so scripts work regardless of the current directory."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
ADS = DATA / "ads"              # real ads scraped from ad libraries (real-world LQ domain, eval only)
HQ = DATA / "hq"                # high-quality source footage used as GT
PUBLIC = DATA / "public"        # public datasets (HQ-VSR, VideoLQ, ...)
PAIRS = DATA / "pairs"          # synthetic (GT, LQ) pairs made by adup.degrade

OUTPUTS = ROOT / "outputs"
ANALYSIS = OUTPUTS / "analysis"  # metric CSVs, figures, logs
RUNS = OUTPUTS / "runs"          # training runs

CONFIGS = ROOT / "configs"
THIRD_PARTY = ROOT / "third_party"

# All outbound traffic goes through the local proxy; override with ADUP_PROXY="" to disable.
PROXY = os.environ.get("ADUP_PROXY", "http://127.0.0.1:7897")
PROXIES = {"http": PROXY, "https": PROXY} if PROXY else None
