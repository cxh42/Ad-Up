"""Split the collected real ads into calibration / dev / test, once, by a hash of the ad id.

  calibration  tunes the degradation and UGC-ification ranges (adup.analysis.compare against synthetic sets)
  dev          metric development and model selection
  test         held-out real-world evaluation; never used to tune anything

The split of an ad never changes when more ads are collected, because it depends only on its id.

Usage (from the repo root): .venv-iqa/bin/python -m adup.real_ads.splits [--ratios calibration=0.3 dev=0.2 test=0.5]
Output: data/stats/real_ads/splits.csv (source, date, ad_id, video_file, video_sd_file, split)
"""

import argparse
import glob
import os

import pandas as pd

from adup.paths import REAL_ADS, REAL_STATS, ROOT
from adup.ugc.director import split_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratios", nargs="+", default=["calibration=0.3", "dev=0.2", "test=0.5"])
    args = ap.parse_args()
    ratios = {k: float(v) for k, v in (r.split("=") for r in args.ratios)}
    rows = []
    for f in sorted(glob.glob(str(REAL_ADS / "*" / "*" / "summary.csv"))):
        source, date = f.split(os.sep)[-3:-1]
        d = pd.read_csv(f, dtype={"ad_id": str})
        for r in d.itertuples():
            if not isinstance(r.video_file, str):
                continue
            sd = getattr(r, "video_sd_file", None)
            rows.append({"source": source, "date": date, "ad_id": r.ad_id, "video_file": r.video_file,
                         "video_sd_file": sd if isinstance(sd, str) else None, "split": split_of(r.ad_id, ratios)})
    out = pd.DataFrame(rows).drop_duplicates("ad_id")
    out = out[out.video_file.map(lambda p: os.path.exists(ROOT / p))]
    REAL_STATS.mkdir(parents=True, exist_ok=True)
    out.to_csv(REAL_STATS / "splits.csv", index=False)
    print(out.groupby(["source", "split"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
