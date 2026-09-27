"""Load a dataset config (configs/pairs/<version>.yaml) with optional command-line overrides.

A config has three sections, one per stage: `gt` (GT gate and plain-clip aspect ratios), `ugc` (director, virtual
camera, burned-in text, splits) and `degrade` (second-order degradation: presets `core` and `tail`).
Overrides use dotted keys with YAML values, e.g. --set gt.max_downup_psnr=99 degrade.tail_prob=0.3.
"""

import yaml

from adup.paths import PAIR_CONFIG


def load_config(path=None, overrides=()):
    cfg = yaml.safe_load(open(path or PAIR_CONFIG))
    for item in overrides:
        key, value = item.split("=", 1)
        *parents, leaf = key.split(".")
        node = cfg
        for p in parents:
            node = node[p]
        if leaf not in node:
            raise KeyError(f"unknown config key {key}")
        node[leaf] = yaml.safe_load(value)
    return cfg


def add_config_args(ap):
    ap.add_argument("--config", help=f"dataset config (default {PAIR_CONFIG.relative_to(PAIR_CONFIG.parents[2])})")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config entries, e.g. gt.max_downup_psnr=99")
