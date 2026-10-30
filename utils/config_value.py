#!/usr/bin/env python3
"""Print config values the way the pipeline sees them: config/config.yaml with
config/site.yaml (if present) deep-merged over it, as run.sh does.

    python utils/config_value.py results_root splits_root
    python utils/config_value.py padding.results_root

One value per line, in argument order. Used by the utils/*.sbatch wrappers so they
read paths from the same place as the pipeline instead of hard-coding them.
"""
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load() -> dict:
    cfg = yaml.safe_load((REPO / "config" / "config.yaml").read_text()) or {}
    site = REPO / "config" / "site.yaml"
    if site.is_file():
        cfg = _merge(cfg, yaml.safe_load(site.read_text()) or {})
    return cfg


def main() -> int:
    cfg = load()
    for key in sys.argv[1:]:
        v = cfg
        for part in key.split("."):
            if not isinstance(v, dict) or part not in v:
                print(f"config_value: no key {key!r}", file=sys.stderr)
                return 1
            v = v[part]
        print(v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
