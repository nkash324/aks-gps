"""Load gps.yaml (+ optional gps.local.yaml) and draw per-run kart parameters."""
import copy
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _merge(base, over):
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path=None, overrides=None):
    """Return the config dict. Order: gps.yaml < gps.local.yaml < overrides."""
    path = pathlib.Path(path) if path else ROOT / "gps.yaml"
    with open(path, encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)
    local = path.with_name("gps.local.yaml")
    if local.exists():
        with open(local, encoding="utf-8-sig") as f:
            _merge(cfg, yaml.safe_load(f))
    if overrides:
        _merge(cfg, copy.deepcopy(overrides))
    return cfg


def track_path(cfg):
    p = pathlib.Path(cfg["track_world"])
    return p if p.is_absolute() else ROOT / p


def sample_kart(cfg, rng=None):
    """Kart parameters for one run: the nominal values, each scaled by a random factor
    in [1 - f, 1 + f] (f from cfg['randomize']) when rng is given and randomize is on."""
    kart = copy.deepcopy(cfg["kart"])
    rnd = cfg.get("randomize", {})
    if rng is not None and rnd.get("enabled", False):
        for k, f in rnd.items():
            if k != "enabled":
                kart[k] = kart[k] * float(rng.uniform(1.0 - f, 1.0 + f))
    return kart
