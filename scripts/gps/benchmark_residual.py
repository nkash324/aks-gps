#!/usr/bin/env python3
"""Benchmark residual-driver checkpoints (driver C = driver B + the network's corrections)
exactly like scripts/gps/benchmark.py benchmarks drivers A and B: seeds 0..--laps-1 at every
noise level, the same karts and sensor errors, the same per-lap stats, and a recorded lap of
seed 0 for the maps. No exploration noise.

  python scripts/gps/benchmark_residual.py runs/gps/residual/<run>/ckpt/1003525
  python scripts/gps/benchmark_residual.py runs/gps/residual/<run>/ckpt/*  --out runs/gps/benchmark_c.json

Writes {"summary", "runs", "drivers"} in benchmark.json's format; each checkpoint is the
driver "C <steps>" (e.g. "C 250k"). results_page.py --extra <file> puts it on the page.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import pathlib
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from benchmark import LEVELS, summarise  # noqa: E402
from kart_gps.config import ROOT, load_config, track_path  # noqa: E402

_W = {}


def label(ckpt):
    """'C 250k' from .../ckpt/250880 (nearest 50k), else 'C <folder name>'."""
    name = pathlib.Path(ckpt).name
    if name.isdigit():
        k = round(int(name) / 50_000) * 50
        return f"C {k / 1000:g}M" if k >= 1000 else f"C {k}k"
    return f"C {name}"


def _init(overrides):
    import torch
    from kart_gps.drivers import build_line
    from kart_gps.track import TrackMap
    torch.set_num_threads(1)
    cfg = load_config(overrides=overrides)
    track = TrackMap(track_path(cfg))
    _W.update(cfg=cfg, track=track, line=build_line("optimised", track, cfg), models={}, envs={})


def _policy(ckpt):
    if ckpt not in _W["models"]:
        from stable_baselines3 import PPO
        model = PPO.load(pathlib.Path(ckpt) / "model.zip", device="cpu",
                         custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0})
        with open(pathlib.Path(ckpt) / "vecnormalize.pkl", "rb") as f:
            norm = pickle.load(f)
        _W["models"][ckpt] = lambda obs: model.predict(norm.normalize_obs(obs[None]), deterministic=True)[0][0]
    return _W["models"][ckpt]


def run_lap(job):
    from kart_gps.rl_env import ResidualEnv
    ckpt, level, seed, record = job
    if level not in _W["envs"]:
        _W["envs"][level] = ResidualEnv(_W["cfg"], track=_W["track"], line=_W["line"], level=level, stress=False)
    env = _W["envs"][level]
    pol = _policy(ckpt)
    obs, _ = env.reset(seed=0, options={"lap_seed": seed})
    lap = env.env                                   # the LapEnv underneath (true kart state, trace)
    lap.record = record
    pos_err, acts, jumps, prev = [], [], [], np.zeros(2)
    while True:
        a = np.clip(pol(obs), -1.0, 1.0)
        obs, _, done, _, info = env.step(a)
        acts.append(float(np.abs(a).mean()))
        jumps.append(float(np.abs(a - prev).mean()))
        prev = a
        rx, ry = lap.kart.rear_axle
        pos_err.append(math.hypot(env.est["x"] - rx, env.est["y"] - ry))
        if done:
            break
    out = {"driver": label(ckpt), "ckpt": str(ckpt), "level": level, "seed": seed, "pp": None,
           "end": info["end"], "lap_time": info["lap_time"], "progress": info["progress"],
           "min_clearance": info["min_clearance"], "max_lat_g": lap.max_lat_g,
           "pos_err_p95": float(np.percentile(pos_err, 95)), "mean_abs_action": float(np.mean(acts)),
           "mean_abs_jump": float(np.mean(jumps)),
           "mu": lap.kart_params["mu"], "map_offset": float(np.hypot(*lap.sensors.shift))}
    if record:
        out["trace"] = [[round(v, 3) for v in row] for row in lap.trace[::2]]
    lap.record = False
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", help="checkpoint folders (model.zip + vecnormalize.pkl)")
    ap.add_argument("--laps", type=int, default=30)
    ap.add_argument("--levels", nargs="+", default=list(LEVELS))
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--map-shift", type=float, default=None, help="as benchmark.py --map-shift")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ckpts = [str(pathlib.Path(c).resolve()) for c in args.ckpts
             if (pathlib.Path(c) / "model.zip").exists()]
    overrides = None
    if args.map_shift is not None:
        overrides = {"sensors": {"map_offset": {"shift": args.map_shift}}}

    # seed 0 recorded for the maps
    jobs = [(c, lv, s, s == 0) for c in ckpts for lv in args.levels for s in range(args.laps)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.procs, initializer=_init, initargs=(overrides,)) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(run_lap, jobs), 1):
            rows.append(r)
            if i % 30 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} laps, {time.time() - t0:.0f} s", flush=True)

    summary = summarise(rows)
    for k, v in summary.items():
        g = [r for r in rows if f"{r['driver']}/{r['level']}" == k]
        v["mean_abs_action"] = float(np.mean([r["mean_abs_action"] for r in g]))
        print(f"{k:16s} {v['finished']}/{v['runs']} laps, mean {v['lap_mean'] or 0:.2f} +- "
              f"{v['lap_std'] or 0:.2f} s, off {v['off_track']}, worst clearance "
              f"{v['min_clearance_worst']:.2f} m, |a| {v['mean_abs_action']:.2f}")
    path = pathlib.Path(args.out) if args.out else ROOT / "runs" / "gps" / "benchmark_c.json"
    path.write_text(json.dumps({"summary": summary, "runs": rows,
                                "drivers": {label(c): c for c in ckpts},
                                "config": load_config(overrides=overrides)}, indent=1))
    print(f"wrote {path} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
