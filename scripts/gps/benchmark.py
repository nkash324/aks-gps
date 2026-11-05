#!/usr/bin/env python3
"""Benchmark the GPS-stack drivers: every driver x noise level x seed, one lap each,
in parallel CPU processes. Every driver gets the same seeds, so the same karts and the
same sensor errors.

  python scripts/gps/benchmark.py                      # 2 drivers x 3 levels x 30 laps
  python scripts/gps/benchmark.py --laps 10 --procs 4
  python scripts/gps/benchmark.py --map-shift 0.3      # stress test: map off by up to 0.3 m
  python scripts/gps/benchmark.py --tune               # pick the Pure Pursuit look-ahead
                                                       # (realistic noise, seeds 1000+)
Writes runs/gps/benchmark.json (or tune.json) plus traces of seed 0 for plots.
"""
import argparse
import itertools
import json
import math
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from kart_gps.config import ROOT, load_config, track_path  # noqa: E402
from kart_gps.drivers import build_line, make_driver  # noqa: E402
from kart_gps.env import LapEnv  # noqa: E402
from kart_gps.track import TrackMap  # noqa: E402

DRIVERS = ("pp_centre", "pp_optimised")
LEVELS = ("perfect", "realistic", "harsh")
_W = {}


def _init(overrides):
    cfg = load_config(overrides=overrides)
    track = TrackMap(track_path(cfg))
    lines = {k: build_line(k, track, cfg) for k in ("centreline", "optimised")}
    _W.update(cfg=cfg, track=track, lines=lines)


def run_lap(job):
    driver_name, level, seed, record, pp = job
    cfg, track = _W["cfg"], _W["track"]
    if pp:
        cfg = dict(cfg, drivers={"pure_pursuit": dict(cfg["drivers"]["pure_pursuit"], **pp)})
    driver = make_driver(driver_name, track, cfg, _W["lines"])
    env = LapEnv(cfg, level=level, seed=seed, track=track, record=record)
    est = env.reset()
    driver.reset()
    pos_err = []
    while True:
        est, done, info = env.step(*driver.act(est))
        t = info["true"]
        pos_err.append(math.hypot(est["x"] - t["x"], est["y"] - t["y"]))
        if done:
            break
    out = {"driver": driver_name, "level": level, "seed": seed, "pp": pp,
           "end": info["end"], "lap_time": info["lap_time"], "progress": info["progress"],
           "min_clearance": info["min_clearance"], "max_lat_g": info["max_lat_g"],
           "pos_err_p95": float(np.percentile(pos_err, 95)),
           "mu": env.kart_params["mu"], "map_offset": float(np.hypot(*env.sensors.shift))}
    if record:
        out["trace"] = [[round(v, 3) for v in row] for row in env.trace[::2]]
    return out


def summarise(rows):
    out = {}
    for (d, lv), grp in itertools.groupby(sorted(rows, key=lambda r: (r["driver"], r["level"])),
                                          key=lambda r: (r["driver"], r["level"])):
        grp = list(grp)
        laps = [r["lap_time"] for r in grp if r["end"] == "lap"]
        out[f"{d}/{lv}"] = {
            "runs": len(grp), "finished": len(laps),
            "off_track": sum(r["end"] == "offtrack" for r in grp),
            "other_fail": sum(r["end"] not in ("lap", "offtrack") for r in grp),
            "lap_mean": float(np.mean(laps)) if laps else None,
            "lap_std": float(np.std(laps)) if laps else None,
            "lap_best": float(np.min(laps)) if laps else None,
            "lap_worst": float(np.max(laps)) if laps else None,
            "min_clearance_median": float(np.median([r["min_clearance"] for r in grp])),
            "min_clearance_worst": float(np.min([r["min_clearance"] for r in grp])),
            "max_lat_g": float(np.max([r["max_lat_g"] for r in grp])),
            "pos_err_p95_median": float(np.median([r["pos_err_p95"] for r in grp])),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laps", type=int, default=30)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--map-shift", type=float, default=None,
                    help="map offset (m, +- per axis) to stress-test, e.g. 0.3 for a base station not calibrated to the survey")
    args = ap.parse_args()
    overrides = None
    if args.map_shift is not None:
        overrides = {"sensors": {"map_offset": {"shift": args.map_shift}}}
    out_dir = ROOT / "runs" / "gps"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.tune:
        grid = [{"gain": g, "base": b} for g in (0.25, 0.35, 0.45, 0.55) for b in (0.5, 1.0, 1.5)]
        jobs = [(d, "realistic", 1000 + s, False, pp) for d in DRIVERS for pp in grid
                for s in range(args.laps)]
    else:
        jobs = [(d, lv, s, s == 0, None) for d in DRIVERS for lv in LEVELS for s in range(args.laps)]

    t0 = time.time()
    with mp.Pool(args.procs, initializer=_init, initargs=(overrides,)) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(run_lap, jobs), 1):
            rows.append(r)
            if i % 20 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} laps, {time.time() - t0:.0f} s", flush=True)

    if args.tune:
        res = []
        for d in DRIVERS:
            for pp in grid:
                grp = [r for r in rows if r["driver"] == d and r["pp"] == pp]
                laps = [r["lap_time"] for r in grp if r["end"] == "lap"]
                fails = len(grp) - len(laps)
                score = (np.mean(laps) if laps else 999) + 20.0 * fails / len(grp)
                res.append({"driver": d, **pp, "finished": len(laps), "of": len(grp),
                            "lap_mean": float(np.mean(laps)) if laps else None,
                            "min_clearance_worst": float(min(r["min_clearance"] for r in grp)),
                            "score": float(score)})
                print(f"{d:13s} gain {pp['gain']:.2f} base {pp['base']:.1f}: {len(laps)}/{len(grp)} "
                      f"laps, mean {res[-1]['lap_mean'] or 0:.2f} s, score {score:.2f}")
        path = pathlib.Path(args.out) if args.out else out_dir / "tune.json"
        path.write_text(json.dumps(res, indent=1))
    else:
        summary = summarise(rows)
        for k, v in summary.items():
            print(f"{k:24s} {v['finished']}/{v['runs']} laps, mean {v['lap_mean'] or 0:.2f} "
                  f"+- {v['lap_std'] or 0:.2f} s, off {v['off_track']}, worst clearance "
                  f"{v['min_clearance_worst']:.2f} m, max {v['max_lat_g']:.2f} g")
        path = pathlib.Path(args.out) if args.out else out_dir / "benchmark.json"
        path.write_text(json.dumps({"summary": summary, "runs": rows,
                                    "config": load_config(overrides=overrides)}, indent=1))
    print(f"wrote {path} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
