#!/usr/bin/env python3
"""Benchmark drivers on the GPU sim: driver B and residual checkpoints (GPU- or CPU-trained),
every driver on the SAME laps: lap j has the same random kart, sensor errors and GPS
events for every driver (same RNG seed, same kart row, the random draws never depend on
the driving). So the comparison is lap by lap, within the GPU sim.

  python scripts/gpu/benchmark.py runs/gpu/<run>/ckpt/45088768 ... --cpu-ckpt runs/gps/residual/<run>/ckpt/6500352

Writes runs/gpu/benchmark_gpu.json in benchmark.json's format ({"summary", "runs",
"drivers"}), with a recorded realistic lap (lap 0) per driver for the maps. Lap numbers
here are GPU kart rows, not CPU seeds: these laps are not the CPU benchmark's seeds 0..29.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu"))
from eval import load_policy  # noqa: E402     (scripts/gpu/eval.py)
from kart_gps.config import load_config  # noqa: E402
from kart_gpu.residual import BatchResidual  # noqa: E402
from kart_gpu.sim import END  # noqa: E402


def _cpu_benchmark():
    """scripts/gps/benchmark.py (same file name as this one, so loaded by path)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("gps_benchmark", ROOT / "scripts" / "gps" / "benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

LEVELS = ("perfect", "realistic", "harsh")
SEED = 20261029


def label(ckpt, prefix):
    """'G 45M' / 'C 6.5M' from .../ckpt/<steps> (nearest 0.5M, or 50k under 1M)."""
    n = int(pathlib.Path(ckpt).name)
    if n >= 1_000_000:
        return f"{prefix} {round(n / 500_000) / 2:g}M"
    return f"{prefix} {round(n / 50_000) * 50}k"


def run_driver(cfg, name, pol, laps, track=None, line=None):
    env = BatchResidual(cfg, laps * len(LEVELS), seed=SEED, track=track, line=line)
    sim = env.sim
    lv = torch.tensor([i for i in range(len(LEVELS)) for _ in range(laps)], device=env.dev)
    obs = env.reset(level=lv)
    N, rec = env.N, laps                       # kart row 'laps' = realistic lap 0: recorded
    first = torch.zeros(N, dtype=torch.long, device=env.dev)
    keep = {k: torch.zeros(N, dtype=torch.float64, device=env.dev)
            for k in ("lap_time", "progress", "min_clearance", "max_lat_g")}
    act = torch.zeros(N, dtype=torch.float64, device=env.dev)
    jump, steps = torch.zeros_like(act), torch.zeros_like(act)
    prev = torch.zeros(N, 2, device=env.dev)
    errs, alive_h, trace = [], [], []
    while True:
        a = pol(obs)
        alive = first == 0
        obs, _, done, info = env.step(a, autoreset=False)
        s = sim.s
        est_x, est_y = sim.s["X"][:, 0], sim.s["X"][:, 1]
        errs.append(torch.hypot(est_x - info["true_x"], est_y - info["true_y"]).half())
        alive_h.append(alive)
        if bool(alive[rec]):
            trace.append(torch.stack([info["t"][rec], info["true_x"][rec], info["true_y"][rec], s["yaw"][rec],
                                      info["true_v"][rec], est_x[rec], est_y[rec], s["X"][rec, 3], s["v_target"][rec],
                                      s["curv"][rec], s["ax"][rec], s["ay"][rec], s["cmd"][1][rec]]).double())
        act += torch.where(alive, a.abs().mean(1).double(), 0.0)
        jump += torch.where(alive, (a - prev).abs().mean(1).double(), 0.0)
        steps += alive.double()
        prev = a
        new = alive & done
        first = torch.where(new, info["end"], first)
        for k in keep:
            keep[k] = torch.where(new, info[k].double(), keep[k])
        if bool((first > 0).all()):
            break
    E = torch.stack(errs).float().cpu().numpy()
    A = torch.stack(alive_h).cpu().numpy()
    F = first.cpu().numpy()
    K = {k: v.cpu().numpy() for k, v in keep.items()}
    act, jump, steps = act.cpu().numpy(), jump.cpu().numpy(), steps.cpu().numpy()
    mu = sim.s["P_mu"].cpu().numpy() if "P_mu" in sim.s else np.full(N, cfg["kart"]["mu"])
    off = torch.linalg.norm(sim.s["shift"], dim=1).cpu().numpy()
    tr = torch.stack(trace).cpu().numpy() if trace else None
    rows = []
    for j in range(N):
        e = E[A[:, j], j]
        r = {"driver": name, "level": LEVELS[j // laps], "seed": j % laps, "pp": None, "end": END[int(F[j])],
             "lap_time": float(K["lap_time"][j]) if F[j] == 1 else None, "progress": float(K["progress"][j]),
             "min_clearance": float(K["min_clearance"][j]), "max_lat_g": float(K["max_lat_g"][j]),
             "pos_err_p95": float(np.percentile(e, 95)), "mean_abs_action": float(act[j] / steps[j]),
             "mean_abs_jump": float(jump[j] / steps[j]), "mu": float(mu[j]), "map_offset": float(off[j])}
        if j == rec and tr is not None:
            r["trace"] = [[round(float(v), 3) for v in row] for row in tr[::2]]
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="*", help="GPU-trained checkpoint folders")
    ap.add_argument("--cpu-ckpt", nargs="*", default=[], help="CPU-trained checkpoint folders")
    ap.add_argument("--laps", type=int, default=1000, help="laps per level per driver")
    ap.add_argument("--out", default=str(ROOT / "runs" / "gpu" / "benchmark_gpu.json"))
    args = ap.parse_args()
    cfg = load_config()
    dev = torch.device("cuda")
    drivers = [("B", None)] + [(label(c, "G"), c) for c in args.ckpts] + [(label(c, "C"), c) for c in args.cpu_ckpt]
    names = [n for n, _ in drivers]
    assert len(set(names)) == len(names), f"two drivers with one label: {names}"
    from kart_gps.drivers import build_line
    from kart_gps.track import TrackMap
    from kart_gps.config import track_path
    track = TrackMap(track_path(cfg))
    line = build_line("optimised", track, cfg)
    rows, t0 = [], time.time()
    for name, c in drivers:
        rows += run_driver(cfg, name, load_policy(c, dev), args.laps, track, line)
        print(f"{name}: done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    summary = _cpu_benchmark().summarise(rows)
    for k, v in summary.items():
        g = [r for r in rows if f"{r['driver']}/{r['level']}" == k]
        v["mean_abs_action"] = float(np.mean([r["mean_abs_action"] for r in g]))
        v["mean_abs_jump"] = float(np.mean([r["mean_abs_jump"] for r in g]))
        print(f"{k:14s} {v['finished']}/{v['runs']} laps, mean {v['lap_mean'] or 0:.2f} +- {v['lap_std'] or 0:.2f} s, "
              f"off {v['off_track']}, clearance median {v['min_clearance_median']:.3f} m, |a| {v['mean_abs_action']:.3f}")
    out = pathlib.Path(args.out)
    out.write_text(json.dumps({"summary": summary, "runs": rows, "sim": "gpu", "laps_per_level": args.laps,
                               "drivers": {n: (str(pathlib.Path(c).resolve()) if c else "driver B")
                                           for n, c in drivers}}, indent=None))
    print(f"wrote {out} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
