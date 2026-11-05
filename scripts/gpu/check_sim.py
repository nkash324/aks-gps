#!/usr/bin/env python3
"""Check the GPU sim (kart_gpu) against the CPU sim (kart_gps), driving with driver B.

  python scripts/gpu/check_sim.py                 # both parts
  python scripts/gpu/check_sim.py --part det      # 1. same lap, no noise
  python scripts/gpu/check_sim.py --part stat     # 2. many noisy laps, compared as statistics

1. Deterministic: perfect sensors, nominal kart, so both sims drive exactly one possible
   lap. Lap time and path should agree (float64 and float32 runs on the GPU).
2. Statistical: --gpu-laps laps per level on the GPU and --cpu-laps per level on the
   CPU (seeds 1000000+, away from every other seed range), realistic and harsh, karts
   and sensor errors randomised. Per-lap random draws differ between the two (numpy vs
   torch), so the two sets are compared as distributions: laps finished, lap time,
   closest edge, position error, lateral g. z = difference / its standard error.

Writes runs/gpu/check_sim.json.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import pathlib
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gps"))
from kart_gps.config import load_config, track_path  # noqa: E402

LEVELS = ("realistic", "harsh")
CPU_SEED = 1_000_000


# ---------------------------------------------------------------------- CPU
def cpu_lap_trace(cfg, track, line):
    from kart_gps.drivers import make_driver
    from kart_gps.env import LapEnv
    env = LapEnv(cfg, level="perfect", seed=0, track=track, randomize=False)
    drv = make_driver("pp_optimised", track, cfg, {"optimised": line})
    est = env.reset()
    drv.reset()
    path = []
    while True:
        est, done, info = env.step(*drv.act(est))
        path.append((info["true"]["x"], info["true"]["y"], info["true"]["v"]))
        if done:
            return info, np.array(path)


# ---------------------------------------------------------------------- GPU
def gpu_laps(cfg, track, line, n, levels, dtype, randomize, seed=0, trace=False, device="cuda"):
    """One lap per kart (no restarts). levels: (n,) level indices. -> list of row dicts (+ path of kart 0)."""
    import torch
    from kart_gpu.sim import BatchLap, END
    sim = BatchLap(cfg, n, device=device, dtype=dtype, track=track, line=line, randomize=randomize, seed=seed)
    est = sim.reset(level=torch.as_tensor(levels, device=sim.dev))
    if n >= 64:
        est = sim.use_graph()                     # same results, much faster
    first = torch.zeros(n, dtype=torch.long, device=sim.dev)
    res = {k: torch.zeros(n, dtype=torch.float64, device=sim.dev)
           for k in ("lap_time", "progress", "min_clearance", "max_lat_g")}
    errs, alive_hist, path = [], [], []
    mu = sim.s["P_mu"] if "P_mu" in sim.s else None
    while True:
        alive = first == 0
        est, done, info = sim.step(*sim.drive_b(est))
        errs.append(torch.hypot(est["x"] - info["true_x"], est["y"] - info["true_y"]).float())
        alive_hist.append(alive)
        if trace:
            path.append((info["true_x"][0].item(), info["true_y"][0].item(), info["true_v"][0].item()))
        new = alive & done
        first = torch.where(new, info["end"], first)
        for k in res:
            res[k] = torch.where(new, info[k].double(), res[k])
        if bool((first > 0).all()):
            break
    E = torch.stack(errs).cpu().numpy()
    A = torch.stack(alive_hist).cpu().numpy()
    out = []
    lv = np.asarray(levels)
    for i in range(n):
        e = E[A[:, i], i]
        out.append({"level": ["perfect", "realistic", "harsh"][lv[i]], "end": END[int(first[i])],
                    "lap_time": float(res["lap_time"][i]) if int(first[i]) == 1 else None,
                    "progress": float(res["progress"][i]), "min_clearance": float(res["min_clearance"][i]),
                    "max_lat_g": float(res["max_lat_g"][i]), "pos_err_p95": float(np.percentile(e, 95)),
                    "mu": float(mu[i]) if mu is not None else cfg["kart"]["mu"]})
    return out, np.array(path)


# ---------------------------------------------------------------------- part 1
def part_det(cfg, track, line):
    import torch
    print("== 1. deterministic lap: perfect sensors, nominal kart", flush=True)
    t = time.time()
    info, cpath = cpu_lap_trace(cfg, track, line)
    print(f"CPU: {info['end']} {info['lap_time']:.3f} s, closest edge {info['min_clearance']:+.3f} m "
          f"({time.time() - t:.0f} s)", flush=True)
    out = {"cpu": {"end": info["end"], "lap_time": info["lap_time"], "min_clearance": info["min_clearance"]}}
    for name, dt in (("float64", torch.float64), ("float32", torch.float32)):
        t = time.time()
        rows, gpath = gpu_laps(cfg, track, line, 1, [0], dt, randomize=False, trace=True)
        r = rows[0]
        # the last step is left out: the CPU stops mid-step when the lap ends, the GPU
        # finishes the step (its end time and results are taken at the same moment)
        m = min(len(cpath), len(gpath)) - 1
        gpath = gpath[:m]
        same_step = np.hypot(*(gpath[:m, :2] - cpath[:m, :2]).T)
        d = np.sqrt(((gpath[:, None, :2] - cpath[None, :, :2]) ** 2).sum(-1)).min(1)   # to the CPU path
        dv = np.abs(gpath[:m, 2] - cpath[:m, 2])
        dlap = (r["lap_time"] - info["lap_time"]) if r["lap_time"] is not None and info["lap_time"] else None
        print(f"GPU {name}: {r['end']} {r['lap_time'] or float('nan'):.3f} s (diff {dlap if dlap is not None else float('nan'):+.3f} s), "
              f"closest edge {r['min_clearance']:+.3f} m | path off the CPU path: max {d.max() * 100:.1f} cm, "
              f"mean {d.mean() * 100:.2f} cm | same-step distance max {same_step.max() * 100:.1f} cm, "
              f"speed diff max {dv.max():.3f} m/s ({time.time() - t:.0f} s)", flush=True)
        out[name] = {"end": r["end"], "lap_time": r["lap_time"], "lap_diff": dlap,
                     "min_clearance": r["min_clearance"], "path_off_max_m": float(d.max()),
                     "path_off_mean_m": float(d.mean()), "same_step_max_m": float(same_step.max()),
                     "speed_diff_max": float(dv.max())}
    return out


# ---------------------------------------------------------------------- part 2
def stats(rows):
    laps = np.array([r["lap_time"] for r in rows if r["end"] == "lap"])
    col = lambda k: np.array([r[k] for r in rows])
    n = len(rows)
    return {"runs": n, "finished": int(len(laps)), "finish_rate": len(laps) / n,
            "off_track": int(sum(r["end"] == "offtrack" for r in rows)),
            "other_fail": int(sum(r["end"] not in ("lap", "offtrack") for r in rows)),
            "lap_mean": float(laps.mean()) if len(laps) else None,
            "lap_std": float(laps.std()) if len(laps) else None,
            "lap_p05": float(np.percentile(laps, 5)) if len(laps) else None,
            "lap_p95": float(np.percentile(laps, 95)) if len(laps) else None,
            "clear_mean": float(col("min_clearance").mean()), "clear_std": float(col("min_clearance").std()),
            "pos_err_mean": float(col("pos_err_p95").mean()), "pos_err_std": float(col("pos_err_p95").std()),
            "lat_g_mean": float(col("max_lat_g").mean()), "lat_g_std": float(col("max_lat_g").std())}


def ks(a, b):
    """Two-sample Kolmogorov-Smirnov statistic and its asymptotic p-value."""
    a, b = np.sort(a), np.sort(b)
    x = np.concatenate([a, b])
    d = np.max(np.abs(np.searchsorted(a, x, "right") / len(a) - np.searchsorted(b, x, "right") / len(b)))
    ne = len(a) * len(b) / (len(a) + len(b))
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * d
    p = 2 * sum((-1) ** (j - 1) * math.exp(-2 * j * j * lam * lam) for j in range(1, 101))
    return float(d), float(min(1.0, max(0.0, p)))


def compare(g, c, gs, cs):
    """z-scores GPU vs CPU for the main numbers."""
    z = {}
    p1, p2 = gs["finish_rate"], cs["finish_rate"]
    pp = (gs["finished"] + cs["finished"]) / (gs["runs"] + cs["runs"])
    se = math.sqrt(max(pp * (1 - pp), 1e-12) * (1 / gs["runs"] + 1 / cs["runs"]))
    z["finish_rate"] = (p1 - p2) / se if se > 0 else 0.0
    for k, n1, n2 in (("lap", gs["finished"], cs["finished"]), ("clear", gs["runs"], cs["runs"]),
                      ("pos_err", gs["runs"], cs["runs"]), ("lat_g", gs["runs"], cs["runs"])):
        m1, m2 = gs[f"{k}_mean"], cs[f"{k}_mean"]
        s1, s2 = gs[f"{k}_std"], cs[f"{k}_std"]
        if m1 is None or m2 is None or n1 < 2 or n2 < 2:
            continue
        se = math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
        z[k] = (m1 - m2) / se if se > 0 else 0.0
    gl = [r["lap_time"] for r in g if r["end"] == "lap"]
    cl = [r["lap_time"] for r in c if r["end"] == "lap"]
    ksr = {"lap_time": ks(gl, cl) if gl and cl else None,
           "min_clearance": ks([r["min_clearance"] for r in g], [r["min_clearance"] for r in c])}
    return z, ksr


def part_stat(cfg, track, line, gpu_n, cpu_n, procs, cpu_seed=CPU_SEED, gpu_seed=12345):
    import torch
    from benchmark import _init, run_lap
    print(f"== 2. statistics: {gpu_n} GPU + {cpu_n} CPU laps per level, randomised karts and sensors", flush=True)
    t = time.time()
    levels = np.repeat([1, 2], gpu_n)
    g_rows, _ = gpu_laps(cfg, track, line, len(levels), levels, torch.float32, randomize=True, seed=gpu_seed)
    t_gpu = time.time() - t
    print(f"GPU: {len(g_rows)} laps in {t_gpu:.0f} s", flush=True)
    t = time.time()
    jobs = [("pp_optimised", lv, cpu_seed + i, False, None) for lv in LEVELS for i in range(cpu_n)]
    c_rows = []
    with mp.get_context("spawn").Pool(procs, initializer=_init, initargs=(None,)) as pool:
        for i, r in enumerate(pool.imap_unordered(run_lap, jobs, chunksize=2), 1):
            c_rows.append(r)
            if i % 60 == 0 or i == len(jobs):
                print(f"CPU: {i}/{len(jobs)} laps, {time.time() - t:.0f} s", flush=True)
    t_cpu = time.time() - t
    out = {"gpu_seconds": t_gpu, "cpu_seconds": t_cpu, "levels": {}}
    for lv in LEVELS:
        g = [r for r in g_rows if r["level"] == lv]
        c = [r for r in c_rows if r["level"] == lv]
        gs, cs = stats(g), stats(c)
        z, ksr = compare(g, c, gs, cs)
        out["levels"][lv] = {"gpu": gs, "cpu": cs, "z": z, "ks": ksr}
        f = lambda v, d=2: "-" if v is None else f"{v:.{d}f}"
        print(f"\n-- {lv}")
        print(f"{'':10s} {'finished':>14s} {'lap mean':>9s} {'lap std':>8s} {'p05':>7s} {'p95':>7s} "
              f"{'off':>5s} {'edge':>7s} {'pos err':>8s} {'lat g':>6s}")
        for name, s in (("GPU", gs), ("CPU", cs)):
            print(f"{name:10s} {s['finished']:>6d}/{s['runs']:<5d}{100 * s['finish_rate']:>3.0f}% {f(s['lap_mean']):>8s} "
                  f"{f(s['lap_std']):>8s} {f(s['lap_p05']):>7s} {f(s['lap_p95']):>7s} {s['off_track']:>5d} "
                  f"{f(s['clear_mean'], 3):>7s} {f(s['pos_err_mean'], 3):>8s} {f(s['lat_g_mean'], 3):>6s}")
        print("z (GPU - CPU) / s.e.: " + ", ".join(f"{k} {v:+.2f}" for k, v in z.items())
              + " | KS p: " + ", ".join(f"{k} {v[1]:.2f}" for k, v in ksr.items() if v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("det", "stat", "all"), default="all")
    ap.add_argument("--gpu-laps", type=int, default=2048, help="per level")
    ap.add_argument("--cpu-laps", type=int, default=300, help="per level")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--cpu-seed", type=int, default=CPU_SEED, help="first CPU lap seed")
    ap.add_argument("--gpu-seed", type=int, default=12345)
    ap.add_argument("--out", default=str(ROOT / "runs" / "gpu" / "check_sim.json"))
    args = ap.parse_args()
    from kart_gps.drivers import build_line
    from kart_gps.track import TrackMap
    cfg = load_config()
    track = TrackMap(track_path(cfg))
    line = build_line("optimised", track, cfg)
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(path.read_text()) if path.exists() else {}
    if args.part in ("det", "all"):
        res["deterministic"] = part_det(cfg, track, line)
        path.write_text(json.dumps(res, indent=1))
    if args.part in ("stat", "all"):
        res["statistical"] = part_stat(cfg, track, line, args.gpu_laps, args.cpu_laps, args.procs,
                                              args.cpu_seed, args.gpu_seed)
        path.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
