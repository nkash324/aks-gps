#!/usr/bin/env python3
"""Benchmark one driver on the benchmark laps (seed 20261029, N per sensor level).

Laps depend only on (seed, number of karts, sensor levels), so run every driver with the
same --per-level and they all drive identical laps (scripts/formula/check_laps.py). The
result goes to <out>/bench_<driver>_<per-level>.json (summary + compact per-lap rows).

ablations: c_mincurv (C's steering + speed limits on the min-curvature line), c_bplan (C on the mintime line
with driver B's speed plan 6.1 / 5.0)
drivers: g2 (460.1M checkpoint), b (published), btuned (6.1 / 5.0 / 0.35), c (tuned formula
driver, runs/formula/tune_c.json), mpc (runs/formula/tune_mpc.json), oracle (C with perfect
knowledge of the lap's grip: --oracle-json)

  python scripts/formula/bench.py --driver c --per-level 1000
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch

from common import (BENCH_SEED, DEVICE, G2, LEVELS, ROOT, level_vector, load_mintime, make_sim, plan_table, rows,
                    run_laps, setup, summary)
from eval import load_policy  # noqa: E402
from tune_b import patch_b  # noqa: E402
from kart_gpu.formula import BatchFormula  # noqa: E402
from kart_gpu.residual import BatchResidual  # noqa: E402

MU_BINS = np.arange(0.84, 1.17, 0.04)


def winner(path):
    return json.load(open(path))["winner"]["cand"]


def run_g2(cfg, track, base, lv, seed, log):
    N = len(lv)
    env = BatchResidual(cfg, N, device=DEVICE, graph=False, seed=seed, track=track, line=base)
    pol = load_policy(G2, env.dev)
    obs = env.reset(level=lv.to(env.dev))
    first = torch.zeros(N, dtype=torch.long)
    keep = {k: torch.zeros(N, dtype=torch.float64) for k in ("lap_time", "progress", "min_clearance", "max_lat_g")}
    n = 0
    while True:
        alive = first == 0
        obs, _, done, info = env.step(pol(obs), autoreset=False)
        new = alive & done
        first = torch.where(new, info["end"], first)
        for k in keep:
            keep[k] = torch.where(new, info[k].double(), keep[k])
        n += 1
        if n % 200 == 0:
            log(n, int((first > 0).sum()), N)
        if bool((first > 0).all()):
            break
    res = {k: v.numpy() for k, v in keep.items()}
    res["end"] = first.numpy()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True, choices=["g2", "b", "btuned", "c", "mpc", "oracle", "c_mincurv", "c_bplan"])
    ap.add_argument("--per-level", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT / "runs" / "formula"))
    ap.add_argument("--seed", type=int, default=BENCH_SEED)
    ap.add_argument("--c-json", default=str(ROOT / "runs" / "formula" / "tune_c.json"))
    ap.add_argument("--mpc-json", default=str(ROOT / "runs" / "formula" / "tune_mpc.json"))
    ap.add_argument("--oracle-json", default=str(ROOT / "runs" / "formula" / "tune_oracle.json"))
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=0, help="MPC only: karts per batch (0 = all at once)")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    cfg, track, base = setup()
    lv = level_vector({l: a.per_level for l in LEVELS})
    N = len(lv)
    t0 = time.time()
    log = lambda n, d, N_: print(f"  step {n}: {d}/{N_} laps ended, {time.time() - t0:.0f} s", flush=True)
    info = {}
    if a.driver == "g2":
        res = run_g2(cfg, track, base, lv, a.seed, log)
    elif a.driver in ("b", "btuned"):
        sim, est = make_sim(cfg, track, base, N, a.seed, lv)
        if a.driver == "btuned":
            patch_b(sim, [(6.1, 5.0, 0.35)], N, base, cfg)
        res = run_laps(sim, est, sim.drive_b, progress=log)
    elif a.driver in ("c", "oracle", "c_mincurv", "c_bplan"):
        line = load_mintime(cfg)
        sim, est = make_sim(cfg, track, base, N, a.seed, lv)
        cand = winner(a.c_json)
        knobs = {k: cand[k] for k in ("tau", "us", "w_space", "w_time", "zeta", "preview")}
        if a.driver == "c":
            drv = BatchFormula(sim, line, plan_table(line, cfg, cand["lat"], cand["brake"]), **knobs)
        elif a.driver == "c_mincurv":
            drv = BatchFormula(sim, base, plan_table(base, cfg, cand["lat"], cand["brake"]), **knobs)
        elif a.driver == "c_bplan":
            from kart_gps.planner import Line, speed_plan
            pl = cfg["planner"]
            ln = Line(line.pts, "mintime")
            drv = BatchFormula(sim, line, speed_plan(ln, cfg["kart"], 6.1, 5.0, pl["top_speed"]), **knobs)
        else:
            o = json.load(open(a.oracle_json))["winner"]
            mu = sim.s["P_mu"].cpu().numpy()
            b = np.abs(mu[:, None] - MU_BINS[None]).argmin(1)
            tabs = np.array([plan_table(line, cfg, o["alpha"] * 11.3 * m, o["beta"] * 4.81 * m) for m in MU_BINS])
            drv = BatchFormula(sim, line, tabs, torch.tensor(b), **knobs)
            info["oracle"] = o
        info["cand"] = cand
        res = run_laps(sim, est, drv, progress=log)
    else:
        from kart_gpu.mpc import BatchMPC
        line = load_mintime(cfg)
        w = json.load(open(a.mpc_json))["winner"]["cand"]
        info["cand"] = w
        keys = ("w_e", "w_b", "margin", "w_g", "w_v", "w_prog", "w_sk", "w_sv", "us", "lag")
        ch = a.chunk or N
        parts = []
        sim, est = make_sim(cfg, track, base, N, a.seed, lv)       # all karts' laps drawn at once
        # (chunking would change the draws; the whole batch is one sim, MPC solves it in one go)
        drv = BatchMPC(sim, line, track, cfg, plan_table(line, cfg, w["lat"], w["brake"]),
                       iters=w.get("iters", 12), lr=w.get("lr", 0.1), **{k: w[k] for k in keys if k in w})
        res = run_laps(sim, est, drv, progress=log)
    R = rows(res, lv)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    laps = {l: [{"end": r["end"], "t": None if r["lap_time"] is None else round(r["lap_time"], 3),
                 "clear": round(r["min_clearance"], 3), "g": round(r["max_lat_g"], 3)}
                for r in R if r["level"] == l] for l in LEVELS}
    s = summary(R)
    f = out / f"bench_{a.driver}_{a.per_level}.json"
    json.dump({"driver": a.driver, "per_level": a.per_level, "seed": a.seed, "device": DEVICE,
               "summary": s, "info": info, "seconds": time.time() - t0, "laps": laps}, open(f, "w"))
    for l in LEVELS:
        x = s[l]
        print(f"{a.driver:8s} {l:9s} finish {x['finish_rate']:.1%} mean {x['lap_mean']:.2f} median {x['lap_median']:.2f} "
              f"offtrack {x['off_track']} clear med {x['clear_median']:.3f} p05 {x['clear_p05']:.3f} "
              f"g {x['max_lat_g_median']:.2f}")
    print(f"-> {f} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
