#!/usr/bin/env python3
"""Oracle C: driver C's steering knobs (runs/formula/tune_c.json) with a speed plan chosen
per lap from the lap's TRUE grip: lat = alpha * 11.3 m/s^2 * mu, brake = beta * 4.81 * mu.
Searches (alpha, beta) on tuning laps (seed 7,000,000, realistic sensors) -- it shows the best a
formula driver could do if the grip were known. Writes <out>/tune_oracle.json.

  python scripts/formula/tune_oracle.py --out runs/formula
"""
import argparse
import itertools
import json

import numpy as np
import torch

from bench import MU_BINS, winner
from common import TUNE_SEED, level_vector, load_mintime, make_sim, plan_table, rows, run_laps, setup, summary
from kart_gpu.formula import BatchFormula


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/formula")
    ap.add_argument("--laps", type=int, default=300)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.65, 0.72, 0.8, 0.88])
    ap.add_argument("--betas", type=float, nargs="+", default=[0.75, 0.85, 0.95])
    ap.add_argument("--min-finish", type=float, default=0.993)
    a = ap.parse_args()
    cfg, track, base = setup()
    line = load_mintime(cfg)
    cand = winner(f"{a.out}/tune_c.json")
    knobs = {k: cand[k] for k in ("tau", "us", "w_space", "w_time", "zeta", "preview")}
    lv = level_vector({"realistic": a.laps})
    res = []
    for al, be in itertools.product(a.alphas, a.betas):
        sim, est = make_sim(cfg, track, base, a.laps, TUNE_SEED, lv)      # same laps every time
        b = np.abs(sim.s["P_mu"].cpu().numpy()[:, None] - MU_BINS[None]).argmin(1)
        tabs = np.array([plan_table(line, cfg, al * 11.3 * m, be * 4.81 * m) for m in MU_BINS])
        r = summary(rows(run_laps(sim, est, BatchFormula(sim, line, tabs, torch.tensor(b), **knobs)), lv))["realistic"]
        res.append({"alpha": al, "beta": be, "s": r})
        print(al, be, f"{r['finish_rate']:.3f}", r["lap_mean"], flush=True)
    ok = [r for r in res if r["s"]["finish_rate"] >= a.min_finish] or res
    win = min(ok, key=lambda r: (r["s"]["lap_mean"] or 99) + 60 * (1 - r["s"]["finish_rate"]))
    json.dump({"all": res, "winner": {"alpha": win["alpha"], "beta": win["beta"], **win["s"]}},
              open(f"{a.out}/tune_oracle.json", "w"), indent=1)
    print("winner", win["alpha"], win["beta"])


if __name__ == "__main__":
    main()
