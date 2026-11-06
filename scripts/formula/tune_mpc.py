#!/usr/bin/env python3
"""Tune the MPC driver on tuning laps (seed 7,000,000), same procedure as tune_c.py but with
fewer laps because the MPC is much slower.

  round 1  random search over (lat, brake, margin, w_b, w_g, w_e, w_sk, us, lag), --screen realistic laps
  round 2  perturb the best --top
  final    the --shortlist best on --laps2 realistic + --laps2/2 harsh fresh laps (another seed)
Tuned MPC = fastest final with a realistic finish rate >= --min-finish (G2: 99.3%).
Writes <out>/tune_mpc.json.

  python scripts/formula/tune_mpc.py --out runs/formula
"""
import argparse
import json
import pathlib
import time

import numpy as np

from common import TUNE_SEED, load_mintime, mpc_laps, setup, summary
from tune_c import score

RANGES = {"lat": (5.5, 8.5), "brake": (3.0, 4.6), "margin": (0.0, 0.35), "w_b": (100.0, 1500.0),
          "w_g": (1.0, 12.0), "w_e": (0.1, 2.0), "w_sk": (0.5, 6.0), "us": (1.1, 1.5), "lag": (0.03, 0.12)}
DEFAULT = dict(lat=7.3, brake=3.85, margin=0.10, w_b=400.0, w_g=4.0, w_e=0.5, w_sk=2.0, us=1.29, lag=0.05)


def evaluate(cfg, track, base, line, cands, laps, seed, block, log=print):
    out = []
    for i in range(0, len(cands), block):
        res = mpc_laps(cfg, track, base, line, cands[i:i + block], laps, seed)
        out += [summary(r) for r in res]
        log(f"  {len(out)}/{len(cands)} done", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/formula")
    ap.add_argument("--n1", type=int, default=16)
    ap.add_argument("--n2", type=int, default=12)
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--screen", type=int, default=40)
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--shortlist", type=int, default=2)
    ap.add_argument("--laps2", type=int, default=200)
    ap.add_argument("--min-finish", type=float, default=0.993)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)
    cfg, track, base = setup()
    line = load_mintime(cfg)
    rng = np.random.default_rng(a.seed)
    laps = {"realistic": a.screen}

    def sample(centre=None, shrink=0.08):
        c = {}
        for k, (lo, hi) in RANGES.items():
            c[k] = float(rng.uniform(lo, hi)) if centre is None else \
                float(np.clip(centre[k] + rng.normal(0, shrink * (hi - lo)), lo, hi))
        return c

    t = time.time()
    c1 = [c1[0]] + [sample(DEFAULT, 0.2) for _ in range(a.n1 - 1)]
    s1 = evaluate(cfg, track, base, line, c1, laps, TUNE_SEED, a.block)
    allc = [{"cand": c, "s": s["realistic"], "round": 1} for c, s in zip(c1, s1)]
    best = sorted(allc, key=lambda r: score(r["s"]))[:a.top]
    print("round 1 best", [round(score(b["s"]), 2) for b in best], f"{time.time() - t:.0f}s", flush=True)
    c2 = [sample(best[i % a.top]["cand"], 0.06) for i in range(a.n2)]
    s2 = evaluate(cfg, track, base, line, c2, laps, TUNE_SEED, a.block)
    allc += [{"cand": c, "s": s["realistic"], "round": 2} for c, s in zip(c2, s2)]
    ranked = sorted(allc, key=lambda r: score(r["s"]))
    short = [r["cand"] for r in ranked[:a.shortlist]]
    print("round 2 best", [round(score(r["s"]), 2) for r in ranked[:a.shortlist]], f"{time.time() - t:.0f}s", flush=True)
    fin = []
    for c in short:
        r = mpc_laps(cfg, track, base, line, [c], {"realistic": a.laps2, "harsh": a.laps2 // 2}, TUNE_SEED + 1)[0]
        s = summary(r)
        fin.append({"cand": c, "s": s})
        print("final", {k: round(v, 3) for k, v in c.items()}, s["realistic"]["finish_rate"], s["realistic"]["lap_mean"],
              s["harsh"]["finish_rate"], flush=True)
        pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    ok = [f for f in fin if f["s"]["realistic"]["finish_rate"] >= a.min_finish]
    win = min(ok or fin, key=lambda f: (f["s"]["realistic"]["lap_mean"] or 99) - 60 * f["s"]["realistic"]["finish_rate"]
              if not ok else f["s"]["realistic"]["lap_mean"])
    json.dump({"all": allc, "final": fin, "winner": win, "met_bar": bool(ok)}, open(f"{a.out}/tune_mpc.json", "w"))
    print("winner", win["cand"], "met bar:", bool(ok), f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
