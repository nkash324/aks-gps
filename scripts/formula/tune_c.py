#!/usr/bin/env python3
"""Tune driver C on tuning laps (seed 7,000,000, never the benchmark laps).

  round 1  random search over (lat, brake, preview, w_space, w_time, us, tau, zeta), --screen
           realistic laps per candidate, --block candidates per batch (same laps for all)
  round 2  perturb the best --top candidates (--n2 of them)
  final    the --shortlist best by score on --laps2 realistic + --laps2/2 harsh fresh laps
           (another seed). Tuned C = fastest with a realistic finish rate >= --min-finish.
Score = mean lap of the finishers + 60 s per unfinished share (screening).
Writes <out>/tune_c.json (all candidates, the shortlist, the winner).

  python scripts/formula/tune_c.py --out runs/formula
"""
import argparse
import json
import pathlib
import time

import numpy as np

from common import TUNE_SEED, c_laps, load_mintime, setup, summary

RANGES = {"lat": (5.5, 10.0), "brake": (2.8, 5.0), "preview": (0.15, 0.9), "w_space": (0.25, 1.0),
          "w_time": (1.5, 8.0), "us": (1.0, 1.6), "tau": (0.03, 0.16), "zeta": (0.7, 1.4)}


def score(s):
    return (s["lap_mean"] or 99.0) + 60.0 * (1 - s["finish_rate"])


def evaluate(cfg, track, base, line, cands, laps, seed, block):
    out = []
    for i in range(0, len(cands), block):
        res = c_laps(cfg, track, base, line, cands[i:i + block], laps, seed)
        out += [summary(r) for r in res]
        print(f"  {len(out)}/{len(cands)} done", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/formula")
    ap.add_argument("--n1", type=int, default=48)
    ap.add_argument("--n2", type=int, default=32)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--screen", type=int, default=150)
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--shortlist", type=int, default=4)
    ap.add_argument("--laps2", type=int, default=1000)
    ap.add_argument("--min-finish", type=float, default=0.993)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg, track, base = setup()
    line = load_mintime(cfg)
    rng = np.random.default_rng(a.seed)
    laps = {"realistic": a.screen}
    allc = []

    def sample(centre=None, shrink=0.25):
        c = {}
        for k, (lo, hi) in RANGES.items():
            if centre is None:
                c[k] = float(rng.uniform(lo, hi))
            else:
                c[k] = float(np.clip(centre[k] + rng.normal(0, shrink * (hi - lo)), lo, hi))
        return c

    t = time.time()
    c1 = [sample() for _ in range(a.n1)]
    c1[0] = dict(lat=6.0, brake=3.5, preview=0.5, w_space=0.5, w_time=3.0, us=1.29, tau=0.08, zeta=1.0)
    s1 = evaluate(cfg, track, base, line, c1, laps, TUNE_SEED, a.block)
    allc += [{"cand": c, "s": s["realistic"], "round": 1} for c, s in zip(c1, s1)]
    best = sorted(allc, key=lambda r: score(r["s"]))[:a.top]
    print("round 1 best", [round(score(b["s"]), 2) for b in best], f"{time.time() - t:.0f}s", flush=True)
    c2 = [sample(best[i % a.top]["cand"], 0.08) for i in range(a.n2)]
    s2 = evaluate(cfg, track, base, line, c2, laps, TUNE_SEED, a.block)
    allc += [{"cand": c, "s": s["realistic"], "round": 2} for c, s in zip(c2, s2)]
    ranked = sorted(allc, key=lambda r: score(r["s"]))
    short = [r["cand"] for r in ranked[:a.shortlist]]
    print("round 2 best", [round(score(r["s"]), 2) for r in ranked[:a.shortlist]], f"{time.time() - t:.0f}s", flush=True)
    fin = []
    for c in short:      # fresh laps: 1000 realistic + 500 harsh, one candidate at a time
        r = c_laps(cfg, track, base, line, [c], {"realistic": a.laps2, "harsh": a.laps2 // 2}, TUNE_SEED + 1)[0]
        s = summary(r)
        fin.append({"cand": c, "s": s})
        print("final", {k: round(v, 3) for k, v in c.items()}, s["realistic"]["finish_rate"], s["realistic"]["lap_mean"],
              s["harsh"]["finish_rate"], flush=True)
    ok = [f for f in fin if f["s"]["realistic"]["finish_rate"] >= a.min_finish]
    win = min(ok or fin, key=lambda f: f["s"]["realistic"]["lap_mean"])
    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump({"all": allc, "final": fin, "winner": win, "met_bar": bool(ok)}, open(f"{a.out}/tune_c.json", "w"))
    print("winner", win["cand"], "met bar:", bool(ok), f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
