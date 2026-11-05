#!/usr/bin/env python3
"""Evaluate residual-driver checkpoints in one batch job, then pick models by aspect.

  python scripts/gps/eval_residual.py runs/gps/residual/<run>               # every checkpoint saved so far
  python scripts/gps/eval_residual.py runs/gps/residual/<run> --procs 2     # e.g. while training still runs
  python scripts/gps/eval_residual.py runs/gps/residual/<run> --select-only # just redo the picks

Each checkpoint drives --realistic laps (seeds 500000+) and --harsh laps (seeds 600000+) with
no exploration noise. These seeds are kept apart from the benchmark seeds (0..29), so the
picks can be benchmarked fairly afterwards (benchmark_residual.py).
Results are stored per checkpoint (ckpt/<steps>/eval.json), so a checkpoint is only
driven once; a later call skips it.

Picks (<run>/selection.json + selection.md):
  fastest       all realistic laps finished, lowest mean lap time
  safest        most laps finished (both levels), then the most room to the edge (worst lap)
  consistent    all realistic laps finished, smallest lap-time spread
  smoothest     all realistic laps finished, smallest change in correction step to step
  best_harsh    most harsh laps finished, then fastest at harsh
  no_compromise no other checkpoint is at least as good on laps finished, lap time,
                room to the edge and smoothness and better on one of them
"""
import argparse
import json
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np
import psutil

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from benchmark_residual import _init, run_lap  # noqa: E402

SEEDS = {"realistic": 500000, "harsh": 600000}
GB_PER_PROC = 0.6
MIN_FREE_GB = 2.5


def auto_procs(cap=8):
    """As many processes as free RAM allows while keeping MIN_FREE_GB free."""
    free = psutil.virtual_memory().available / 1e9
    return max(1, min(cap, int((free - MIN_FREE_GB) / GB_PER_PROC)))


def summarise(rows):
    out = {}
    for lv in SEEDS:
        g = [r for r in rows if r["level"] == lv]
        if not g:
            continue
        laps = [r["lap_time"] for r in g if r["end"] == "lap"]
        out[lv] = {"finished": len(laps), "of": len(g),
                   "lap_mean": float(np.mean(laps)) if laps else None,
                   "lap_std": float(np.std(laps)) if laps else None,
                   "clear_worst": float(min(r["min_clearance"] for r in g)),
                   "clear_median": float(np.median([r["min_clearance"] for r in g])),
                   "action": float(np.mean([r["mean_abs_action"] for r in g])),
                   "jump": float(np.mean([r["mean_abs_jump"] for r in g]))}
    return out


def select(results):
    """results: {steps: summary} -> picks per aspect (steps) + the no-compromise set."""
    R = {s: v for s, v in results.items() if "realistic" in v}
    clean = [s for s, v in R.items() if v["realistic"]["finished"] == v["realistic"]["of"]]
    fin = {s: sum(v[lv]["finished"] for lv in v) for s, v in R.items()}
    picks = {}
    if clean:
        picks["fastest"] = min(clean, key=lambda s: R[s]["realistic"]["lap_mean"])
        picks["consistent"] = min(clean, key=lambda s: R[s]["realistic"]["lap_std"])
        picks["smoothest"] = min(clean, key=lambda s: R[s]["realistic"]["jump"])
    if R:
        picks["safest"] = max(R, key=lambda s: (fin[s], R[s]["realistic"]["clear_worst"]))
    H = [s for s in R if "harsh" in R[s]]
    if H:
        picks["best_harsh"] = max(H, key=lambda s: (R[s]["harsh"]["finished"], -(R[s]["harsh"]["lap_mean"] or 999)))

    # no-compromise (Pareto) set; every score is "higher is better"
    def scores(s):
        r = R[s]["realistic"]
        return (fin[s], -(r["lap_mean"] or 999), r["clear_worst"], -r["jump"])
    sc = {s: scores(s) for s in R}
    front = [s for s in R if not any(all(a >= b for a, b in zip(sc[o], sc[s])) and sc[o] != sc[s] for o in R)]
    return picks, sorted(front)


def write_selection(run, results):
    picks, front = select(results)
    (run / "selection.json").write_text(json.dumps({"picks": picks, "no_compromise": front,
                                                    "results": {str(k): v for k, v in sorted(results.items())}}, indent=1))

    def row(s):
        v = results[s]
        r, h = v["realistic"], v.get("harsh")
        hs = f"{h['finished']}/{h['of']}" if h else "-"
        return (f"| {s:,} | {r['finished']}/{r['of']} | {r['lap_mean'] or float('nan'):.2f} s | ±{r['lap_std'] or 0:.2f} | "
                f"{r['clear_worst']:+.2f} m | {r['jump']:.3f} | {hs} |")
    head = "| Steps | Realistic | Mean lap | Spread | Closest edge | Jumpiness | Harsh |\n|---|---|---|---|---|---|---|"
    md = ["# Checkpoint picks", "", head]
    md += [row(s).replace("| ", f"| **{a}** ", 1) for a, s in picks.items()]
    md += ["", f"## No-compromise set ({len(front)} checkpoints)", "", head] + [row(s) for s in front]
    (run / "selection.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="runs/gps/residual/<run>")
    ap.add_argument("--realistic", type=int, default=20, help="realistic laps per checkpoint")
    ap.add_argument("--harsh", type=int, default=10, help="harsh laps per checkpoint")
    ap.add_argument("--procs", default="auto", help="processes, or 'auto' (from free RAM)")
    ap.add_argument("--select-only", action="store_true")
    args = ap.parse_args()
    run = pathlib.Path(args.run)

    ckpts = sorted((p for p in (run / "ckpt").glob("*") if (p / "saved").exists()), key=lambda p: int(p.name))
    todo = [p for p in ckpts if not (p / "eval.json").exists()] if not args.select_only else []
    if todo:
        procs = auto_procs() if args.procs == "auto" else int(args.procs)
        n = {"realistic": args.realistic, "harsh": args.harsh}
        jobs = [(str(p.resolve()), lv, SEEDS[lv] + i, False) for p in todo for lv in SEEDS for i in range(n[lv])]
        print(f"{len(todo)} checkpoints x {args.realistic} realistic + {args.harsh} harsh laps = {len(jobs)} laps "
              f"on {procs} processes", flush=True)
        t0 = time.time()
        rows = {str(p.resolve()): [] for p in todo}
        need = {k: args.realistic + args.harsh for k in rows}
        with mp.get_context("spawn").Pool(procs, initializer=_init, initargs=(None,)) as pool:
            for i, r in enumerate(pool.imap_unordered(run_lap, jobs, chunksize=1), 1):
                rows[r["ckpt"]].append(r)
                if len(rows[r["ckpt"]]) == need[r["ckpt"]]:        # checkpoint complete: store it
                    p = pathlib.Path(r["ckpt"])
                    s = summarise(rows[r["ckpt"]])
                    (p / "eval.json").write_text(json.dumps({"summary": s, "laps": rows[r["ckpt"]]}, indent=1))
                    rr = s["realistic"]
                    print(f"[{int(p.name):,} steps] realistic {rr['finished']}/{rr['of']} "
                          f"{rr['lap_mean'] or float('nan'):.2f} s, edge {rr['clear_worst']:+.2f} m"
                          + (f", harsh {s['harsh']['finished']}/{s['harsh']['of']}" if "harsh" in s else "")
                          + f"  ({i}/{len(jobs)} laps, {(time.time() - t0) / 60:.1f} min)", flush=True)

    results = {int(p.name): json.loads((p / "eval.json").read_text())["summary"]
               for p in ckpts if (p / "eval.json").exists()}
    if results:
        write_selection(run, results)
    print(f"eval done -> {run / 'selection.md'}", flush=True)


if __name__ == "__main__":
    main()
