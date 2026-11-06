#!/usr/bin/env python3
"""Join the bench_<driver>_<n>.json files into one table and compare each driver with G2.

For the realistic laps both finished: median and mean gap (driver - G2, seconds; positive =
G2 faster), bootstrap 95% CI of the mean gap and the share of laps G2 is faster.
Writes <dir>/summary_<n>.json (no per-lap rows) and prints a markdown table.

  python scripts/formula/compare.py --per-level 1000
"""
import argparse
import json
import pathlib

import numpy as np

from common import LEVELS, ROOT, bootstrap_ci

ORDER = ["g2", "b", "btuned", "c", "mpc", "oracle"]
NAMES = {"g2": "G2 460.1M (learned)", "b": "B (published)", "btuned": "B tuned", "c": "C tuned (formula)",
         "mpc": "MPC tuned", "oracle": "Oracle C (true grip)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "runs" / "formula"))
    ap.add_argument("--per-level", type=int, default=1000)
    a = ap.parse_args()
    d = pathlib.Path(a.dir)
    data = {k: json.load(open(d / f"bench_{k}_{a.per_level}.json")) for k in ORDER
            if (d / f"bench_{k}_{a.per_level}.json").exists()}
    out = {"per_level": a.per_level, "drivers": {k: v["summary"] for k, v in data.items()}, "vs_g2": {}}
    md = []
    for lv in LEVELS:
        md += [f"\n**{lv}** ({a.per_level} laps)\n", "| Driver | Finished | Mean lap | Median lap | Off-track | Clearance med / p5 | Max lat g (med) |",
               "|---|---|---|---|---|---|---|"]
        for k, v in data.items():
            s = v["summary"][lv]
            md.append(f"| {NAMES[k]} | {s['finish_rate']:.1%} | {s['lap_mean']:.2f} s | {s['lap_median']:.2f} s | "
                      f"{s['off_track']} | {s['clear_median']:.2f} / {s['clear_p05']:.2f} m | {s['max_lat_g_median']:.2f} |")
    if "g2" in data:
        md += ["\n**Versus G2, realistic laps both finished** (gap = driver - G2; positive = G2 faster)\n",
               "| Driver | Laps | Median gap | Mean gap | 95% CI of mean | G2 faster on |", "|---|---|---|---|---|---|"]
        g = data["g2"]["laps"]["realistic"]
        for k, v in data.items():
            if k == "g2":
                continue
            o = v["laps"]["realistic"]
            diff = np.array([y["t"] - x["t"] for x, y in zip(g, o) if x["t"] is not None and y["t"] is not None])
            lo, hi = bootstrap_ci(diff)
            out["vs_g2"][k] = {"laps": len(diff), "median_gap": float(np.median(diff)), "mean_gap": float(diff.mean()),
                               "ci95": [lo, hi], "g2_faster_share": float((diff > 0).mean())}
            r = out["vs_g2"][k]
            md.append(f"| {NAMES[k]} | {r['laps']} | {r['median_gap']:+.2f} s | {r['mean_gap']:+.2f} s | "
                      f"[{lo:+.2f}, {hi:+.2f}] | {r['g2_faster_share']:.1%} |")
    json.dump(out, open(d / f"summary_{a.per_level}.json", "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
