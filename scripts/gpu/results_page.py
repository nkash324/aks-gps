#!/usr/bin/env python3
"""Build the GPU benchmark page (one self-contained HTML file) from scripts/gpu/bench_all.py output.

  python scripts/gpu/results_page.py                       # runs/gpu/bench_all/ -> runs/gpu/results.html
  python scripts/gpu/results_page.py --dir <bench_all dir> --out <file>

Sections: training progress (every checkpoint, stage 1), the full benchmark of A, B and the
top 5 of each training type (stage 2), lap-by-lap differences to B, and the driving pattern
on one lap for A, B and the top 3 of each type.
"""
import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kart_gps.config import load_config, track_path  # noqa: E402
from kart_gps.drivers import build_line  # noqa: E402
from kart_gps.planner import kart_half_width  # noqa: E402
from kart_gps.track import TrackMap  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
LEVELS = ("perfect", "realistic", "harsh")


def _gps_page():
    """scripts/gps/results_page.py (same file name as this one, so loaded by path): trace_view etc."""
    spec = importlib.util.spec_from_file_location("gps_results_page", ROOT / "scripts" / "gps" / "results_page.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _r(a, nd=2):
    return [round(float(v), nd) for v in a]


def rnd(d, nd=4):
    return {k: (round(v, nd) if isinstance(v, float) else v) for k, v in d.items()}


def cpu_check(path, s2):
    """GPU (stage 2) vs CPU (seeds 0..29) for the models the CPU benchmark ran, plus A and B
    from the CPU benchmark page's data (runs/gps/benchmark.json, same settings)."""
    cpu = json.loads(path.read_text())
    by_path = {str(pathlib.Path(v["ckpt"]).resolve()).lower(): n for n, v in s2["drivers"].items() if v.get("ckpt")}
    names = {lab: by_path.get(str(pathlib.Path(c).resolve()).lower()) for lab, c in cpu["drivers"].items()}
    rows = [dict(r, driver=names[r["driver"]]) for r in cpu["runs"] if names.get(r["driver"])]
    ab = json.loads((ROOT / "runs" / "gps" / "benchmark.json").read_text())
    rows += [dict(r, driver={"pp_centre": "A", "pp_optimised": "B"}[r["driver"]]) for r in ab["runs"]
             if r["driver"] in ("pp_centre", "pp_optimised")]
    out = {"models": [n for n in names.values() if n], "laps": None, "cpu": {}, "gpu": {}, "delta_b": {}}
    order = ["A", "B"] + out["models"]
    for k in order:
        for lv in LEVELS:
            g = [r for r in rows if r["driver"] == k and r["level"] == lv]
            if not g:
                continue
            t_ = np.array([r["lap_time"] for r in g if r["end"] == "lap"])
            out["laps"] = len(g)
            out["cpu"][f"{k}/{lv}"] = {"runs": len(g), "finished": int(len(t_)),
                                       "lap_mean": round(float(t_.mean()), 3) if len(t_) else None,
                                       "lap_se": round(float(t_.std(ddof=1) / np.sqrt(len(t_))), 3) if len(t_) > 1 else None}
            G = s2["summary"][k][lv]
            out["gpu"][f"{k}/{lv}"] = {"finish_rate": round(G["finish_rate"], 4), "lap_mean": round(G["lap_mean"], 3),
                                       "lap_se": round(G["lap_std"] / np.sqrt(G["finished"]), 3)}
    # gap to B on the same CPU seeds (both finished)
    for k in out["models"] + ["A"]:
        for lv in LEVELS:
            b = {r["seed"]: r for r in rows if r["driver"] == "B" and r["level"] == lv}
            d = [r["lap_time"] - b[r["seed"]]["lap_time"] for r in rows
                 if r["driver"] == k and r["level"] == lv and r["end"] == "lap" and r["seed"] in b and b[r["seed"]]["end"] == "lap"]
            if d:
                out["delta_b"][f"{k}/{lv}"] = {"mean": round(float(np.mean(d)), 3),
                                               "se": round(float(np.std(d, ddof=1) / np.sqrt(len(d))), 3) if len(d) > 1 else None}
    # the same gap on the GPU, on its shared laps
    L = s2["laps_by_driver"]
    out["gpu_delta_b"] = {}
    for k in out["models"] + ["A"]:
        for lv in LEVELS:
            bl = {r[1]: r for r in L["B"] if r[0] == lv}
            d = [r[3] - bl[r[1]][3] for r in L[k] if r[0] == lv and r[2] == "lap" and bl[r[1]][2] == "lap"]
            if d:
                out["gpu_delta_b"][f"{k}/{lv}"] = {"mean": round(float(np.mean(d)), 3),
                                                   "se": round(float(np.std(d, ddof=1) / np.sqrt(len(d))), 3)}
    out["order"] = order
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "runs" / "gpu" / "bench_all"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "gpu" / "results.html"))
    ap.add_argument("--template", default=str(HERE / "results_template.html"))
    ap.add_argument("--cpu-check", default=str(ROOT / "runs" / "gpu" / "cpu_check.json"),
                    help="benchmark_residual.py output for a few of the models (CPU, seeds 0..29)")
    args = ap.parse_args()
    d = pathlib.Path(args.dir)
    s1 = json.loads((d / "stage1.json").read_text())
    s2 = json.loads((d / "stage2.json").read_text())
    gp = _gps_page()
    cfg = load_config()
    track = TrackMap(track_path(cfg))
    lines = {k: build_line(k, track, cfg) for k in ("centreline", "optimised")}

    runs = s2.get("runs") or {"C": "CPU-trained", "G": "GPU-trained"}
    refs = s2.get("refs", [])
    # ---- training progress: every checkpoint, stage 1
    prog = {k: [] for k in runs}
    for name, info in s1["drivers"].items():
        if info["kind"] in prog:
            r, h = s1["summary"][name]["realistic"], s1["summary"][name]["harsh"]
            prog[info["kind"]].append({"name": name, "steps": info["steps"], "lap": r["lap_mean"] and round(r["lap_mean"], 3),
                                       "fin": round(r["finish_rate"], 4), "hfin": round(h["finish_rate"], 4),
                                       "hlap": h["lap_mean"] and round(h["lap_mean"], 3)})
    for v in prog.values():
        v.sort(key=lambda x: x["steps"])
    ref1 = {k: {"lap": round(s1["summary"][k]["realistic"]["lap_mean"], 3),
                "fin": round(s1["summary"][k]["realistic"]["finish_rate"], 4),
                "hfin": round(s1["summary"][k]["harsh"]["finish_rate"], 4)} for k in ("A", "B")}

    # ---- stage 2: summaries, lap-by-lap difference to B
    top = s2["top"]
    order = ["A", "B"] + [x for k in runs for x in top[k]] + refs
    summ = {k: {lv: rnd(v) for lv, v in s2["summary"][k].items()} for k in order}
    laps = s2["laps_by_driver"]
    bl = {(r[0], r[1]): r for r in laps["B"]}
    delta = {}
    for k in order:
        if k == "B":
            continue
        for lv in LEVELS:
            both, only_b, only_d = [], 0, 0
            for r in laps[k]:
                if r[0] != lv:
                    continue
                b = bl[(lv, r[1])]
                fa, fb = r[2] == "lap", b[2] == "lap"
                if fa and fb:
                    both.append(r[3] - b[3])
                only_b += fb and not fa
                only_d += fa and not fb
            if both:
                delta[f"{k}/{lv}"] = {"q": _r(np.percentile(both, [10, 25, 50, 75, 90]), 3),
                                      "faster": round(float(np.mean(np.array(both) < 0)), 3), "n": len(both),
                                      "only_b": only_b, "only_d": only_d}
    # the lap shown on the maps: lap 0 at realistic (same kart and sensor errors for everyone)
    show = ["A", "B"] + [x for k in runs for x in top[k][:3]]
    traces = {}
    for k in show:
        t = s2["traces"].get(k)
        if t and len(t) > 10:
            traces[k] = gp.trace_view({"trace": t, "lap_time": next((r[3] for r in laps[k] if r[0] == "realistic" and r[1] == 0), None)})
    lap0 = {k: next(r for r in laps[k] if r[0] == "realistic" and r[1] == 0) for k in show}

    check = cpu_check(pathlib.Path(args.cpu_check), s2) if pathlib.Path(args.cpu_check).exists() else None

    data = {
        "check": check,
        "outer": _r(track.outer[::2].ravel()), "inner": _r(track.inner[::2].ravel()),
        "lines": {k: {"length": v.length} for k, v in lines.items()},
        "track_length": track.length, "kart_width": round(2 * kart_half_width(cfg["kart"]), 3),
        "stage1": {"laps": s1["laps"], "progress": prog, "ref": ref1,
                   "shortlist": s2["shortlist"], "n": {k: len(v) for k, v in prog.items()}},
        "stage2": {"laps": s2["laps"], "order": order, "top": top, "summary": summ, "delta": delta,
                   "steps": {k: s2["drivers"][k]["steps"] for k in order}},
        "show": show, "traces": traces,
        "lap0": {k: {"end": v[2], "lap": v[3]} for k, v in lap0.items()},
    }
    data["groups"] = [{"kind": k, "title": v} for k, v in runs.items()]
    data["refs"] = refs
    data["kinds"] = {n: s2["drivers"][n]["kind"] for n in order if n in s2["drivers"]}
    html = pathlib.Path(args.template).read_text(encoding="utf-8")
    html = html.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    out = pathlib.Path(args.out)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
