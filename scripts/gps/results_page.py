#!/usr/bin/env python3
"""Build the benchmark results page (one self-contained HTML file) from
runs/gps/benchmark.json (map calibrated) and runs/gps/benchmark_offset.json (--map-shift 0.3).

  python scripts/gps/results_page.py                                    -> runs/gps/results.html
  python scripts/gps/results_page.py --extra runs/gps/benchmark_c.json  # + learned drivers (benchmark_residual.py)
  python scripts/gps/results_page.py --gpu runs/gpu/benchmark_gpu.json   # + the GPU benchmark section (scripts/gpu/benchmark.py)
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from kart_gps.config import ROOT, load_config, track_path  # noqa: E402
from kart_gps.drivers import build_line  # noqa: E402
from kart_gps.planner import kart_half_width  # noqa: E402
from kart_gps.track import TrackMap  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent


def _r(a, nd=2):
    return [round(float(v), nd) for v in a]


DOT_EVERY = 0.5      # s between dots on the maps
SLOW_ACCEL = -0.7    # m/s^2: below = slowing (regen and/or hydraulic brake)
FAST_ACCEL = 0.7     # m/s^2: above = speeding up; between = holding speed
FINE_DOT_EVERY = 0.25  # s between dots on the single-driver maps
SMOOTH_S = 0.2       # s: centred moving average on acceleration before picking the phase
RELEASE = 0.4        # m/s^2: a slowing / speeding-up phase ends once |accel| drops below this
MIN_PHASE_S = 0.15   # s: shorter phases are absorbed into the one before


def phases(t, ax):
    """Per-sample phase with smoothing, hysteresis and a minimum length.
    Returns the smoothed acceleration and a list of phase names."""
    dt = float(np.median(np.diff(t)))
    w = max(1, int(round(SMOOTH_S / dt)) | 1)                 # odd window, centred
    pad = np.pad(ax, w // 2, mode="edge")
    acc = np.convolve(pad, np.ones(w) / w, mode="valid")
    state, out = "hold", []
    for a in acc:
        if state == "slow" and a > -RELEASE or state == "fast" and a < RELEASE:
            state = "hold"
        if a < SLOW_ACCEL:
            state = "slow"
        elif a > FAST_ACCEL:
            state = "fast"
        out.append(state)
    # absorb phases shorter than MIN_PHASE_S into the previous one
    min_n = max(1, int(np.ceil(MIN_PHASE_S / dt - 1e-9)))
    i = 0
    while i < len(out):
        j = i
        while j < len(out) and out[j] == out[i]:
            j += 1
        if i > 0 and j - i < min_n:
            out[i:j] = [out[i - 1]] * (j - i)
        i = j
    return acc, out


def segments(ph):
    """Runs of equal phase as [first index, last index (shared with the next run), phase]."""
    segs, i = [], 0
    while i < len(ph):
        j = i
        while j + 1 < len(ph) and ph[j + 1] == ph[i]:
            j += 1
        segs.append([i, min(j + 1, len(ph) - 1), ph[i]])
        i = j + 1
    return segs


def corners(d, v, yaw, x, y):
    """Slowest point of each corner: a local speed minimum at least 1 m/s below the
    speed 60 m before and after it. Label goes 6 m to the outside of the turn."""
    out = []
    for i in range(len(v)):
        near = np.abs(d - d[i]) < 30
        if v[i] > v[near].min():
            continue
        before = v[(d > d[i] - 60) & (d < d[i])]
        after = v[(d > d[i]) & (d < d[i] + 60)]
        if len(before) == 0 or len(after) == 0 or before.max() - v[i] < 1 or after.max() - v[i] < 1:
            continue
        if out and d[i] - out[-1]["d"] < 20:
            continue
        j0, j1 = max(0, i - 5), min(len(v) - 1, i + 5)
        turn = np.angle(np.exp(1j * (yaw[j1] - yaw[j0])))
        h = yaw[i]
        side = -1.0 if turn > 0 else 1.0           # outside of a left turn is to the right
        nx, ny = -np.sin(h) * side, np.cos(h) * side
        out.append({"d": float(d[i]), "x": float(x[i]), "y": float(y[i]), "v": round(float(v[i]), 1),
                    "lx": float(x[i] + 6 * nx), "ly": float(y[i] + 6 * ny)})
    return out


def trace_view(r):
    """A recorded lap -> what the maps and the speed trace draw."""
    tr = np.array(r["trace"])
    xy = tr[:, 1:3]
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    t, v, yaw, ax, brake = tr[:, 0], tr[:, 4], tr[:, 3], tr[:, 10], tr[:, 12]
    t = t - t[0]
    k = np.flatnonzero(np.diff(np.floor(t / DOT_EVERY + 1e-6), prepend=-1) > 0)   # one row per 0.5 s
    # phase from the average acceleration over each dot's 0.5 s
    acc = [float(ax[a:b].mean()) for a, b in zip(k, np.r_[k[1:], len(t)])]
    phase = ["slow" if a < SLOW_ACCEL else ("fast" if a > FAST_ACCEL else "hold") for a in acc]
    # single-driver maps: instantaneous phase, dots every 0.25 s interpolated between samples
    sacc, sph = phases(t, ax)
    tf = np.arange(0.0, t[-1] + 1e-9, FINE_DOT_EVERY)
    near = np.clip(np.searchsorted(t, tf - 1e-9), 0, len(t) - 1)
    fine = [{"x": round(float(np.interp(s, t, xy[:, 0])), 2), "y": round(float(np.interp(s, t, xy[:, 1])), 2),
             "t": round(float(s), 2), "d": round(float(np.interp(s, t, d)), 1),
             "v": round(float(np.interp(s, t, v)), 2), "a": round(float(np.interp(s, t, sacc)), 2),
             "p": sph[n], "b": bool(brake[n] > 0)} for s, n in zip(tf, near)]
    return {
        "fine": fine, "segs": segments(sph),
        "x": _r(xy[:, 0]), "y": _r(xy[:, 1]), "d": _r(d, 1), "v": _r(v), "t": _r(t), "lap": r["lap_time"],
        "dots": [{"x": round(float(xy[i, 0]), 2), "y": round(float(xy[i, 1]), 2), "t": round(float(t[i]), 1),
                  "d": round(float(d[i]), 1), "v": round(float(v[i]), 2), "a": round(a, 2), "p": ph,
                  "b": bool(brake[i:i + 5].max() > 0)}
                 for i, a, ph in zip(k, acc, phase)],
        "corners": corners(d, v, yaw, xy[:, 0], xy[:, 1])}


def gpu_block(path):
    """The GPU benchmark: per driver and level the summary plus the lap-by-lap difference
    to driver B on the laps both finished (same kart and sensor errors on each lap)."""
    g = json.loads(pathlib.Path(path).read_text())
    rows = g["runs"]
    names = list(g["drivers"])
    order = ["B"] + [n for n in names if n.startswith("C ")] + sorted(
        (n for n in names if n.startswith("G ")), key=lambda n: float(n[2:-1]) * (1e6 if n.endswith("M") else 1e3))
    lap = {(r["driver"], r["level"], r["seed"]): r for r in rows}
    out = {"laps": g["laps_per_level"], "drivers": order, "summary": {}, "delta": {}, "traces": {}}
    for k, v in g["summary"].items():
        out["summary"][k] = {a: (round(v[a], 4) if isinstance(v[a], float) else v[a]) for a in
                             ("runs", "finished", "off_track", "lap_mean", "lap_std", "lap_best", "min_clearance_median",
                              "max_lat_g", "mean_abs_action", "mean_abs_jump") if a in v}
    for d in order[1:]:
        for lv in ("perfect", "realistic", "harsh"):
            both, only_b, only_d = [], 0, 0
            for s in range(g["laps_per_level"]):
                a, b = lap.get((d, lv, s)), lap.get(("B", lv, s))
                if not a or not b:
                    continue
                fa, fb = a["end"] == "lap", b["end"] == "lap"
                if fa and fb:
                    both.append(a["lap_time"] - b["lap_time"])
                only_b += fb and not fa
                only_d += fa and not fb
            if both:
                q = np.percentile(both, [10, 25, 50, 75, 90])
                out["delta"][f"{d}/{lv}"] = {"q": _r(q, 3), "faster": round(float(np.mean(np.array(both) < 0)), 3),
                                             "n": len(both), "only_b": only_b, "only_d": only_d}
    for r in rows:
        if r.get("trace") and r["level"] == "realistic":
            out["traces"][r["driver"]] = trace_view(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", action="append", default=[],
                    help="benchmark_residual.py output (map calibrated) to add as more drivers")
    ap.add_argument("--gpu", default=None, help="scripts/gpu/benchmark.py output: its own section")
    args = ap.parse_args()
    out_dir = ROOT / "runs" / "gps"
    calibrated = json.loads((out_dir / "benchmark.json").read_text())
    offset = json.loads((out_dir / "benchmark_offset.json").read_text())
    extra = []
    for p in args.extra:
        e = json.loads(pathlib.Path(p).read_text())
        calibrated["summary"].update(e["summary"])
        calibrated["runs"] += e["runs"]
        extra += [d for d in e["drivers"] if d not in extra]
    cfg = load_config()
    track = TrackMap(track_path(cfg))
    lines = {k: build_line(k, track, cfg) for k in ("centreline", "optimised")}

    traces = {r["driver"]: trace_view(r) for r in calibrated["runs"]
              if r.get("trace") and r["level"] == "realistic"}

    def runs(data):
        return [{"driver": r["driver"], "level": r["level"], "seed": r["seed"], "end": r["end"],
                 "lap": r["lap_time"], "clear": round(r["min_clearance"], 3),
                 "offset": round(r["map_offset"], 3), "mu": round(r["mu"], 3)} for r in data["runs"]]

    data = {
        "outer": _r(track.outer[::2].ravel()), "inner": _r(track.inner[::2].ravel()),
        "lines": {k: {"x": _r(v.pts[::2, 0]), "y": _r(v.pts[::2, 1]), "plan": v.lap_estimate,
                      "length": v.length} for k, v in lines.items()},
        "traces": traces,
        "summary": {"calibrated": calibrated["summary"], "offset": offset["summary"]},
        "runs": {"calibrated": runs(calibrated), "offset": runs(offset)},
        "extra": extra,
        "gpu": gpu_block(args.gpu) if args.gpu else None,
        "track_length": track.length,
        "kart_width": round(2 * kart_half_width(cfg["kart"]), 3),
    }
    html = (HERE / "results_template.html").read_text(encoding="utf-8")
    html = html.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    path = out_dir / "results.html"
    path.write_text(html, encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
