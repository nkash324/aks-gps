#!/usr/bin/env python3
"""Give driver B its best hand tuning, then compare it fairly with the learned driver.

Driver B's speed plan uses safe default limits (gps.yaml planner: 6 m/s^2 cornering,
4 m/s^2 braking) and a Pure Pursuit look-ahead tuned once for those speeds. The learned
driver had 460M steps of training to find speed, so to compare like with like this
searches B's settings on the GPU, on tuning laps that are NOT the benchmark laps:

  stage 1  a grid of (cornering limit, braking limit, look-ahead gain), --screen laps
           each at realistic sensors
  stage 2  the fastest --shortlist settings that finish nearly every screening lap,
           on --laps2 realistic + --laps2 / 2 harsh fresh laps. Tuned B = the fastest
           with a realistic finish rate >= --min-finish (G2 460.1M: 99.3%)
  final    tuned B, B and the learned driver on the exact laps of the published
           benchmark (bench_all.py stage 2: same seed, 1,000 laps per sensor level),
           so B and G2 reproduce their published numbers

The path stays the optimised line; only the speed plan and the look-ahead change
(the learned driver drives that same path).

  python scripts/gpu/tune_b.py --out runs/gpu/tune_b
"""
import argparse
import itertools
import json
import pathlib
import sys
import time

import numpy as np
import psutil
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu"))
from eval import load_policy  # noqa: E402     (scripts/gpu/eval.py)
from kart_gps.config import load_config, track_path  # noqa: E402
from kart_gps.drivers import build_line  # noqa: E402
from kart_gps.planner import Line, speed_plan  # noqa: E402
from kart_gps.track import TrackMap  # noqa: E402
from kart_gpu.residual import BatchResidual  # noqa: E402
from kart_gpu.sim import END, BatchLap  # noqa: E402

LEVELS = ("perfect", "realistic", "harsh")
BENCH_SEED = 20261029          # bench_all.py: the published benchmark laps
TUNE_SEED = 7_000_000          # tuning laps, kept apart from the benchmark
G2 = ROOT / "runs" / "gpu" / "20261029-234306-run2" / "ckpt" / "460062720"
MIN_FREE_GB = 2.5


def ram_check():
    free = psutil.virtual_memory().available / 2**30
    if free < MIN_FREE_GB:
        raise SystemExit(f"stopping: only {free:.1f} GB RAM free (keeping {MIN_FREE_GB} GB for the computer)")
    return free


def plan(base, kart, lat, brake, top):
    """The optimised path with a speed plan for these limits."""
    ln = Line(base.pts, "optimised")
    speed_plan(ln, kart, lat, brake, top)
    return ln


def patch_b(sim, settings, per, base_line, cfg):
    """Make sim.drive_b drive block i (karts i*per .. ) with settings[i] = (lat, brake, gain)
    or None (B as published)."""
    pl, kart, pp = cfg["planner"], cfg["kart"], cfg["drivers"]["pure_pursuit"]
    tabs, gains = [], []
    for s in settings:
        lat, brake, gain = s if s else (pl["lat_accel"], pl["brake_decel"], pp["gain"])
        tabs.append(plan(base_line, kart, lat, brake, pl["top_speed"]).speed)
        gains.append(gain)
    dev, f = sim.dev, sim.f
    tab = torch.tensor(np.array(tabs), device=dev, dtype=f)              # (blocks, line points)
    gain = torch.tensor(gains, device=dev, dtype=f)
    blk = torch.arange(sim.N, device=dev) // per

    def drive_b(est):
        s, c = sim.s, pp
        x, y, yaw, v = est["x"], est["y"], est["yaw"], est["v"]
        k = sim._nearest(sim.l_pts, sim.l_n, x, y, s["bk"])
        s["bk"].copy_(k)
        ld = (gain[blk] * v + c["base"]).clamp(c["min"], c["max"])
        j = sim._ahead(k, ld)
        dx, dy = sim.l_pts[j, 0] - x, sim.l_pts[j, 1] - y
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        lx, ly = cy * dx + sy * dy, -sy * dx + cy * dy
        curv = 2.0 * ly / (lx * lx + ly * ly).clamp(min=1e-6)
        speed = tab[blk, sim._ahead(k, v * c["speed_preview"])]
        speed = torch.where(est["gnss_age"] > 0.5, speed * c["lost_slowdown"],
                            torch.where(~est["rtk_fixed"], speed * c["float_slowdown"], speed))
        return speed, curv
    sim.drive_b = drive_b


def b_laps(cfg, track, base_line, settings, laps, seed):
    """Driver B with each of settings on the same laps. laps: {level: n}. -> list of row lists."""
    per = sum(laps.values())
    K = len(settings)
    lv = torch.tensor([LEVELS.index(l) for l in LEVELS for _ in range(laps.get(l, 0))] * K)
    sim = BatchLap(cfg, per * K, track=track, line=base_line, seed=seed, tile=K)
    patch_b(sim, settings, per, base_line, cfg)
    sim.reset(level=lv.to(sim.dev))
    est = sim.use_graph()
    return _run(sim, lambda est, obs: sim.step(*sim.drive_b(est)), est, None, per, K, lv)


def _run(sim, step, est, obs, per, K, lv):
    N = sim.N
    first = torch.zeros(N, dtype=torch.long, device=sim.dev)
    keep = {k: torch.zeros(N, dtype=torch.float64, device=sim.dev)
            for k in ("lap_time", "progress", "min_clearance", "max_lat_g")}
    n = 0
    while True:
        alive = first == 0
        est, done, info = step(est, obs)
        new = alive & done
        first = torch.where(new, info["end"], first)
        for k in keep:
            keep[k] = torch.where(new, info[k].double(), keep[k])
        n += 1
        if n % 200 == 0:
            ram_check()
        if bool((first > 0).all()):
            break
    F = first.cpu().numpy()
    Kp = {k: v.cpu().numpy() for k, v in keep.items()}
    lvs = lv.numpy()
    out = []
    for i in range(K):
        out.append([{"level": LEVELS[lvs[j]], "end": END[int(F[j])],
                     "lap_time": float(Kp["lap_time"][j]) if F[j] == 1 else None,
                     "progress": float(Kp["progress"][j]), "min_clearance": float(Kp["min_clearance"][j]),
                     "max_lat_g": float(Kp["max_lat_g"][j])} for j in range(i * per, (i + 1) * per)])
    return out


def summary(rows):
    out = {}
    for lv in LEVELS:
        g = [r for r in rows if r["level"] == lv]
        if not g:
            continue
        laps = [r["lap_time"] for r in g if r["end"] == "lap"]
        cl = [r["min_clearance"] for r in g]
        out[lv] = {"runs": len(g), "finished": len(laps), "finish_rate": len(laps) / len(g),
                   "off_track": sum(r["end"] == "offtrack" for r in g),
                   "lap_mean": float(np.mean(laps)) if laps else None,
                   "lap_std": float(np.std(laps)) if laps else None,
                   "clear_median": float(np.median(cl)), "clear_p05": float(np.percentile(cl, 5)),
                   "max_lat_g": float(np.median([r["max_lat_g"] for r in g]))}
    return out


def name(s):
    return "B (published)" if s is None else f"B {s[0]:g}/{s[1]:g}/{s[2]:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, nargs="+", default=[6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0])
    ap.add_argument("--brake", type=float, nargs="+", default=[4.0, 4.5, 5.0])
    ap.add_argument("--gain", type=float, nargs="+", default=[0.25, 0.35, 0.45])
    ap.add_argument("--screen", type=int, default=150, help="stage 1 realistic laps per setting")
    ap.add_argument("--shortlist", type=int, default=8)
    ap.add_argument("--laps2", type=int, default=1000, help="stage 2 realistic laps (+ half as many harsh)")
    ap.add_argument("--min-finish", type=float, default=0.993, help="realistic finish rate tuned B must reach")
    ap.add_argument("--laps", type=int, default=1000, help="final, per sensor level")
    ap.add_argument("--max-karts", type=int, default=12000)
    ap.add_argument("--out", default=str(ROOT / "runs" / "gpu" / "tune_b"))
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    track = TrackMap(track_path(cfg))
    base_line = build_line("optimised", track, cfg)
    t0 = time.time()
    print(f"RAM free {ram_check():.1f} GB", flush=True)

    # ---- stage 1: the grid
    grid = list(itertools.product(args.lat, args.brake, args.gain))
    per_batch = max(1, args.max_karts // args.screen)
    # progress is saved after every batch; a rerun with the same laps skips finished settings
    # (every batch drives the same tuning laps: same seed, same laps per block)
    part = out / "stage1_partial.json"
    key = lambda s: "B" if s is None else f"{s[0]},{s[1]},{s[2]}"
    done = {}
    if part.exists():
        saved = json.loads(part.read_text())
        if saved["laps"] == args.screen and saved["seed"] == TUNE_SEED:
            done = saved["results"]
            print(f"stage 1: {len(done)} settings already done, reusing them", flush=True)
    todo = [s for s in grid + [None] if key(s) not in done]
    for b in range(0, len(todo), per_batch):
        chunk = todo[b:b + per_batch]
        res = b_laps(cfg, track, base_line, chunk, {"realistic": args.screen}, TUNE_SEED)
        for s, rows in zip(chunk, res):
            done[key(s)] = summary(rows)["realistic"]
        part.write_text(json.dumps({"laps": args.screen, "seed": TUNE_SEED, "results": done}))
        print(f"stage 1: {len(done)}/{len(grid) + 1} settings, {time.time() - t0:.0f} s, "
              f"RAM free {ram_check():.1f} GB", flush=True)
    st1 = {s: done[key(s)] for s in grid}
    ref1 = done["B"]
    print(f"stage 1 B (published settings): {ref1['finish_rate']:.1%} finished, {ref1['lap_mean']:.2f} s", flush=True)
    for s in sorted(st1, key=lambda s: st1[s]["lap_mean"] or 1e9):
        r = st1[s]
        print(f"  {name(s):18s} {r['finish_rate']:6.1%} finished  {r['lap_mean'] or 0:6.2f} s", flush=True)
    (out / "stage1.json").write_text(json.dumps(
        {"grid": [{"lat": s[0], "brake": s[1], "gain": s[2], **st1[s]} for s in grid], "b": ref1,
         "laps": args.screen, "seed": TUNE_SEED}, indent=1))

    # ---- stage 2: the fastest settings that finish nearly every screening lap
    need = args.min_finish - 2.0 / args.screen           # screening allows ~2 laps of slack
    ok = sorted((s for s in grid if st1[s]["finish_rate"] >= need and st1[s]["lap_mean"]),
                key=lambda s: st1[s]["lap_mean"])[:args.shortlist]
    laps2 = {"realistic": args.laps2, "harsh": args.laps2 // 2}
    prev = out / "stage2.json"
    prev = json.loads(prev.read_text()) if prev.exists() else None
    if prev and prev["laps"] == laps2 and [(r["lat"], r["brake"], r["gain"]) for r in prev["shortlist"]] == ok:
        print("stage 2: already done with this shortlist, reusing it", flush=True)
        st2 = {(r["lat"], r["brake"], r["gain"]): {lv: r[lv] for lv in ("realistic", "harsh")}
               for r in prev["shortlist"]}
        st2[None] = prev["b"]
    else:
        res = b_laps(cfg, track, base_line, ok + [None], laps2, TUNE_SEED + 1)
        st2 = {s: summary(rows) for s, rows in zip(ok + [None], res)}
    print(f"stage 2 ({args.laps2} realistic + {args.laps2 // 2} harsh fresh laps), {time.time() - t0:.0f} s:", flush=True)
    for s in ok + [None]:
        r, h = st2[s]["realistic"], st2[s]["harsh"]
        print(f"  {name(s):18s} realistic {r['finish_rate']:6.1%} {r['lap_mean'] or 0:6.2f} s | "
              f"harsh {h['finish_rate']:6.1%} {h['lap_mean'] or 0:6.2f} s", flush=True)
    good = [s for s in ok if st2[s]["realistic"]["finish_rate"] >= args.min_finish]
    best = min(good, key=lambda s: st2[s]["realistic"]["lap_mean"]) if good else None
    (out / "stage2.json").write_text(json.dumps(
        {"shortlist": [{"lat": s[0], "brake": s[1], "gain": s[2], **st2[s]} for s in ok],
         "b": st2[None], "laps": laps2, "seed": TUNE_SEED + 1, "min_finish": args.min_finish,
         "tuned": None if best is None else {"lat": best[0], "brake": best[1], "gain": best[2]}}, indent=1))
    if best is None:
        print("no setting reached the finish-rate bar; tuned B = the published B", flush=True)
    else:
        print(f"tuned B = {name(best)} (cornering {best[0]} m/s^2, braking {best[1]} m/s^2, "
              f"look-ahead gain {best[2]})", flush=True)

    # ---- final: the published benchmark laps, tuned B vs B vs G2
    laps = {lv: args.laps for lv in LEVELS}
    per = sum(laps.values())
    drivers = [("G2 460.1M", G2, None), ("B", None, None), ("B tuned", None, best)]
    K = len(drivers)
    lv = torch.tensor([LEVELS.index(l) for l in LEVELS for _ in range(laps[l])] * K)
    env = BatchResidual(cfg, per * K, seed=BENCH_SEED, track=track, line=base_line, tile=K)
    patch_b(env.sim, [d[2] for d in drivers], per, base_line, cfg)
    pols = [load_policy(c, env.dev) for _, c, _ in drivers]
    obs = env.reset(level=lv.to(env.dev))

    def step(est, obs_):
        o = state["obs"]
        a = torch.cat([p(o[i * per:(i + 1) * per]) for i, p in enumerate(pols)])
        o, _, done, info = env.step(a, autoreset=False)
        state["obs"] = o
        return None, done, info
    state = {"obs": obs}
    res = _run(env.sim, step, None, None, per, K, lv)
    final = {d[0]: summary(rows) for d, rows in zip(drivers, res)}
    # lap by lap vs G2 on laps both finished
    pairs = {}
    for i, d in enumerate(drivers[1:], 1):
        g = [(a["lap_time"], b["lap_time"]) for a, b in zip(res[0], res[i])
             if a["level"] == "realistic" and a["end"] == "lap" and b["end"] == "lap"]
        diff = np.array([b - a for a, b in g])
        pairs[d[0]] = {"laps": len(g), "median_gap": float(np.median(diff)), "mean_gap": float(diff.mean()),
                       "g2_faster_share": float((diff > 0).mean())}
    print(f"final (published benchmark laps, {args.laps} per level), {time.time() - t0:.0f} s:", flush=True)
    for d in drivers:
        line_ = " | ".join(f"{lv} {final[d[0]][lv]['finish_rate']:6.1%} {final[d[0]][lv]['lap_mean'] or 0:6.2f} s"
                           for lv in LEVELS)
        print(f"  {d[0]:10s} {line_}", flush=True)
    for k, v in pairs.items():
        print(f"  G2 vs {k}: G2 faster by {v['median_gap']:.2f} s (median) on {v['laps']} realistic laps "
              f"both finished; G2 faster on {v['g2_faster_share']:.1%}", flush=True)
    (out / "final.json").write_text(json.dumps(
        {"summary": final, "vs_g2": pairs, "laps": laps, "seed": BENCH_SEED,
         "tuned": None if best is None else {"lat": best[0], "brake": best[1], "gain": best[2]},
         "laps_by_driver": {d[0]: rows for d, rows in zip(drivers, res)}}, indent=1))
    print(f"done in {time.time() - t0:.0f} s -> {out}", flush=True)


if __name__ == "__main__":
    main()
