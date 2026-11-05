#!/usr/bin/env python3
"""Benchmark every residual checkpoint (CPU- and GPU-trained) plus drivers A and B on the GPU sim.

  python scripts/gpu/bench_all.py --cpu-run runs/gps/residual/<run> --gpu-run runs/gpu/<run>
  python scripts/gpu/bench_all.py --run "G:GPU run 2:runs/gpu/<run2>" --run "F:Fine-tuned:runs/gpu/<ft>" \
      --ref "C 5.1M=runs/gps/residual/<run>/ckpt/5105664" --out runs/gpu/bench_run2

Every driver drives the same laps: the batch is split into one block per driver and every
block gets the same random draws (kart, sensor errors, GPS events), so lap j is the same
lap for every driver and they compare lap by lap.

  stage 1  every checkpoint, --screen-real realistic + --screen-harsh harsh laps: ranks them
  stage 2  A, B and the best --shortlist of each type (CPU, GPU), --laps laps per sensor level,
           with lap 0 at realistic recorded for the maps

Ranking: finish at least as many realistic laps as driver B (on the same laps), then the
lowest mean realistic lap time; if too few qualify, the most laps finished, then lap time.
Writes <out>/stage1.json and <out>/stage2.json (default runs/gpu/bench_all/).
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu"))
from eval import load_policy  # noqa: E402   (scripts/gpu/eval.py)
from kart_gps.config import load_config, track_path  # noqa: E402
from kart_gps.drivers import build_line  # noqa: E402
from kart_gps.track import TrackMap  # noqa: E402
from kart_gpu.residual import BatchResidual  # noqa: E402
from kart_gpu.sim import END  # noqa: E402

LEVELS = ("perfect", "realistic", "harsh")
SEED = 20261029


def label(ckpt, prefix):
    return f"{prefix} {int(pathlib.Path(ckpt).name) / 1e6:.1f}M"


def checkpoints(run):
    return sorted((p for p in (pathlib.Path(run) / "ckpt").glob("*") if (p / "saved").exists()),
                  key=lambda p: int(p.name))


def drive(cfg, track, lines, drivers, laps, max_karts, record=False, log=""):
    """drivers: list of (name, kind, ckpt or None); kind 'A' drives the centreline, the rest
    the optimised line. laps: {level: n}. -> {name: list of lap dicts}, {name: trace rows}."""
    per = sum(laps.values())
    lv_block = torch.tensor([LEVELS.index(lv) for lv in LEVELS for _ in range(laps.get(lv, 0))])
    rec_row = laps.get("perfect", 0)                    # realistic lap 0 in each block
    dev = torch.device("cuda")
    rows, traces = {}, {}
    groups = [[d for d in drivers if d[1] == "A"], [d for d in drivers if d[1] != "A"]]
    G = max(1, max_karts // per)
    chunks = [g[i:i + G] for g in groups for i in range(0, len(g), G)]
    t0 = time.time()
    for ci, chunk in enumerate(chunks):
        K = len(chunk)
        line = lines["centreline" if chunk[0][1] == "A" else "optimised"]
        env = BatchResidual(cfg, per * K, seed=SEED, track=track, line=line, tile=K)
        sim = env.sim
        pols = [load_policy(c, dev) for _, _, c in chunk]
        obs = env.reset(level=lv_block.repeat(K).to(dev))
        N = env.N
        first = torch.zeros(N, dtype=torch.long, device=dev)
        keep = {k: torch.zeros(N, dtype=torch.float64, device=dev)
                for k in ("lap_time", "progress", "min_clearance", "max_lat_g")}
        act = torch.zeros(N, dtype=torch.float64, device=dev)
        jump, steps = torch.zeros_like(act), torch.zeros_like(act)
        perr = [[] for _ in range(1)]
        prev = torch.zeros(N, 2, device=dev)
        rec_idx = torch.tensor([i * per + rec_row for i in range(K)], device=dev)
        tr = []
        while True:
            a = torch.cat([p(obs[i * per:(i + 1) * per]) for i, p in enumerate(pols)])
            alive = first == 0
            obs, _, done, info = env.step(a, autoreset=False)
            s = sim.s
            err = torch.hypot(s["X"][:, 0] - info["true_x"], s["X"][:, 1] - info["true_y"])
            perr[0].append(torch.where(alive, err, torch.full_like(err, float("nan"))).half())
            if record:
                r = rec_idx
                tr.append((alive[r], torch.stack([info["t"][r], info["true_x"][r], info["true_y"][r], s["yaw"][r],
                                                  info["true_v"][r], s["X"][r, 0], s["X"][r, 1], s["X"][r, 3],
                                                  s["v_target"][r], s["curv"][r], s["ax"][r], s["ay"][r],
                                                  s["cmd"][1][r]], 1).double()))
            act += torch.where(alive, a.abs().mean(1).double(), 0.0)
            jump += torch.where(alive, (a - prev).abs().mean(1).double(), 0.0)
            steps += alive.double()
            prev = a
            new = alive & done
            first = torch.where(new, info["end"], first)
            for k in keep:
                keep[k] = torch.where(new, info[k].double(), keep[k])
            if bool((first > 0).all()):
                break
        E = torch.stack(perr[0]).float()
        p95 = torch.nanquantile(E, 0.95, dim=0).cpu().numpy()
        F = first.cpu().numpy()
        Kp = {k: v.cpu().numpy() for k, v in keep.items()}
        act, jump, steps = act.cpu().numpy(), jump.cpu().numpy(), steps.cpu().numpy()
        mu = sim.s["P_mu"].cpu().numpy() if "P_mu" in sim.s else np.full(N, cfg["kart"]["mu"])
        off = torch.linalg.norm(sim.s["shift"], dim=1).cpu().numpy()
        lvs = lv_block.repeat(K).numpy()
        for i, (name, kind, _) in enumerate(chunk):
            out = []
            for j in range(i * per, (i + 1) * per):
                lv = LEVELS[lvs[j]]
                out.append({"level": lv, "lap": (j - i * per) - sum(laps.get(x, 0) for x in LEVELS[:lvs[j]]),
                            "end": END[int(F[j])], "lap_time": float(Kp["lap_time"][j]) if F[j] == 1 else None,
                            "progress": float(Kp["progress"][j]), "min_clearance": float(Kp["min_clearance"][j]),
                            "max_lat_g": float(Kp["max_lat_g"][j]), "pos_err_p95": float(p95[j]),
                            "mean_abs_action": float(act[j] / steps[j]), "mean_abs_jump": float(jump[j] / steps[j]),
                            "mu": float(mu[j]), "map_offset": float(off[j])})
            rows[name] = out
            if record:
                pts = [row[i].cpu().numpy() for al, row in tr if bool(al[i])]
                traces[name] = [[round(float(v), 3) for v in p] for p in pts[::2]]
        print(f"{log}batch {ci + 1}/{len(chunks)}: {K} drivers x {per} laps done "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        del env, sim
        torch.cuda.empty_cache()
    return rows, traces


def summary(laps):
    out = {}
    for lv in LEVELS:
        g = [r for r in laps if r["level"] == lv]
        if not g:
            continue
        t = np.array([r["lap_time"] for r in g if r["end"] == "lap"])
        cl = np.array([r["min_clearance"] for r in g])
        out[lv] = {"runs": len(g), "finished": int(len(t)), "finish_rate": len(t) / len(g),
                   "off_track": int(sum(r["end"] == "offtrack" for r in g)),
                   "other_fail": int(sum(r["end"] not in ("lap", "offtrack") for r in g)),
                   "lap_mean": float(t.mean()) if len(t) else None, "lap_std": float(t.std()) if len(t) else None,
                   "lap_best": float(t.min()) if len(t) else None,
                   "lap_p10": float(np.percentile(t, 10)) if len(t) else None,
                   "lap_p90": float(np.percentile(t, 90)) if len(t) else None,
                   "clear_median": float(np.median(cl)), "clear_p05": float(np.percentile(cl, 5)),
                   "max_lat_g": float(np.max([r["max_lat_g"] for r in g])),
                   "pos_err_p95_median": float(np.median([r["pos_err_p95"] for r in g])),
                   "action": float(np.mean([r["mean_abs_action"] for r in g])),
                   "jump": float(np.mean([r["mean_abs_jump"] for r in g]))}
    return out


def rank(S, names, b_rate, n):
    """Best n of names: realistic finish rate >= B's, then fastest; then the rest by laps finished."""
    ok = sorted((k for k in names if S[k]["realistic"]["finish_rate"] >= b_rate and S[k]["realistic"]["lap_mean"]),
                key=lambda k: S[k]["realistic"]["lap_mean"])
    rest = sorted((k for k in names if k not in ok),
                  key=lambda k: (-S[k]["realistic"]["finish_rate"], S[k]["realistic"]["lap_mean"] or 999))
    return (ok + rest)[:n], len(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-run", default=None, help="shorthand for --run 'C:CPU-trained:<path>'")
    ap.add_argument("--gpu-run", default=None, help="shorthand for --run 'G:GPU-trained:<path>'")
    ap.add_argument("--run", action="append", default=[],
                    help="KIND:TITLE:PATH, e.g. 'F:Fine-tuned from C 5.1M:runs/gpu/<run>' (KIND = label prefix)")
    ap.add_argument("--ref", action="append", default=[],
                    help="NAME=PATH: a reference checkpoint driven in stage 2 only (e.g. 'C 5.1M=runs/...')")
    ap.add_argument("--screen-real", type=int, default=100)
    ap.add_argument("--screen-harsh", type=int, default=50)
    ap.add_argument("--shortlist", type=int, default=8, help="per type, into stage 2")
    ap.add_argument("--top", type=int, default=5, help="per type, picked from stage 2")
    ap.add_argument("--laps", type=int, default=1000, help="stage 2, per sensor level")
    ap.add_argument("--max-karts", type=int, default=18000)
    ap.add_argument("--out", default=str(ROOT / "runs" / "gpu" / "bench_all"))
    ap.add_argument("--skip-stage1", action="store_true", help="reuse stage1.json")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    track = TrackMap(track_path(cfg))
    lines = {k: build_line(k, track, cfg) for k in ("centreline", "optimised")}
    runs = ([("C", "CPU-trained", args.cpu_run)] if args.cpu_run else []) +         ([("G", "GPU-trained", args.gpu_run)] if args.gpu_run else []) +         [tuple(r.split(":", 2)) for r in args.run]
    ck = {kind: [(label(c, kind), kind, str(c)) for c in checkpoints(path)] for kind, _, path in runs}
    refs = [(n, "R", c) for n, c in (r.split("=", 1) for r in args.ref)]
    base = [("A", "A", None), ("B", "B", None)]
    screened = base + [d for kind, _, _ in runs for d in ck[kind]]
    names = [d[0] for d in screened + refs]
    assert len(set(names)) == len(names), "two checkpoints with one label"
    t0 = time.time()

    if args.skip_stage1:
        st1 = json.loads((out / "stage1.json").read_text())
    else:
        print(f"stage 1: " + " + ".join(f"{len(ck[k])} {k}" for k, _, _ in runs) + " checkpoints + A, B x "
              f"({args.screen_real} realistic + {args.screen_harsh} harsh laps)", flush=True)
        rows, _ = drive(cfg, track, lines, screened,
                        {"realistic": args.screen_real, "harsh": args.screen_harsh}, args.max_karts, log="stage 1 ")
        st1 = {"laps": {"realistic": args.screen_real, "harsh": args.screen_harsh},
               "runs": {k: title for k, title, _ in runs},
               "drivers": {n: {"kind": k, "ckpt": c, "steps": int(pathlib.Path(c).name) if c else None}
                           for n, k, c in screened},
               "summary": {n: summary(r) for n, r in rows.items()}}
        (out / "stage1.json").write_text(json.dumps(st1, indent=1))
    S = st1["summary"]
    b_rate = S["B"]["realistic"]["finish_rate"]
    short = {}
    for kind, title, _ in runs:
        short[kind], ok = rank(S, [d[0] for d in ck[kind]], b_rate, args.shortlist)
        print(f"  {title}: {ok} checkpoints finish as many realistic laps as B ({100 * b_rate:.1f}%); "
              f"shortlist {short[kind]}", flush=True)
    print(f"stage 1 done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    D = {n: (n, k, c) for n, k, c in screened}
    drivers = base + [D[n] for kind, _, _ in runs for n in short[kind]] + refs
    print(f"stage 2: {len(drivers)} drivers x {args.laps} laps per level", flush=True)
    rows, traces = drive(cfg, track, lines, drivers, {lv: args.laps for lv in LEVELS}, args.max_karts,
                         record=True, log="stage 2 ")
    S2 = {n: summary(r) for n, r in rows.items()}
    b2 = S2["B"]["realistic"]["finish_rate"]
    top = {kind: rank(S2, short[kind], b2, args.top)[0] for kind, _, _ in runs}
    info = dict(st1["drivers"])
    info.update({n: {"kind": "R", "ckpt": c, "steps": int(pathlib.Path(c).name)} for n, _, c in refs})
    st2 = {"laps": args.laps, "levels": list(LEVELS), "runs": {k: title for k, title, _ in runs},
           "shortlist": short, "top": top, "refs": [n for n, _, _ in refs],
           "drivers": {n: info[n] for n, _, _ in drivers}, "summary": S2,
           "laps_by_driver": {n: [[r["level"], r["lap"], r["end"], r["lap_time"], round(r["min_clearance"], 4)]
                                  for r in rw] for n, rw in rows.items()},
           "traces": traces}
    (out / "stage2.json").write_text(json.dumps(st2))
    for n in ["A", "B"] + [x for k in top for x in top[k]] + [n for n, _, _ in refs]:
        r, h = S2[n]["realistic"], S2[n]["harsh"]
        print(f"{n:10s} realistic {100 * r['finish_rate']:.1f}% {r['lap_mean'] or 0:.2f} s | "
              f"harsh {100 * h['finish_rate']:.1f}% {h['lap_mean'] or 0:.2f} s", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
