#!/usr/bin/env python3
"""Evaluate every checkpoint of a GPU run on the GPU, then pick the ones worth a CPU benchmark.

  python scripts/gpu/eval.py runs/gpu/<run>
  python scripts/gpu/eval.py runs/gpu/<run> --select-only

Each checkpoint drives --realistic + --harsh laps (one lap per kart, no exploration
noise), several checkpoints side by side in one batch. Driver B (zero correction) drives
the same number of laps as the baseline. Results per checkpoint: ckpt/<steps>/eval_gpu.json
(a checkpoint is only driven once).

Picks (<run>/selection_gpu.json + .md), among checkpoints that finish at least
--min-finish of the realistic laps:
  fastest     lowest mean realistic lap time
  safest      most laps finished (both levels), then most room to the edge (median)
  consistent  smallest realistic lap-time spread
  smoothest   smallest change in correction step to step
  best_harsh  most harsh laps finished, then fastest at harsh
  no_compromise  no other checkpoint is at least as good on laps finished, lap time,
              edge room and smoothness and better on one of them
plus 'top': up to --top checkpoints for the CPU benchmark (every pick, then the
no-compromise set, then the best by laps finished and lap time).
The CPU benchmark (scripts/gps/benchmark_residual.py, seeds 0..29) has the final say.
"""
import argparse
import json
import pathlib
import pickle
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kart_gps.config import load_config  # noqa: E402
from kart_gpu.ppo import CLIP_OBS, EPS  # noqa: E402
from kart_gpu.residual import BatchResidual  # noqa: E402

LEVELS = {"realistic": 1, "harsh": 2}


def load_policy(ckpt, dev):
    """-> f(raw obs (n, 45)) -> deterministic actions (n, 2), as benchmark_residual._policy."""
    if ckpt is None:                                       # driver B
        return lambda o: torch.zeros(len(o), 2, device=dev)
    from stable_baselines3 import PPO
    model = PPO.load(pathlib.Path(ckpt) / "model.zip", device=dev,
                     custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0})
    with open(pathlib.Path(ckpt) / "vecnormalize.pkl", "rb") as f:
        vn = pickle.load(f)
    mean = torch.as_tensor(vn.obs_rms.mean, dtype=torch.float64, device=dev)
    std = torch.sqrt(torch.as_tensor(vn.obs_rms.var, dtype=torch.float64, device=dev) + EPS)
    pol = model.policy

    @torch.no_grad()
    def f(o):
        on = ((o.double() - mean) / std).clamp(-CLIP_OBS, CLIP_OBS).float()
        return pol._predict(on, deterministic=True).clamp(-1.0, 1.0)
    return f


def drive_group(env, policies, n_real, n_harsh):
    """One lap per kart; kart rows are [policy 0: realistic..., harsh..., policy 1: ...].
    -> list (per policy) of per-lap row dicts."""
    per = n_real + n_harsh
    lv = torch.tensor(([1] * n_real + [2] * n_harsh) * len(policies), device=env.dev)
    obs = env.reset(level=lv)
    N = env.N
    first = torch.zeros(N, dtype=torch.long, device=env.dev)
    keep = {k: torch.zeros(N, dtype=torch.float64, device=env.dev)
            for k in ("lap_time", "progress", "min_clearance")}
    act_sum = torch.zeros(N, dtype=torch.float64, device=env.dev)
    jump_sum = torch.zeros_like(act_sum)
    steps = torch.zeros_like(act_sum)
    prev = torch.zeros(N, 2, device=env.dev)
    while True:
        a = torch.cat([p(obs[i * per:(i + 1) * per]) for i, p in enumerate(policies)])
        alive = first == 0
        obs, _, done, info = env.step(a, autoreset=False)
        act_sum += torch.where(alive, a.abs().mean(1).double(), 0.0)
        jump_sum += torch.where(alive, (a - prev).abs().mean(1).double(), 0.0)
        steps += alive.double()
        prev = a
        new = alive & done
        first = torch.where(new, info["end"], first)
        for k in keep:
            keep[k] = torch.where(new, info[k].double(), keep[k])
        if bool((first > 0).all()):
            break
    from kart_gpu.sim import END
    F, LT, MC = first.cpu().numpy(), keep["lap_time"].cpu().numpy(), keep["min_clearance"].cpu().numpy()
    A, J, S = act_sum.cpu().numpy(), jump_sum.cpu().numpy(), steps.cpu().numpy()
    out = []
    for i in range(len(policies)):
        rows = []
        for j in range(i * per, (i + 1) * per):
            rows.append({"level": "realistic" if j - i * per < n_real else "harsh", "end": END[int(F[j])],
                         "lap_time": float(LT[j]) if F[j] == 1 else None, "min_clearance": float(MC[j]),
                         "mean_abs_action": float(A[j] / S[j]), "mean_abs_jump": float(J[j] / S[j])})
        out.append(rows)
    return out


def summarise(rows):
    out = {}
    for lv in LEVELS:
        g = [r for r in rows if r["level"] == lv]
        if not g:
            continue
        laps = [r["lap_time"] for r in g if r["end"] == "lap"]
        cl = [r["min_clearance"] for r in g]
        out[lv] = {"finished": len(laps), "of": len(g), "finish_rate": len(laps) / len(g),
                   "off_track": sum(r["end"] == "offtrack" for r in g),
                   "lap_mean": float(np.mean(laps)) if laps else None,
                   "lap_std": float(np.std(laps)) if laps else None,
                   "lap_p05": float(np.percentile(laps, 5)) if laps else None,
                   "clear_median": float(np.median(cl)), "clear_p05": float(np.percentile(cl, 5)),
                   "action": float(np.mean([r["mean_abs_action"] for r in g])),
                   "jump": float(np.mean([r["mean_abs_jump"] for r in g]))}
    return out


def select(results, min_finish, top):
    R = {s: v for s, v in results.items() if v.get("realistic", {}).get("lap_mean")}
    ok = [s for s in R if R[s]["realistic"]["finish_rate"] >= min_finish]
    fin = {s: sum(R[s][lv]["finished"] for lv in R[s]) / sum(R[s][lv]["of"] for lv in R[s]) for s in R}
    picks = {}
    if ok:
        picks["fastest"] = min(ok, key=lambda s: R[s]["realistic"]["lap_mean"])
        picks["consistent"] = min(ok, key=lambda s: R[s]["realistic"]["lap_std"])
        picks["smoothest"] = min(ok, key=lambda s: R[s]["realistic"]["jump"])
    if R:
        picks["safest"] = max(R, key=lambda s: (fin[s], R[s]["realistic"]["clear_median"]))
        H = [s for s in R if R[s].get("harsh", {}).get("lap_mean")]
        if H:
            picks["best_harsh"] = max(H, key=lambda s: (R[s]["harsh"]["finish_rate"], -R[s]["harsh"]["lap_mean"]))

    def sc(s):
        r = R[s]["realistic"]
        return (fin[s], -r["lap_mean"], r["clear_median"], -r["jump"])
    S = {s: sc(s) for s in R}
    front = sorted(s for s in R if not any(all(a >= b for a, b in zip(S[o], S[s])) and S[o] != S[s] for o in R))
    order = list(dict.fromkeys(list(picks.values()) + sorted(front, key=lambda s: (-fin[s], -S[s][1]))))
    rest = sorted(R, key=lambda s: (-round(fin[s] * 200) / 200, R[s]["realistic"]["lap_mean"]))
    order += [s for s in rest if s not in order]
    return picks, front, order[:top]


def fmt_row(name, v):
    r, h = v["realistic"], v.get("harsh")
    hs = f"{100 * h['finish_rate']:.1f}% {h['lap_mean'] or float('nan'):.2f} s" if h else "-"
    return (f"| {name} | {100 * r['finish_rate']:.1f}% | {r['lap_mean'] or float('nan'):.2f} s | ±{r['lap_std'] or 0:.2f} | "
            f"{r['clear_median']:+.3f} m | {r['jump']:.3f} | {hs} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--realistic", type=int, default=1024, help="realistic laps per checkpoint")
    ap.add_argument("--harsh", type=int, default=512, help="harsh laps per checkpoint")
    ap.add_argument("--group", type=int, default=8, help="checkpoints driven side by side")
    ap.add_argument("--min-finish", type=float, default=0.99, help="realistic finish share needed for the speed picks")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--select-only", action="store_true")
    args = ap.parse_args()
    run = pathlib.Path(args.run)
    dev = torch.device("cuda")
    ckpts = sorted((p for p in (run / "ckpt").glob("*") if (p / "saved").exists()), key=lambda p: int(p.name))
    todo = [] if args.select_only else [p for p in ckpts if not (p / "eval_gpu.json").exists()]
    base_path = run / "eval_gpu_B.json"
    need_b = not args.select_only and not base_path.exists()
    jobs = ([None] if need_b else []) + todo
    if jobs:
        cfg = load_config()
        per = args.realistic + args.harsh
        env = BatchResidual(cfg, per * min(args.group, len(jobs)), seed=4242)
        t0 = time.time()
        print(f"{len(todo)} checkpoints{' + driver B' if need_b else ''} x ({args.realistic} realistic + "
              f"{args.harsh} harsh laps), {env.N} karts at a time", flush=True)
        G = env.N // per
        for b in range(0, len(jobs), G):
            grp = jobs[b:b + G]
            pols = [load_policy(c, dev) for c in grp] + [load_policy(None, dev)] * (G - len(grp))
            res = drive_group(env, pols, args.realistic, args.harsh)
            for c, rows in zip(grp, res):
                s = summarise(rows)
                path = base_path if c is None else c / "eval_gpu.json"
                path.write_text(json.dumps({"summary": s, "laps": args.realistic + args.harsh}, indent=1))
                r, h = s["realistic"], s["harsh"]
                name = "driver B" if c is None else f"{int(c.name):,} steps"
                print(f"[{name}] realistic {100 * r['finish_rate']:.1f}% {r['lap_mean'] or float('nan'):.2f} s, "
                      f"harsh {100 * h['finish_rate']:.1f}% {h['lap_mean'] or float('nan'):.2f} s "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    results = {int(p.name): json.loads((p / "eval_gpu.json").read_text())["summary"]
               for p in ckpts if (p / "eval_gpu.json").exists()}
    if not results:
        print("nothing evaluated yet")
        return
    picks, front, top = select(results, args.min_finish, args.top)
    base = json.loads(base_path.read_text())["summary"] if base_path.exists() else None
    (run / "selection_gpu.json").write_text(json.dumps(
        {"picks": picks, "no_compromise": front, "top": top, "driver_B": base,
         "results": {str(k): v for k, v in sorted(results.items())}}, indent=1))
    head = ("| Checkpoint | Realistic finished | Mean lap | Spread | Edge room (median) | Jumpiness | Harsh |\n"
            "|---|---|---|---|---|---|---|")
    md = ["# GPU eval picks", "", head]
    if base:
        md.append(fmt_row("driver B (no network)", base))
    md += [fmt_row(f"**{a}** {s:,}", results[s]) for a, s in picks.items()]
    md += ["", f"## No-compromise set ({len(front)})", "", head] + [fmt_row(f"{s:,}", results[s]) for s in front]
    md += ["", f"## Top {len(top)} for the CPU benchmark", "", head] + [fmt_row(f"{s:,}", results[s]) for s in top]
    md += ["", "Benchmark them on the CPU:", "",
           "    python scripts/gps/benchmark_residual.py " + " ".join(f"{run.as_posix()}/ckpt/{s}" for s in top)
           + f" --out {run.as_posix()}/benchmark_top.json"]
    (run / "selection_gpu.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
