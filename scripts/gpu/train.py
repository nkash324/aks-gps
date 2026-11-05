#!/usr/bin/env python3
"""Train the residual driver on the GPU: thousands of karts at once (kart_gpu), own PPO
(kart_gpu/ppo.py, checked against SB3 in tests/test_gpu_ppo.py). Settings: gpu.yaml, plus
gps.yaml for the sim, the reward and the network.

  python scripts/gpu/train.py --name first
  python scripts/gpu/train.py --resume runs/gpu/<run>/last
  python scripts/gpu/train.py --name quick --steps 20000000 --envs 2048
  python scripts/gpu/train.py --config gpu_run2.yaml --name run2          # another settings file
  python scripts/gpu/train.py --config gpu_ft.yaml --name ft --init runs/gps/residual/<run>/ckpt/<steps>

Same curriculum as the CPU run (realistic only, then the harsh share ramps up) and the
same checkpoint layout: runs/gpu/<time>-<name>/ckpt/<steps>/{model.zip, vecnormalize.pkl,
saved}, loadable by scripts/gps/benchmark_residual.py and eval_residual.py. progress.csv
gets one row per update. --pause-at N saves and stops at N steps; Ctrl+C saves and stops.
"""
import argparse
import csv
import json
import pathlib
import sys
import time

import numpy as np
import torch
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kart_gps.config import load_config  # noqa: E402
from kart_gpu.ppo import GpuPPO  # noqa: E402
from kart_gpu.residual import BatchResidual  # noqa: E402


def harsh_share(steps, g, mix):
    """0 until curriculum_at, then a linear ramp to mix['harsh'] over ramp steps."""
    f = (steps - g["curriculum_at"]) / max(g["ramp"], 1)
    return mix["harsh"] * float(np.clip(f, 0.0, 1.0))


class Laps:
    """Counts finished laps on the GPU without syncing every step."""
    def __init__(self, dev):
        self.dev = dev
        self.clear()

    def clear(self):
        z = lambda: torch.zeros((), device=self.dev, dtype=torch.float64)
        self.n, self.lap, self.off, self.t = z(), z(), z(), z()

    def __call__(self, rew, done, info):
        e = info["end"]
        self.n += done.sum()
        self.lap += (e == 1).sum()
        self.off += (e == 2).sum()
        self.t += torch.where(e == 1, info["lap_time"].double(), 0.0).sum()

    def read(self):
        n, lap, off, t = (float(v) for v in (self.n, self.lap, self.off, self.t))
        self.clear()
        return {"laps_ended": n, "finished_share": lap / n if n else None,
                "offtrack_share": off / n if n else None, "lap_time": t / lap if lap else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="gpu")
    ap.add_argument("--steps", type=int, default=None, help="total (default gpu.yaml)")
    ap.add_argument("--envs", type=int, default=None, help="karts (default gpu.yaml)")
    ap.add_argument("--ckpt-every", type=int, default=None)
    ap.add_argument("--pause-at", type=int, default=None)
    ap.add_argument("--resume", default=None, help="checkpoint folder")
    ap.add_argument("--config", default="gpu.yaml", help="settings file in the repo root")
    ap.add_argument("--init", default=None, help="start a new run from this checkpoint's weights (step 0)")
    args = ap.parse_args()

    cfg = load_config()
    g = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    for k in ("steps", "envs", "ckpt_every"):
        if getattr(args, k) is not None:
            g[k] = getattr(args, k)
    ppo = {k: v for k, v in cfg["residual"]["ppo"].items() if k not in ("n_steps", "batch_size")}
    ppo.update(g["ppo"])
    mix = cfg["residual"]["mix"]

    if args.resume:
        ck = pathlib.Path(args.resume)
        run = ck.parent if ck.name == "last" else ck.parent.parent
    else:
        run = ROOT / "runs" / "gpu" / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.name}"
        run.mkdir(parents=True, exist_ok=True)
    (run / "finished").unlink(missing_ok=True)
    (run / f"settings-{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(
        json.dumps({"args": vars(args), "gpu": g, "ppo": ppo, "residual": cfg["residual"]}, indent=1))

    seed = g["seed"] + (int(time.time()) % 100_000 if args.resume else 0)   # new laps on a resume
    env = BatchResidual(cfg, g["envs"], seed=seed)
    algo = GpuPPO(env.n_obs, env.n_act, g["envs"], ppo, g["steps"], seed=g["seed"], resume=args.resume, init=args.init)
    env.harsh = harsh_share(algo.steps, g, mix)
    obs = algo.normalise_reset(env.reset())
    laps = Laps(env.dev)

    def save(name):
        algo.save(run / name)

    next_ck = (algo.steps // g["ckpt_every"] + 1) * g["ckpt_every"]
    csv_path = run / "progress.csv"
    new_csv = not csv_path.exists()
    fcsv = open(csv_path, "a", newline="")
    writer = None
    t0, s0 = time.time(), algo.steps
    print(f"training to {g['steps']:,} steps on {g['envs']} karts ({g['envs'] * ppo['n_steps']:,} steps per update) "
          f"-> {run}", flush=True)
    paused = False
    try:
        it = 0
        while algo.steps < g["steps"]:
            env.harsh = harsh_share(algo.steps, g, mix)
            tc = time.time()
            B, obs = algo.collect(env, obs, ppo["n_steps"], on_step=laps)
            tt = time.time()
            st = algo.train(B)
            now = time.time()
            row = {"steps": algo.steps, "minutes": (now - t0) / 60, "steps_per_s": (algo.steps - s0) / (now - t0),
                   "collect_s": tt - tc, "train_s": now - tt, "harsh_share": env.harsh, **laps.read(), **st}
            if writer is None:
                writer = csv.DictWriter(fcsv, fieldnames=list(row))
                if new_csv:
                    writer.writeheader()
            writer.writerow(row)
            fcsv.flush()
            it += 1
            if it % 10 == 1:
                f = lambda v, d=2: "-" if v is None else f"{v:.{d}f}"
                print(f"[{algo.steps:,}] {row['steps_per_s']:,.0f} steps/s | laps finished {f(row['finished_share'])}, "
                      f"off {f(row['offtrack_share'])}, lap {f(row['lap_time'])} s | harsh {env.harsh:.2f} | "
                      f"vf {st['vf']:.3f} kl {st['kl']:.4f} std {st['std']:.3f} ev {st['explained_var']:.2f}", flush=True)
            if algo.steps >= next_ck:
                next_ck += g["ckpt_every"]
                save(f"ckpt/{algo.steps}")
                save("last")
                print(f"[{algo.steps:,} steps, {(now - t0) / 60:.1f} min] saved ckpt/{algo.steps}", flush=True)
            if args.pause_at and algo.steps >= args.pause_at:
                print(f"[{algo.steps:,}] pause point reached", flush=True)
                paused = True
                break
    except KeyboardInterrupt:
        print("stopped by Ctrl+C", flush=True)
        paused = True
    if not (run / "ckpt" / str(algo.steps) / "saved").exists():
        save(f"ckpt/{algo.steps}")
    save("last")
    fcsv.close()
    if not paused:
        (run / "finished").write_text(str(algo.steps))
    print(f"done at {algo.steps:,} steps -> {run}" + (f"\nresume: python scripts/gpu/train.py --resume {run / 'last'}"
                                                      if paused else ""), flush=True)


if __name__ == "__main__":
    main()
