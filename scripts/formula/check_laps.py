#!/usr/bin/env python3
"""Check that separate runs with the same seed / N / levels drive identical laps.

Driver B is driven two ways on the same laps: through BatchLap + drive_b, and through
BatchResidual with zero actions (how G2 is driven). Per-lap times must match exactly,
otherwise the benchmark has to use the tile=K block approach (scripts/gpu/tune_b.py).

  python scripts/formula/check_laps.py --laps 20
"""
import argparse
import sys

import numpy as np
import torch

from common import DEVICE, level_vector, make_sim, rows, run_laps, setup  # noqa: E402
from kart_gpu.residual import BatchResidual  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laps", type=int, default=20, help="per sensor level")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    cfg, track, base = setup()
    lv = level_vector({l: args.laps for l in ("perfect", "realistic", "harsh")})
    N = len(lv)
    sim, est = make_sim(cfg, track, base, N, args.seed, lv)
    a = run_laps(sim, est, sim.drive_b)

    env = BatchResidual(cfg, N, device=DEVICE, graph=DEVICE == "cuda", seed=args.seed, track=track, line=base)
    env.reset(level=lv.to(env.dev))
    first = torch.zeros(N, dtype=torch.long, device=env.dev)
    keep = torch.zeros(N, dtype=torch.float64, device=env.dev)
    zero = torch.zeros(N, 2, device=env.dev)
    while True:
        alive = first == 0
        _, _, done, info = env.step(zero, autoreset=False)
        new = alive & done
        first = torch.where(new, info["end"], first)
        keep = torch.where(new, info["lap_time"].double(), keep)
        if bool((first > 0).all()):
            break
    b = keep.cpu().numpy()
    ok = (a["end"] == 1) & (first.cpu().numpy() == 1)
    diff = np.abs(a["lap_time"][ok] - b[ok])
    print(f"{ok.sum()} laps finished by both; max |time difference| = {diff.max():.6f} s; "
          f"ends equal: {bool((a['end'] == first.cpu().numpy()).all())}")
    sys.exit(0 if diff.max() < 1e-4 else 1)


if __name__ == "__main__":
    main()
