#!/usr/bin/env python3
"""Build the minimum-time line once and cache it (runs/formula/mintime_line.npz).

Reports the planned lap of the GGV speed plan on the min-curvature line vs the mintime
line. Later scripts load the cache with common.load_mintime().

  python scripts/formula/build_line.py [--lat 7.3 --brake 3.85]
"""
import argparse
import json
import time

import numpy as np

from common import ROOT, setup
from kart_gps.planner import mintime_line, min_curvature_line, speed_plan_ggv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--brake", type=float, default=None)
    ap.add_argument("--out", default=str(ROOT / "runs" / "formula" / "mintime_line.npz"))
    a = ap.parse_args()
    cfg, track, base = setup()
    kart, buf = cfg["kart"], cfg["planner"]["buffer"]
    plan = {k: v for k, v in (("lat", a.lat), ("brake", a.brake)) if v is not None}
    mc = min_curvature_line(track, kart, buf)
    speed_plan_ggv(mc, kart, **plan)
    t = time.time()
    ml = mintime_line(track, kart, buf, plan=plan, verbose=True)
    print(f"min-curvature planned lap {mc.lap_estimate:.3f} s; mintime {ml.lap_estimate:.3f} s "
          f"({time.time() - t:.0f} s to build)")
    import pathlib
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, pts=ml.pts, offset=ml.offset, plan=json.dumps(plan))
    json.dump({"mincurv_lap": mc.lap_estimate, "mintime_lap": ml.lap_estimate, "plan": plan},
              open(str(a.out).replace(".npz", ".json"), "w"))


if __name__ == "__main__":
    main()
