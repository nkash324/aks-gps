"""Shared helpers for the formula-driver scripts: build sims, run laps, summarise.

Works on CUDA (graph replay) and on the CPU. Laps depend only on (seed, number of karts,
sensor levels), never on the driving, so separate runs with the same seed / N / levels
drive identical laps (checked by scripts/formula/check_laps.py).
"""
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu"))
from kart_gps.config import load_config, track_path  # noqa: E402
from kart_gps.drivers import build_line  # noqa: E402
from kart_gps.track import TrackMap  # noqa: E402
from kart_gpu.sim import END, BatchLap  # noqa: E402

LEVELS = ("perfect", "realistic", "harsh")
BENCH_SEED = 20261029          # the published benchmark laps' seed
TUNE_SEED = 7_000_000          # tuning laps, apart from the benchmark
G2 = ROOT / "runs" / "gpu" / "20261029-234306-run2" / "ckpt" / "460062720"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def setup():
    cfg = load_config()
    track = TrackMap(track_path(cfg))
    base = build_line("optimised", track, cfg)
    return cfg, track, base


def level_vector(laps, K=1):
    """laps {level: n} -> (per * K,) level indices, the per-block pattern repeated K times."""
    return torch.tensor([LEVELS.index(l) for l in LEVELS for _ in range(laps.get(l, 0))] * K)


def make_sim(cfg, track, base, n, seed, lv, tile=1):
    sim = BatchLap(cfg, n, device=DEVICE, track=track, line=base, seed=seed, tile=tile)
    est = sim.reset(level=lv.to(sim.dev))
    if sim.dev.type == "cuda":
        est = sim.use_graph()       # restarts the laps: same draws
    return sim, est


def run_laps(sim, est, act, per=None, K=1, lv=None, progress=None):
    """Drive every kart one lap. act(est) -> (target speed, curvature) every decision.
    Returns a numpy dict per kart: end (codes), lap_time, progress, min_clearance, max_lat_g."""
    N = sim.N
    first = torch.zeros(N, dtype=torch.long, device=sim.dev)
    keep = {k: torch.zeros(N, dtype=torch.float64, device=sim.dev)
            for k in ("lap_time", "progress", "min_clearance", "max_lat_g")}
    n = 0
    while True:
        alive = first == 0
        speed, curv = act(est)
        est, done, info = sim.step(speed, curv)
        new = alive & done
        first = torch.where(new, info["end"], first)
        for k in keep:
            keep[k] = torch.where(new, info[k].double(), keep[k])
        n += 1
        if progress and n % 200 == 0:
            progress(n, int((first > 0).sum()), N)
        if bool((first > 0).all()):
            break
    out = {k: v.cpu().numpy() for k, v in keep.items()}
    out["end"] = first.cpu().numpy()
    return out


def rows(res, lv, sl=slice(None)):
    """per-kart result dict -> list of row dicts (one per kart in the slice)."""
    lvs = lv.numpy()[sl]
    F = res["end"][sl]
    out = []
    for j in range(len(F)):
        out.append({"level": LEVELS[lvs[j]], "end": END[int(F[j])],
                    "lap_time": float(res["lap_time"][sl][j]) if F[j] == 1 else None,
                    "progress": float(res["progress"][sl][j]),
                    "min_clearance": float(res["min_clearance"][sl][j]),
                    "max_lat_g": float(res["max_lat_g"][sl][j])})
    return out


def summary(rs):
    out = {}
    for lv in LEVELS:
        g = [r for r in rs if r["level"] == lv]
        if not g:
            continue
        laps = [r["lap_time"] for r in g if r["end"] == "lap"]
        cl = [r["min_clearance"] for r in g]
        out[lv] = {"runs": len(g), "finished": len(laps), "finish_rate": len(laps) / len(g),
                   "off_track": sum(r["end"] == "offtrack" for r in g),
                   "lap_mean": float(np.mean(laps)) if laps else None,
                   "lap_median": float(np.median(laps)) if laps else None,
                   "lap_std": float(np.std(laps)) if laps else None,
                   "clear_median": float(np.median(cl)), "clear_p05": float(np.percentile(cl, 5)),
                   "max_lat_g_median": float(np.median([r["max_lat_g"] for r in g])),
                   "max_lat_g_p95": float(np.percentile([r["max_lat_g"] for r in g], 95))}
    return out


def bootstrap_ci(x, n=5000, seed=0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def load_mintime(cfg, path=None, plan=None):
    """The cached minimum-time Line (pts from build_line.py) with a GGV speed plan."""
    from kart_gps.planner import Line, speed_plan_ggv
    z = np.load(path or ROOT / "runs" / "formula" / "mintime_line.npz")
    ln = Line(z["pts"], "mintime")
    speed_plan_ggv(ln, cfg["kart"], **(plan or {}))
    return ln


def plan_table(line, cfg, lat, brake, cache={}):
    """GGV speed table (line points,) for these limits (cached)."""
    from kart_gps.planner import speed_plan_ggv
    key = (id(line), round(lat, 4), round(brake, 4))
    if key not in cache:
        cache[key] = speed_plan_ggv(line, cfg["kart"], lat=lat, brake=brake, store=False)
    return cache[key]


def c_laps(cfg, track, base, line, cands, laps, seed, mk=None, log=None):
    """Driver C for each candidate (dict: lat, brake + formula knobs) on the SAME laps.
    laps {level: n}. -> list (one per candidate) of row lists. mk(sim, line, cand_list, per)
    can build another batched driver (default BatchFormula)."""
    from kart_gpu.formula import BatchFormula
    per = sum(laps.values())
    K = len(cands)
    lv = level_vector(laps, K)
    sim, est = make_sim(cfg, track, base, per * K, seed, lv, tile=K)
    blk = torch.arange(per * K) // per
    tabs = np.array([plan_table(line, cfg, c["lat"], c["brake"]) for c in cands])
    knobs = {k: torch.tensor([c[k] for c in cands], dtype=sim.f)[blk]
             for k in ("tau", "us", "w_space", "w_time", "zeta", "preview") if k in cands[0]}
    drv = (mk or (lambda *a, **k: BatchFormula(*a, **k)))(sim, line, tabs, blk, **knobs)
    res = run_laps(sim, est, drv, progress=log)
    return [rows(res, lv, slice(i * per, (i + 1) * per)) for i in range(K)]
