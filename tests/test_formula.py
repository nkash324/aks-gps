"""Formula driver C (numpy vs batched torch), the minimum-time line and the MPC smoke test."""
import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kart_gps.config import load_config, track_path  # noqa: E402
from kart_gps.drivers import Formula, build_line  # noqa: E402
from kart_gps.planner import mintime_line, speed_plan_ggv, track_bounds  # noqa: E402
from kart_gps.track import TrackMap  # noqa: E402


@pytest.fixture(scope="module")
def world():
    cfg = load_config()
    track = TrackMap(track_path(cfg))
    return cfg, track


@pytest.fixture(scope="module")
def mt(world):
    cfg, track = world
    line = mintime_line(track, cfg["kart"], cfg["planner"]["buffer"], sweeps=2, step0=0.4)
    speed_plan_ggv(line, cfg["kart"])
    return line


def test_mintime_inside_bounds(world, mt):
    cfg, track = world
    C, N, lo, hi = track_bounds(track, cfg["kart"], cfg["planner"]["buffer"])
    assert mt.offset.shape == lo.shape
    assert np.all(mt.offset >= lo - 1e-9) and np.all(mt.offset <= hi + 1e-9)
    base = build_line("optimised", track, cfg)
    speed_plan_ggv(base, cfg["kart"])
    assert mt.lap_estimate <= base.lap_estimate + 1e-6


def test_formula_numpy_matches_torch(world, mt):
    from kart_gpu.formula import BatchFormula
    cfg, track = world
    rng = np.random.default_rng(1)
    n = 64
    k = rng.integers(0, mt.n, n)
    ests = []
    for i in range(n):
        j = k[i]
        x, y = mt.pts[j] + rng.normal(0, 0.3, 2)
        ests.append({"x": x, "y": y, "yaw": mt.heading[j] + rng.normal(0, 0.15), "v": rng.uniform(0.5, 11),
                     "yaw_rate": rng.normal(0, 0.5), "ax": 0.0, "ay": 0.0, "rtk_fixed": bool(rng.random() > 0.2),
                     "gnss_age": float(rng.choice([0.0, 0.7])), "pos_std": 0.02})
    knobs = dict(tau=0.07, us=1.2, w_space=0.9, w_time=5.0, zeta=0.8, preview=0.4)
    want = []
    for e in ests:
        d = Formula(mt, cfg, **knobs)
        want.append(d.act(e))
    class Sim:  # the bits BatchFormula reads from a BatchLap
        dev, f, N = torch.device("cpu"), torch.float32, n
        kn, pp = cfg["kart"], cfg["drivers"]["pure_pursuit"]
        L = cfg["kart"]["wheelbase"]
        import math as _m
        _t = _m.tan(_m.radians(cfg["kart"]["max_steer_deg"]))
        max_delta = _m.atan(_t * L / (L + cfg["kart"]["kingpin_half"] * _t))
    drv = BatchFormula(Sim(), mt, mt.speed, **knobs)
    T = lambda key, dt=torch.float32: torch.tensor([e[key] for e in ests], dtype=dt)
    est = {k_: T(k_) for k_ in ("x", "y", "yaw", "v", "yaw_rate", "gnss_age")}
    est["rtk_fixed"] = T("rtk_fixed", torch.bool)
    speed, curv = drv(est)
    np.testing.assert_allclose(speed.numpy(), [w[0] for w in want], atol=2e-3)
    np.testing.assert_allclose(curv.numpy(), [w[1] for w in want], atol=2e-3)


def test_mpc_smoke(world, mt):
    """The MPC drives the first seconds of a lap on the CPU: valid commands, no end, makes progress."""
    from kart_gpu.mpc import BatchMPC
    from kart_gpu.sim import BatchLap
    cfg, track = world
    base = build_line("optimised", track, cfg)
    sim = BatchLap(cfg, 4, device="cpu", track=track, line=base, seed=3)
    est = sim.reset(level=torch.ones(4, dtype=torch.long))
    drv = BatchMPC(sim, mt, track, cfg, mt.speed, iters=4)
    for _ in range(40):
        speed, curv = drv(est)
        assert torch.isfinite(speed).all() and torch.isfinite(curv).all()
        assert curv.abs().max() <= drv.kmax + 1e-6 and speed.min() >= 0
        est, done, info = sim.step(speed, curv)
    assert not done.any() and info["progress"].min() > 3.0
