import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gps"))
from kart_checks import Rig, accel, brake, skidpad, steering  # noqa: E402
from kart_gps.config import load_config, sample_kart  # noqa: E402
from kart_gps.kart import G, Kart  # noqa: E402


def cfg():
    return load_config()


def test_static_loads_add_up():
    c = cfg()
    k = Kart(c["kart"], 0.002)
    assert math.isclose(sum(k.fz0), c["kart"]["mass"] * G, rel_tol=1e-9)
    assert math.isclose((k.fz0[0] + k.fz0[1]) / sum(k.fz0), c["kart"]["front_weight"])


def test_acceleration_and_top_speed():
    marks, vmax = accel(cfg())
    assert 2.5 < marks[10.0] < 3.5
    assert 11.5 < vmax <= 12.0


def test_braking():
    t_regen, _, peak_regen, _ = brake(cfg(), regen_only=True)
    t_both, d_both, peak_both, slip = brake(cfg(), regen_only=False)
    assert peak_regen < 4.2                  # regen is limited by the battery charge current
    assert t_both < t_regen
    assert 4.0 < peak_both < 5.5             # rear-only braking is rear-grip limited
    assert slip > -0.3                       # anti-lock keeps the rears turning
    assert d_both < 20.0


def test_steering_rate_and_ackermann():
    t, md, (left, right) = steering(cfg())
    assert 0.5 < t < 0.65                    # ~42 deg at 80 deg/s + 40 ms delay
    assert math.isclose(left, 25.0, abs_tol=0.01) and right < left


def test_skidpad_grip():
    g, understeer = skidpad(cfg())
    assert 0.9 < g < 1.25
    assert 1.0 < understeer < 1.5


def test_standstill_is_stable():
    r = Rig(cfg())
    for _ in range(1000):                    # 2 s, full lock, no throttle
        r.step((0.0, 0.0, 0.4))
    assert abs(r.kart.x - r.kart.lr) < 1e-6 and abs(r.kart.y) < 1e-6


def test_randomised_karts_stay_finite():
    c = cfg()
    rng = np.random.default_rng(0)
    for _ in range(5):
        r = Rig(c, sample_kart(c, rng))
        r.ll.set_target(12.0, 0.2)           # flat out on a 5 m circle: slides, must not blow up
        for _ in range(2500):
            r.step()
        k = r.kart
        assert all(np.isfinite([k.x, k.y, k.vx, k.vy, k.r, k.omega]))
