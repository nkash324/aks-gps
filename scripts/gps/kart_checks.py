#!/usr/bin/env python3
"""Check the 2D kart against its spec: weight split, acceleration, top speed, braking
(regen only / regen + hydraulic brake), steering speed and cornering grip.

  python scripts/gps/kart_checks.py
  python scripts/gps/kart_checks.py --mujoco     # the same checks on the MuJoCo kart (mj_kart.py)
"""
import argparse
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from kart_gps.config import load_config  # noqa: E402
from kart_gps.control import LowLevel  # noqa: E402
from kart_gps.env import make_kart  # noqa: E402
from kart_gps.kart import G, Kart  # noqa: E402


class Rig:
    """Kart + low-level controller with perfect measurements, for the checks."""
    def __init__(self, cfg, kart=None):
        c = cfg["control"]
        self.kart = make_kart(cfg, kart or cfg["kart"], 1.0 / c["physics_hz"])
        self.ll = LowLevel(cfg["kart"], c)
        self.sub = c["physics_hz"] // c["lowlevel_hz"]
        self.cmd = (0.0, 0.0, 0.0)
        self.n = 0

    def step(self, override=None):
        k = self.kart
        if self.n % self.sub == 0:
            self.cmd = self.ll.update(k.vx, k.axle_speed, self.sub * k.dt)
        self.n += 1
        k.step(*(override or self.cmd))


def accel(cfg):
    r = Rig(cfg)
    r.ll.set_target(20.0, 0.0)
    marks, t = {}, 0.0
    while t < 15.0:
        r.step()
        t = r.kart.t
        for v in (5.0, 10.0, 11.5):
            if v not in marks and r.kart.vx >= v:
                marks[v] = t
    return marks, r.kart.vx


def brake(cfg, regen_only):
    r = Rig(cfg)
    r.kart.reset(0, 0, 0, speed=12.0)
    x0, peak, lock = r.kart.x, 0.0, 0.0
    r.ll.set_target(0.0, 0.0)
    while r.kart.vx > 0.05 and r.kart.t < 20:
        if regen_only:
            motor, _, steer = r.ll.update(r.kart.vx, r.kart.axle_speed, r.kart.dt)
            r.step((motor, 0.0, steer))
        else:
            r.step()
        peak = max(peak, -r.kart.ax)
        if r.kart.vx > 1:
            lock = min(lock, (r.kart.axle_speed - r.kart.vx) / r.kart.vx)
    return r.kart.t, r.kart.x - x0, peak, lock


def steering(cfg):
    r = Rig(cfg)
    r.kart.reset(0, 0, 0, speed=0.0)
    md = r.kart.max_delta
    r.step((0.0, 0.0, -md))
    while r.kart.delta > -md + 1e-4:
        r.step((0.0, 0.0, -md))
    t0 = r.kart.t
    while r.kart.delta < md - 1e-4:
        r.step((0.0, 0.0, md))
    return r.kart.t - t0, math.degrees(md), [math.degrees(a) for a in r.kart.wheel_angles()]


def skidpad(cfg, radius=8.0):
    """Drive an 8 m circle, adding steering as needed to hold the radius, and creep the
    speed up. The highest lateral acceleration while still on the circle = the grip
    limit. Also returns how much extra steering the kart needed at 3 m/s (understeer)."""
    r = Rig(cfg)
    r.kart.reset(0, 0, 0, speed=3.0)
    best, v_t, curv, extra_low = 0.0, 3.0, 1.0 / radius, None
    while v_t < 14.0:
        k = r.kart
        v_t += 0.1 * k.dt                       # 0.1 m/s per s, slow enough to be steady
        if k.vx > 1.0:                          # integrate steering until the radius is right
            curv += 2.0 * (1.0 / radius - k.r / k.vx) * k.dt
        r.ll.set_target(v_t, curv)
        r.step()
        if k.t > 5.0 and abs(k.vx / max(k.r, 1e-6) - radius) < 0.05 * radius:
            best = max(best, k.vx * k.r)
            if extra_low is None:
                extra_low = curv * radius
        if k.delta >= k.max_delta * 0.999:
            break
    return best / G, extra_low


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mujoco", action="store_true", help="check the MuJoCo kart instead of the 2D one")
    args = ap.parse_args()
    cfg = load_config(overrides={"physics": "mujoco"} if args.mujoco else None)
    p = cfg["kart"]
    k = Kart(p, 0.002)
    print("physics:", cfg["physics"])
    print(f"mass {p['mass']:.0f} kg, front {p['front_weight']:.0%}, static tyre loads "
          + " ".join(f"{f:.0f}" for f in k.fz0) + " N (FL FR RL RR)")
    marks, vmax = accel(cfg)
    print("0-5 m/s {:.2f} s, 0-10 m/s {:.2f} s, 0-11.5 m/s {:.2f} s, speed after 15 s {:.2f} m/s".format(
        marks.get(5.0, float("nan")), marks.get(10.0, float("nan")), marks.get(11.5, float("nan")), vmax))
    for regen_only in (True, False):
        t, d, peak, lock = brake(cfg, regen_only)
        name = "regen only        " if regen_only else "regen + hydraulic "
        print(f"brake 12->0 {name}: {t:.2f} s, {d:.1f} m, peak {peak:.2f} m/s^2, worst rear slip {lock:.2f}")
    t, md, wa = steering(cfg)
    print(f"steering lock to lock: {t:.2f} s (bicycle angle +-{md:.1f} deg, wheels at full left "
          f"{wa[0]:.1f}/{wa[1]:.1f} deg)")
    g, extra = skidpad(cfg)
    print(f"skidpad (8 m radius): max lateral {g:.2f} g; at low speed it needs {extra:.2f}x the "
          "geometric steering (>1 = understeer)")


if __name__ == "__main__":
    main()
