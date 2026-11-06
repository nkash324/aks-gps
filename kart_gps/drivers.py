"""Drivers: take the EKF estimate, return (target speed m/s, curvature 1/m) at 20 Hz.

  PurePursuit(line): steer at a point a speed-dependent distance ahead on the line;
  drive at the line's planned speed. "A" follows the centreline, "B" the optimised line.
Both slow down when RTK drops to float or the GNSS goes quiet (the safety layer the
real kart will run around any driver).
"""
import math

import numpy as np

from .planner import centreline, min_curvature_line, speed_plan


def build_line(name, track, cfg):
    """'centreline' or 'optimised', with its speed plan, from the nominal kart."""
    pl, kart = cfg["planner"], cfg["kart"]
    line = centreline(track) if name == "centreline" else min_curvature_line(track, kart, pl["buffer"])
    speed_plan(line, kart, pl["lat_accel"], pl["brake_decel"], pl["top_speed"])
    return line


class PurePursuit:
    def __init__(self, line, cfg):
        self.line = line
        self.c = cfg
        self.k = None

    def reset(self):
        self.k = None

    def lookahead(self, v):
        c = self.c
        return min(c["max"], max(c["min"], c["gain"] * v + c["base"]))

    def act(self, est):
        line, c = self.line, self.c
        x, y, yaw, v = est["x"], est["y"], est["yaw"], est["v"]
        self.k = line.nearest(x, y, self.k)
        ld = self.lookahead(v)
        j = line.ahead(self.k, ld)
        dx, dy = line.pts[j, 0] - x, line.pts[j, 1] - y
        cy, sy = math.cos(yaw), math.sin(yaw)
        lx, ly = cy * dx + sy * dy, -sy * dx + cy * dy
        curvature = 2.0 * ly / max(lx * lx + ly * ly, 1e-6)
        speed = float(line.speed[line.ahead(self.k, v * c["speed_preview"])])
        if est["gnss_age"] > 0.5:
            speed *= c["lost_slowdown"]
        elif not est["rtk_fixed"]:
            speed *= c["float_slowdown"]
        return speed, curvature


FORMULA_DEFAULTS = {"tau": 0.08, "us": 1.29, "w_space": 1.0, "w_time": 6.0, "zeta": 0.9, "preview": 0.5}


class Formula:
    """Driver C: feedforward + feedback steering on the minimum-time line.

    Predict the pose `tau` s ahead (actuator lag), find the nearest line point, then
      curvature = us * kappa_ref - k_e * e - k_psi * epsi         (e left +, epsi = yaw - line heading)
      k_e = w^2, k_psi = 2 zeta w, w = min(w_space, w_time / max(v, 1))
    and drive the line's GGV speed plan `preview` s ahead. `us` is the understeer factor
    (the kart needs ~1.29x the kinematic steering). Same safety slowdowns as driver B.
    line: a Line with .speed set (planner.speed_plan_ggv); knobs: FORMULA_DEFAULTS keys.
    """
    def __init__(self, line, cfg, **knobs):
        self.line = line
        self.c = cfg["drivers"]["pure_pursuit"]
        self.k = {**FORMULA_DEFAULTS, **knobs}
        kart = cfg["kart"]
        L = kart["wheelbase"]
        t_in = math.tan(math.radians(kart["max_steer_deg"]))
        self.L = L
        self.kmax = math.tan(math.atan(t_in * L / (L + kart["kingpin_half"] * t_in))) / L
        self.hint = None

    def reset(self):
        self.hint = None

    def act(self, est):
        line, kn, c = self.line, self.k, self.c
        x, y, yaw, v, r = est["x"], est["y"], est["yaw"], est["v"], est["yaw_rate"]
        tau = kn["tau"]
        yaw_m = yaw + 0.5 * r * tau
        x, y = x + v * tau * math.cos(yaw_m), y + v * tau * math.sin(yaw_m)
        yaw = yaw + r * tau
        self.hint = line.nearest(x, y, self.hint)
        k = self.hint
        tx, ty = line.tangent[k]
        e = (x - line.pts[k, 0]) * (-ty) + (y - line.pts[k, 1]) * tx
        epsi = (yaw - line.heading[k] + math.pi) % (2 * math.pi) - math.pi
        w = min(kn["w_space"], kn["w_time"] / max(v, 1.0))
        curv = kn["us"] * line.curvature[k] - w * w * e - 2.0 * kn["zeta"] * w * epsi
        curv = max(-self.kmax, min(self.kmax, curv))
        speed = float(line.speed[line.ahead(k, v * kn["preview"])])
        if est["gnss_age"] > 0.5:
            speed *= c["lost_slowdown"]
        elif not est["rtk_fixed"]:
            speed *= c["float_slowdown"]
        return speed, curv


def make_driver(name, track, cfg, lines=None):
    """'pp_centre' (A) or 'pp_optimised' (B)."""
    lines = lines or {}
    key = {"pp_centre": "centreline", "pp_optimised": "optimised"}[name]
    line = lines.get(key) or build_line(key, track, cfg)
    return PurePursuit(line, cfg["drivers"]["pure_pursuit"])
