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


def make_driver(name, track, cfg, lines=None):
    """'pp_centre' (A) or 'pp_optimised' (B)."""
    lines = lines or {}
    key = {"pp_centre": "centreline", "pp_optimised": "optimised"}[name]
    line = lines.get(key) or build_line(key, track, cfg)
    return PurePursuit(line, cfg["drivers"]["pure_pursuit"])
