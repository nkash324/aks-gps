"""Driver C ("formula") for many karts at once: the batched twin of kart_gps.drivers.Formula.

Every knob (tau, us, w_space, w_time, zeta, preview) is a float or an (N,) tensor, so
different karts can drive different candidates in one batch. The speed plan is a table
(blocks, line points); blk (N,) says which table row each kart uses (per-candidate
lat / brake limits).

    drv = BatchFormula(sim, line, tables, blk, tau=..., us=...)
    speed, curv = drv(est)
"""
import math

import numpy as np
import torch

from .sim import _wrap

KNOBS = {"tau": 0.08, "us": 1.29, "w_space": 1.0, "w_time": 6.0, "zeta": 0.9, "preview": 0.5}


class BatchFormula:
    def __init__(self, sim, line, tables, blk=None, hint=None, **knobs):
        self.sim, self.dev, f = sim, sim.dev, sim.f
        self.f, self.N = f, sim.N
        T = lambda a: torch.tensor(np.asarray(a), device=self.dev, dtype=f)
        self.pts, self.s = T(line.pts), T(line.s)
        self.tan, self.head, self.curv = T(line.tangent), T(line.heading), T(line.curvature)
        self.n, self.length = line.n, line.length
        tab = np.asarray(tables, dtype=float)
        self.tab = T(tab if tab.ndim == 2 else tab[None])
        self.blk = torch.zeros(self.N, dtype=torch.long, device=self.dev) if blk is None \
            else torch.as_tensor(blk, device=self.dev)
        self.k = {}
        for key, default in KNOBS.items():
            v = knobs.pop(key, default)
            self.k[key] = torch.as_tensor(v, device=self.dev, dtype=f).expand(self.N)
        assert not knobs, f"unknown knobs {list(knobs)}"
        kart = sim.kn
        t_in = math.tan(math.radians(kart["max_steer_deg"]))
        self.kmax = math.tan(sim.max_delta) / sim.L
        self.pp = sim.pp
        self.win = torch.arange(-60, 61, device=self.dev)
        self.hint = hint                      # (N,) nearest-index memory; None = global search first call

    def reset_hint(self):
        self.hint = None

    def nearest(self, x, y):
        if self.hint is None:
            d = (self.pts[None, :, 0] - x[:, None]) ** 2 + (self.pts[None, :, 1] - y[:, None]) ** 2
            self.hint = d.argmin(1)
            return self.hint
        idx = torch.remainder(self.hint[:, None] + self.win, self.n)
        q = self.pts[idx]
        d = (q[..., 0] - x[:, None]) ** 2 + (q[..., 1] - y[:, None]) ** 2
        self.hint = idx.gather(1, d.argmin(1, keepdim=True))[:, 0]
        return self.hint

    def ahead(self, k, dist):
        target = torch.remainder(self.s[k] + dist, self.length)
        return torch.remainder(torch.searchsorted(self.s, target.contiguous()), self.n)

    def __call__(self, est):
        kn, c = self.k, self.pp
        x, y, yaw, v, r = est["x"], est["y"], est["yaw"], est["v"], est["yaw_rate"]
        tau = kn["tau"]
        yaw_m = yaw + 0.5 * r * tau
        x, y = x + v * tau * torch.cos(yaw_m), y + v * tau * torch.sin(yaw_m)
        yaw = yaw + r * tau
        k = self.nearest(x, y)
        tx, ty = self.tan[k, 0], self.tan[k, 1]
        e = (x - self.pts[k, 0]) * (-ty) + (y - self.pts[k, 1]) * tx
        epsi = _wrap(yaw - self.head[k])
        w = torch.minimum(kn["w_space"], kn["w_time"] / v.clamp(min=1.0))
        curv = kn["us"] * self.curv[k] - w * w * e - 2.0 * kn["zeta"] * w * epsi
        curv = curv.clamp(-self.kmax, self.kmax)
        speed = self.tab[self.blk, self.ahead(k, v * kn["preview"])]
        speed = torch.where(est["gnss_age"] > 0.5, speed * c["lost_slowdown"],
                            torch.where(~est["rtk_fixed"], speed * c["float_slowdown"], speed))
        return speed, curv
