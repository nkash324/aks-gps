"""MPC driver for many karts at once (batched torch, runs on the CPU or a GPU).

Every decision (20 Hz) each kart solves a short optimal control problem in the path
frame of the minimum-time line and applies the first command:

  state    (s, e, epsi, v, k_act): arc length, lateral offset (left +), heading error,
           speed, actual path curvature (k_act = yaw_rate / v at the start)
  controls H steps of dt s: curvature command k_cmd (steering) and target speed v_t
  model    k_act follows k_cmd / us with a lag `lag` and a rate limit; v follows the speed
           loop a = clip(kp (v_t - v), -brake, accel(v)); ds = v cos(epsi) / (1 - kref e),
           de = v sin(epsi), depsi = v k_act - kref ds   (Euler, `sub` substeps per step)
  cost     -progress + w_e e^2 + w_b (edge room - kart half width - margin)^2 (soft barrier)
           + w_g friction-ellipse excess + w_v (v above plan speed)^2 + smoothness
           + a terminal penalty when v at the horizon end exceeds the plan speed there
  solve    `iters` Adam steps on normalised variables (clipped to their limits), warm-started
           from the previous solution shifted one step (first call: the feed-forward guess)
Then the same safety slowdowns as driver B (RTK float, GNSS lost).

Knobs (floats or (N,) tensors): w_e, w_b, margin, w_g, w_v, w_prog, w_sk, w_sv, lat, brake, us,
lag, vpre (see KNOBS). The plan-speed table is a (blocks, points) array with blk (N,) as in
BatchFormula.
"""
import math

import numpy as np
import torch

from kart_gps.planner import kart_half_width
from .sim import _wrap

KNOBS = {"w_e": 0.5, "w_b": 400.0, "margin": 0.10, "w_g": 4.0, "w_v": 2.0, "w_prog": 1.0, "w_sk": 2.0,
         "w_sv": 0.05, "lat": 7.3, "brake": 3.85, "us": 1.29, "lag": 0.05}
G = 9.81


def _smooth(a, n=5):
    k = np.ones(n) / n
    return sum(np.roll(a, i) * k[0] for i in range(-(n // 2), n // 2 + 1))


class BatchMPC:
    def __init__(self, sim, line, track, cfg, tables, blk=None, H=15, dt=0.1, sub=2, iters=12, lr=0.1,
                 **knobs):
        self.sim, self.dev, self.f, self.N = sim, sim.dev, sim.f, sim.N
        self.H, self.dt, self.sub, self.iters, self.lr = H, dt, sub, iters, lr
        T = lambda a: torch.tensor(np.asarray(a), device=self.dev, dtype=self.f)
        kart, c = cfg["kart"], cfg["control"]
        self.pp = sim.pp
        self.pts, self.s_pts = T(line.pts), T(line.s)
        self.tan, self.head = T(line.tangent), T(line.heading)
        self.n, self.length = line.n, line.length
        # uniform-s tables for interpolation: curvature, plan speed(s), room to the edges
        self.du = line.length / line.n
        su = np.arange(line.n) * self.du
        ip = lambda a: np.interp(su, np.r_[line.s, line.length], np.r_[a, a[0]])
        self.kref = T(ip(_smooth(line.curvature)))
        lat = np.array([track.project(x, y)[2] for x, y in line.pts])
        kk = np.array([track.nearest(x, y) for x, y in line.pts])
        self.room_l = T(ip(track.hw_left[kk] - lat))
        self.room_r = T(ip(track.hw_right[kk] + lat))
        tab = np.asarray(tables, dtype=float)
        tab = tab if tab.ndim == 2 else tab[None]
        self.tab = T(np.stack([ip(t) for t in tab]))
        self.blk = torch.zeros(self.N, dtype=torch.long, device=self.dev) if blk is None \
            else torch.as_tensor(blk, device=self.dev)
        self.k = {}
        for key, d in KNOBS.items():
            self.k[key] = torch.as_tensor(knobs.pop(key, d), device=self.dev, dtype=self.f).expand(self.N)
        self.vpre = int(knobs.pop("vpre", 1))
        self.delay_sub = int(knobs.pop("delay_sub", 0))     # substeps of the first step on the previous command
        assert not knobs, f"unknown knobs {list(knobs)}"
        L = kart["wheelbase"]
        self.kmax = math.tan(sim.max_delta) / L
        self.krate = math.radians(kart["steer_rate_deg"]) / L           # (1/m)/s
        self.kh = kart_half_width(kart)
        self.mass, self.roll = kart["mass"], kart["rolling_resistance"] * kart["mass"] * G
        self.cda = 0.5 * 1.2 * kart["drag_area"]
        self.f_drive, self.v_top, self.taper = kart["drive_force"], kart["top_speed"], kart["top_speed_taper"]
        self.kp = c["speed_kp"]
        self.vmax = 11.5
        self.win = torch.arange(-60, 61, device=self.dev)
        self.hint = None
        self.prev = None                       # (N, 2H) normalised solution
        self.prev_k = None
        self.win_ms = 0.0

    # ------------------------------------------------------------------ helpers
    def _interp(self, tab, s):
        x = torch.remainder(s, self.length) / self.du
        i0 = x.floor().long()
        fr = x - i0
        i0 = i0 % self.n
        return tab[i0] * (1 - fr) + tab[(i0 + 1) % self.n] * fr

    def _interp_blk(self, s):
        x = torch.remainder(s, self.length) / self.du
        i0 = x.floor().long()
        fr = x - i0
        i0 = i0 % self.n
        return self.tab[self.blk, i0] * (1 - fr) + self.tab[self.blk, (i0 + 1) % self.n] * fr

    def _nearest(self, x, y):
        if self.hint is None:
            d = (self.pts[None, :, 0] - x[:, None]) ** 2 + (self.pts[None, :, 1] - y[:, None]) ** 2
            self.hint = d.argmin(1)
            return self.hint
        idx = torch.remainder(self.hint[:, None] + self.win, self.n)
        q = self.pts[idx]
        d = (q[..., 0] - x[:, None]) ** 2 + (q[..., 1] - y[:, None]) ** 2
        self.hint = idx.gather(1, d.argmin(1, keepdim=True))[:, 0]
        return self.hint

    def _accel_max(self, v):
        f = self.f_drive * ((self.v_top - v) / self.taper).clamp(0.0, 1.0)
        return (f - self.roll - self.cda * v * v) / self.mass

    # ------------------------------------------------------------------ model + cost
    def rollout(self, z, st):
        """z (N, 2H) -> cost (N,). st: dict of initial s, e, epsi, v, ka."""
        kn, H, dt = self.k, self.H, self.dt
        kc = z[:, :H] * self.kmax                      # curvature commands
        vt = z[:, H:] * 10.0                           # target speeds
        s, e, ep, v, ka = st["s"], st["e"], st["epsi"], st["v"], st["ka"]
        h = dt / self.sub
        cost = torch.zeros_like(s)
        us, lag = kn["us"], kn["lag"]
        kprev, vprev = st["kc_prev"], st["vt_prev"]
        s0 = s
        for i in range(H):
            for j in range(self.sub):
                kr = self._interp(self.kref, s)
                # command delay: the first substep(s) still run on the command issued last time
                kcur = st["kc_prev"] if (i == 0 and j < self.delay_sub) else kc[:, i]
                dka = ((kcur / us - ka) / lag).clamp(-self.krate, self.krate)
                a = (self.kp * (vt[:, i] - v)).clamp(min=-kn["brake"] - 0.3)
                a = torch.minimum(a, self._accel_max(v))
                ds = v * torch.cos(ep) / (1 - kr * e).clamp(min=0.3)
                s = s + ds * h
                e = e + v * torch.sin(ep) * h
                ep = ep + (v * ka - kr * ds) * h
                ka = ka + dka * h
                v = (v + a * h).clamp(min=0.0)
            # step-end penalties
            vp = self._interp_blk(s)
            rl, rr = self._interp(self.room_l, s), self._interp(self.room_r, s)
            m = self.kh + kn["margin"]
            bar = (m - (rl - e)).clamp(min=0.0) ** 2 + (m - (rr + e)).clamp(min=0.0) ** 2
            ay = v * v * ka
            ex = ((ay / kn["lat"]) ** 2 + (a / kn["brake"]) ** 2 * (a < 0) +
                  (a / self._accel_max(v).clamp(min=1.0)) ** 2 * (a >= 0) - 1.0).clamp(min=0.0)
            cost = cost + kn["w_e"] * e * e + kn["w_b"] * bar + kn["w_g"] * ex * ex \
                + kn["w_v"] * (v - vp).clamp(min=0.0) ** 2 * dt
            cost = cost + kn["w_sk"] * ((kc[:, i] - kprev) / self.kmax) ** 2 + kn["w_sv"] * (vt[:, i] - vprev) ** 2
            kprev, vprev = kc[:, i], vt[:, i]
        prog = s - s0
        vend = self._interp_blk(s)
        term = kn["w_v"] * 5.0 * (v - vend).clamp(min=0.0) ** 2
        return cost - kn["w_prog"] * prog + term

    # ------------------------------------------------------------------ driver
    def __call__(self, est):
        kn, c, H = self.k, self.pp, self.H
        x, y, yaw, v, r = est["x"], est["y"], est["yaw"], est["v"], est["yaw_rate"]
        k = self._nearest(x, y)
        tx, ty = self.tan[k, 0], self.tan[k, 1]
        dx, dy = x - self.pts[k, 0], y - self.pts[k, 1]
        along = dx * tx + dy * ty
        e = -dx * ty + dy * tx
        st = {"s": self.s_pts[k] + along, "e": e, "epsi": _wrap(yaw - self.head[k]), "v": v.clamp(min=0.0),
              "ka": (r / v.clamp(min=1.0)) * (v > 1.0)}
        if self.prev is None:
            tt = (torch.arange(H, device=self.dev, dtype=self.f) + 0.5) * self.dt
            sp = st["s"][:, None] + v.clamp(min=2.0)[:, None] * tt[None]
            kr = self._interp(self.kref, sp)
            z0 = torch.cat([(kn["us"][:, None] * kr / self.kmax).clamp(-1, 1),
                            self._interp_blk_seq(sp) / 10.0], 1)
            self.prev_k = (kn["us"] * self._interp(self.kref, st["s"]))
            self.prev_v = v.clone()
        else:
            z0 = torch.cat([self.prev[:, 1:H], self.prev[:, H - 1:H], self.prev[:, H + 1:], self.prev[:, -1:]], 1)
        st["kc_prev"], st["vt_prev"] = self.prev_k, self.prev_v
        # optimise y with z = y * scale: one Adam step of lr moves 0.05*lr/... (curvature 0.05 1/m, speed 1 m/s per unit)
        scale = torch.cat([torch.full((H,), 0.05 / self.kmax), torch.full((H,), 0.1)]).to(self.dev)
        y = (z0 / scale).detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([y], lr=self.lr)
        lo = torch.cat([-torch.ones(H), torch.zeros(H)]).to(self.dev)
        hi = torch.cat([torch.ones(H), torch.full((H,), self.vmax / 10.0)]).to(self.dev)
        for _ in range(self.iters):
            opt.zero_grad()
            self.rollout(y * scale, st).sum().backward()
            opt.step()
            with torch.no_grad():
                y.copy_(torch.maximum(torch.minimum(y * scale, hi), lo) / scale)
        z = (y * scale).detach()
        self.prev = z
        self.prev_k = z[:, 0] * self.kmax
        self.prev_v = z[:, H] * 10.0
        curv = z[:, 0] * self.kmax
        speed = z[:, H + self.vpre] * 10.0
        speed = torch.where(est["gnss_age"] > 0.5, speed * c["lost_slowdown"],
                            torch.where(~est["rtk_fixed"], speed * c["float_slowdown"], speed))
        return speed, curv

    def _interp_blk_seq(self, sp):
        out = []
        for i in range(sp.shape[1]):
            out.append(self._interp_blk(sp[:, i]))
        return torch.stack(out, 1)
