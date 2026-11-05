"""The residual driver's env on the GPU: kart_gps/rl_env.py ResidualEnv for many karts.

Driver B drives, the network adds a correction to B's (target speed, curvature) at 20 Hz.
Same observation (45 numbers, same order), reward and lap rules as the CPU env, so a
policy trained here runs unchanged in the CPU benchmark.

    env = BatchResidual(cfg, n=4096)
    obs = env.reset()                                   # (N, 45) raw (not normalised)
    obs, reward, done, info = env.step(action)          # finished laps restart by themselves

Each new lap draws its sensor level: harsh with probability env.harsh (the curriculum
sets it), else realistic. eval mode (autoreset=False): fixed levels, one lap per kart.
"""
import math

import numpy as np
import torch

from .sim import BatchLap, _wrap


class BatchResidual:
    def __init__(self, cfg, n, device="cuda", dtype=torch.float32, seed=0, graph=True, track=None, line=None,
                 randomize=True, tile=1):
        self.cfg, self.N = cfg, n
        self.rc = rc = cfg["residual"]
        self.sim = sim = BatchLap(cfg, n, device=device, dtype=dtype, seed=seed, track=track, line=line,
                                   randomize=randomize, tile=tile)
        self.dev, self.f = sim.dev, sim.f
        self.g = torch.Generator(self.dev).manual_seed(seed + 1)
        self.graph = graph
        self.harsh = 0.0
        tr, ln = sim.track, sim.line
        T = lambda v: torch.tensor(np.asarray(v), device=self.dev, dtype=self.f)
        self.l_tan, self.l_head, self.l_curv = T(ln.tangent), T(ln.heading), T(ln.curvature)
        self.hw_l, self.hw_r = T(tr.hw_left), T(tr.hw_right)
        # room from each line point to the left / right surveyed edge (as rl_env)
        lat = np.array([tr.project(x, y)[2] for x, y in ln.pts])
        k = np.array([tr.nearest(x, y) for x, y in ln.pts])
        self.room_l, self.room_r = T(tr.hw_left[k] - lat), T(tr.hw_right[k] + lat)
        self.pre = T(rc["preview"])
        self.room_pre = T(rc["room_preview"])
        self.n_obs = 6 + 2 * len(rc["preview"]) + 2 * len(rc["room_preview"]) + 2 + 3 + 2 + 2
        self.n_act = 2

    # ------------------------------------------------------------------ API
    def _levels(self):
        """Per-lap sensor level: harsh (2) with probability self.harsh, else realistic (1)."""
        r = torch.rand(self.N, generator=self.g, device=self.dev)
        return torch.where(r < self.harsh, 2, 1)

    def reset(self, level=None):
        """Every kart on a new lap. level: None (draw from self.harsh), a name or (N,) indices."""
        sim = self.sim
        lv = self._levels() if level is None else level
        if self.graph and sim.graph is None:
            sim.reset(level=lv)
            est = sim.use_graph()
        else:
            est = sim.reset(level=lv)
        self.prev_a = torch.zeros(self.N, 2, device=self.dev, dtype=self.f)
        self.last_progress = torch.zeros(self.N, device=self.dev, dtype=self.f)
        self.base = sim.drive_b(est)
        return self._obs(est)

    def step(self, action, autoreset=True):
        """action (N, 2) in [-1, 1] -> (obs, reward, done, info). With autoreset the karts
        whose lap ended start a new one and obs is their first observation."""
        rc, rw, sim = self.rc, self.rc["reward"], self.sim
        a = action.to(self.f).clamp(-1.0, 1.0)
        speed_b, curv_b = self.base
        dv = a[:, 0] * rc["max_dv"]
        dv = torch.where(self.gnss_age > 0.5, dv.clamp(max=0.0), dv)
        speed = (speed_b + dv).clamp(min=0.0)
        curv = curv_b + a[:, 1] * rc["max_dk"]
        est, done, info = sim.step(speed, curv)
        info = {k: v.clone() for k, v in info.items()}       # the graph reuses its buffers
        ds = info["progress"] - self.last_progress
        clear = info["clear"]
        r = rw["progress"] * ds - rw["time"]
        r = r - rw["edge"] * ((rw["edge_margin"] - clear).clamp(min=0.0) / rw["edge_margin"]) ** 2
        r = r - rw["action"] * (a * a).sum(1) - rw["jerk"] * ((a - self.prev_a) ** 2).sum(1)
        end = info["end"]
        r = r + torch.where(end == 1, rw["lap"], torch.where(end > 0, -rw["fail"], 0.0))
        info["level"] = sim.level.clone()
        if autoreset:
            est = sim.reset(done, self._levels())
            self.prev_a = torch.where(done[:, None], torch.zeros_like(a), a)
            self.last_progress = torch.where(done, torch.zeros_like(ds), info["progress"])
        else:
            self.prev_a = a
            self.last_progress = info["progress"]
        self.base = sim.drive_b(est)
        return self._obs(est), r, done, info

    # ------------------------------------------------------------------ observation
    def _obs(self, est):
        sim = self.sim
        x, y, yaw, v = est["x"], est["y"], est["yaw"], est["v"]
        self.gnss_age = est["gnss_age"].clone()
        k = sim.s["bk"]
        tx, ty = self.l_tan[k, 0], self.l_tan[k, 1]
        lat = (x - sim.l_pts[k, 0]) * (-ty) + (y - sim.l_pts[k, 1]) * tx
        head = _wrap(yaw - self.l_head[k])
        idx = sim._ahead(k[:, None], self.pre[None, :])
        room = sim._ahead(k[:, None], self.room_pre[None, :])
        # the track's nearest sample: global search, as TrackMap.project without a hint
        xy = torch.stack([x, y], 1)
        kt = torch.cdist(xy, sim.t_pts, compute_mode="donot_use_mm_for_euclid_dist").argmin(1)
        tlat = ((xy - sim.t_pts[kt]) * sim.t_nrm[kt]).sum(1)
        speed_b, curv_b = self.base
        o = torch.cat([
            torch.stack([v / 10.0, est["yaw_rate"], est["ax"] / 5.0, est["ay"] / 5.0, lat, head], 1),
            self.l_curv[idx] * 10.0, (sim.l_speed[idx] - v[:, None]) / 5.0,
            self.room_l[room], self.room_r[room],
            torch.stack([self.hw_l[kt] - tlat, self.hw_r[kt] + tlat], 1),
            torch.stack([est["rtk_fixed"].to(self.f), est["gnss_age"].clamp(max=2.0), est["pos_std"].clamp(max=2.0)], 1),
            torch.stack([speed_b / 10.0, curv_b * 10.0], 1),
            self.prev_a,
        ], 1)
        return o
