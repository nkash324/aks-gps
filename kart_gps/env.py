"""One timed lap on the surveyed track: kart physics + sensors + EKF + low-level control.

A driver gets the EKF estimate (never the true pose) and the map, and every 1/20 s sends
(target speed m/s, curvature 1/m):

    env = LapEnv(cfg, level="realistic", seed=3)
    est = env.reset()
    while True:
        est, done, info = env.step(*driver.act(est))
        if done: break
    info["end"], info["lap_time"]

Rules:
  - standing start with the rear axle on the start line, one full lap
  - off track = any part of any tyre (seen from above) past a surveyed edge
  - also ends on a rollover, getting stuck (5 s under 0.2 m/s) or 150 s
Each run draws its own kart (gps.yaml 'randomize') and sensor errors from the seed.
"""
import math

import numpy as np

from .config import sample_kart, track_path
from .control import LowLevel
from .estimator import Ekf
from .kart import Kart
from .sensors import Sensors
from .track import TrackMap

TIME_LIMIT = 150.0


def make_kart(cfg, p, dt):
    """The kart physics gps.yaml 'physics' asks for: 2d (kart.Kart) or mujoco (mj_kart.MjKart)."""
    if cfg.get("physics", "2d") == "mujoco":
        from .mj_kart import MjKart
        return MjKart(p, dt, cfg, lines=cfg["mujoco"].get("lines", False))
    return Kart(p, dt)


class LapEnv:
    def __init__(self, cfg, level=None, seed=None, track=None, randomize=True, record=False):
        self.cfg = cfg
        self.track = track or TrackMap(track_path(cfg))
        self.level = level or cfg["sensors"]["level"]
        self.randomize = randomize
        self.record = record
        c = cfg["control"]
        self.dt = 1.0 / c["physics_hz"]
        self.n_sub = c["physics_hz"] // c["rate_hz"]
        self.ll_every = c["physics_hz"] // c["lowlevel_hz"]
        self.sensors = Sensors(cfg["sensors"], self.level)
        self.ekf = Ekf()
        self.ll = LowLevel(cfg["kart"], c)
        self.seed = seed

    def reset(self, seed=None):
        seed = self.seed if seed is None else seed
        self.rng = np.random.default_rng(seed)
        self.kart_params = sample_kart(self.cfg, self.rng if self.randomize else None)
        self.kart = make_kart(self.cfg, self.kart_params, self.dt)
        x, y, yaw = self.track.start_pose
        self.kart.reset(x, y, yaw)
        self.sensors.reset(self.rng)
        self.ll.reset()
        self.ekf.ready = False
        self.cmd = (0.0, 0.0, 0.0)
        self.wheel_speed = 0.0
        self.k_idx, self.progress = 0, 0.0
        self.k_idx, s0, _ = self.track.project(*self.kart.rear_axle)
        self.s_last = s0
        self.min_clear = math.inf
        self.max_lat_g = 0.0
        self.n = 0
        self.slow_since = None
        self.trace = []
        # wait for the first GNSS fix, then start the filter facing along the track
        while not self.ekf.ready:
            self._physics_step(start_filter=True)
        self.t0 = self.kart.t                 # lap clock starts now
        return self.ekf.state(self.kart.t)

    def _physics_step(self, start_filter=False):
        k = self.kart
        self.sensors.sample(k)
        for msg in self.sensors.receive(k.t):
            if msg["type"] == "wheel":
                self.wheel_speed = msg["speed"]
            if start_filter and not self.ekf.ready and msg["type"] == "gnss":
                self.ekf.reset(msg["x"], msg["y"], self.track.start_pose[2], k.t)
            self.ekf.handle(msg, k.t)
        if self.n % self.ll_every == 0 and self.ekf.ready:
            est = self.ekf.state(k.t)
            self.cmd = self.ll.update(est["v"], self.wheel_speed, self.ll_every * self.dt, est["ay"])
        self.n += 1
        k.step(*self.cmd)

    def step(self, speed, curvature):
        """Hold (target speed, curvature) for 1/20 s. -> (estimate, done, info)."""
        self.ll.set_target(speed, curvature)
        k, tr = self.kart, self.track
        end = None
        for i in range(self.n_sub):
            self._physics_step()
            if (i + 1) % self.ll_every == 0:
                self.k_idx, s, _ = tr.project(*k.rear_axle, hint=self.k_idx)
                ds = (s - self.s_last + tr.length / 2) % tr.length - tr.length / 2
                self.progress += ds
                self.s_last = s
                clear = float(tr.clearance(k.tyre_corners(), self.k_idx).min())
                self.min_clear = min(self.min_clear, clear)
                self.max_lat_g = max(self.max_lat_g, abs(k.ay) / 9.81)
                if clear < 0.0:
                    end = "offtrack"
                elif k.rolled:
                    end = "rollover"
                elif self.progress >= tr.length:
                    end = "lap"
                if end:
                    break
        if end is None:
            if k.speed < 0.2 and k.t > 5.0:
                self.slow_since = self.slow_since if self.slow_since is not None else k.t
                if k.t - self.slow_since > 5.0:
                    end = "stuck"
            else:
                self.slow_since = None
            if k.t > TIME_LIMIT:
                end = "timeout"
        est = self.ekf.state(k.t)
        if self.record:
            self.trace.append((k.t, *k.rear_axle, k.yaw, k.speed, est["x"], est["y"], est["v"],
                               speed, curvature, k.ax, k.ay, self.cmd[1]))
        info = {"t": k.t, "progress": self.progress, "end": end,
                "lap_time": k.t - self.t0 if end == "lap" else None,
                "min_clearance": self.min_clear, "max_lat_g": self.max_lat_g,
                "true": {"x": k.rear_axle[0], "y": k.rear_axle[1], "yaw": k.yaw, "v": k.speed}}
        return est, end is not None, info
