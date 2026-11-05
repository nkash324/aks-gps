"""Many laps at once on the GPU: kart physics, sensors, EKF, low-level control, lap rules
and driver B, each a batched copy of its kart_gps counterpart (kart.py, sensors.py,
estimator.py, control.py, env.py, drivers.py). One row of every tensor is one kart.

    sim = BatchLap(cfg, n=4096)
    est = sim.reset()                         # every kart on the start line
    while True:
        est, done, info = sim.step(*sim.drive_b(est))
        ...                                   # read info["end"] etc. where done
        est = sim.reset(done)                 # new laps for the karts that finished

Timing is the CPU sim's, to the physics step: 500 Hz physics, IMU + wheel + EKF +
low-level loop every 5 steps, GNSS every 25 steps (arriving 25 steps later at the
realistic / harsh levels), one decision every 25 steps. Every lap starts on the step
after its first GNSS fix arrives, so all karts share the same phase within a decision
and the substep schedule is the same for the whole batch.

Differences from the CPU sim, all too small to change a lap:
  - float32 by default (dtype=torch.float64 for checks)
  - nearest-point searches stay in their +-60 sample window (the CPU falls back to a
    global search when the window misses, which never happens during a lap)
  - the order of IMU / wheel / GNSS messages that arrive on the same step is fixed
  - the per-lap random draws come from torch, not numpy: the same distributions, not
    the same laps as a CPU seed (benchmarks always run on the CPU)
"""
import math

import numpy as np
import torch

from kart_gps.config import track_path
from kart_gps.drivers import build_line
from kart_gps.env import TIME_LIMIT
from kart_gps.estimator import GATE, SIG_POS_FIXED, SIG_POS_FLOAT
from kart_gps.track import TrackMap

G = 9.81
AIR = 1.2
V_LOW = 1.0
BRAKE_DEADBAND = 40.0
RING = 32                 # steering delay buffer, physics steps (delay is at most RING - 1)
LEVELS = ("perfect", "realistic", "harsh")
END = {0: None, 1: "lap", 2: "offtrack", 3: "rollover", 4: "stuck", 5: "timeout"}


def _wrap(a):
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def _clip(x, lo, hi):
    """Clamp with tensor or float bounds."""
    lo = lo if torch.is_tensor(lo) else torch.full_like(x, lo)
    hi = hi if torch.is_tensor(hi) else torch.full_like(x, hi)
    return torch.maximum(lo, torch.minimum(hi, x))


class BatchLap:
    def __init__(self, cfg, n, device="cuda", dtype=torch.float32, track=None, line=None,
                 randomize=True, seed=0, tile=1):
        """tile: the batch holds `tile` blocks of n / tile karts that get the same random
        draws (same karts, sensor errors and GPS events), so different drivers in different
        blocks drive the very same laps."""
        self.cfg, self.N, self.dev, self.f = cfg, n, torch.device(device), dtype
        assert n % tile == 0
        self.tile = tile
        self.randomize = randomize and cfg.get("randomize", {}).get("enabled", False)
        self.g = torch.Generator(self.dev).manual_seed(seed)
        c, sc = cfg["control"], cfg["sensors"]
        self.dt = 1.0 / c["physics_hz"]
        self.n_sub = c["physics_hz"] // c["rate_hz"]
        self.ll_every = c["physics_hz"] // c["lowlevel_hz"]
        self.gnss_every = c["physics_hz"] // sc["gnss"]["rate_hz"]
        self.lat_steps = int(round(sc["gnss"]["latency"] / self.dt))
        # the schedule this port relies on (true for gps.yaml)
        assert self.n_sub == self.gnss_every and self.n_sub % self.ll_every == 0
        assert self.lat_steps in (0, self.gnss_every), "GNSS latency must be 0 or one GNSS period"
        assert c["physics_hz"] // sc["imu"]["rate_hz"] == self.ll_every
        assert c["physics_hz"] // sc["wheel"]["rate_hz"] == self.ll_every
        self.sc, self.cc = sc, c

        # ---- kart constants (nominal) and which parameters are drawn per lap
        p = cfg["kart"]
        self.kn = p
        self.rand = {k: float(v) for k, v in cfg.get("randomize", {}).items() if k != "enabled"} \
            if self.randomize else {}
        for k in ("wheelbase", "front_weight", "kingpin_half", "max_steer_deg", "track_front",
                  "track_rear", "cg_height", "tyre_front", "tyre_rear"):
            assert k not in self.rand, f"randomising {k} is not supported on the GPU"
        self.L = p["wheelbase"]
        self.lr = p["front_weight"] * self.L
        self.lf = self.L - self.lr
        self.h = p["cg_height"]
        self.Rr = p["tyre_rear"]["diameter"] / 2
        self.tf, self.tr = p["track_front"], p["track_rear"]
        self.kh = p["kingpin_half"]
        self.Cs = p["shape"]
        self.Bs = math.tan(math.pi / (2 * self.Cs))
        t_in = math.tan(math.radians(p["max_steer_deg"]))
        self.max_delta = math.atan(t_in * self.L / (self.L + self.kh * t_in))
        self.held = p.get("brake_hold_at_bite", True)
        T = lambda v: torch.tensor(v, device=self.dev, dtype=self.f)
        self.wx = T([self.lf, self.lf, -self.lr, -self.lr])
        self.wy = T([self.tf / 2, -self.tf / 2, self.tr / 2, -self.tr / 2])
        # tyre rectangle corners in the tyre frame: (4 tyres, 4 corners)
        hl = [p["tyre_front"]["diameter"] / 2] * 2 + [p["tyre_rear"]["diameter"] / 2] * 2
        hw = [p["tyre_front"]["width"] / 2] * 2 + [p["tyre_rear"]["width"] / 2] * 2
        self.ca_ = T([[a, a, -a, -a] for a in hl])
        self.cb_ = T([[b, -b, -b, b] for b in hw])

        # ---- track and the optimised line, on the device
        self.track = track or TrackMap(track_path(cfg))
        self.line = line or build_line("optimised", self.track, cfg)
        tr, ln = self.track, self.line
        self.t_pts, self.t_s = T(tr.pts), T(tr.s)
        self.t_tan, self.t_nrm = T(tr.tangent), T(tr.normal)
        self.t_len, self.t_n = tr.length, tr.n
        self.outer, self.inner = T(tr.outer), T(tr.inner)
        self.near_out = torch.tensor(tr._near_out, device=self.dev)
        self.near_in = torch.tensor(tr._near_in, device=self.dev)
        self.side_out, self.side_in = tr._side_out, tr._side_in
        self.l_pts, self.l_s, self.l_speed = T(ln.pts), T(ln.s), T(ln.speed)
        self.l_len, self.l_n = ln.length, ln.n
        self.pp = cfg["drivers"]["pure_pursuit"]
        sx, sy, syaw = tr.start_pose
        self.start = (sx, sy, syaw)
        k0, s0, _ = tr.project(sx, sy)            # rear axle sits on the start pose
        self.k0, self.s0 = k0, s0
        self.win60 = torch.arange(-60, 61, device=self.dev)
        self.win16 = torch.arange(-16, 16, device=self.dev)
        lv = sc["levels"]
        self.lv_scale = T([float(lv[n]["scale"]) for n in LEVELS])
        self.lv_lat = torch.tensor([bool(lv[n]["latency"]) for n in LEVELS], device=self.dev)
        self.eye6 = torch.eye(6, device=self.dev, dtype=self.f)
        self.P0 = T([0.05, 0.05, math.radians(3.0), 0.1, 0.01, 0.1]) ** 2
        self.s = {}
        self.graph = None
        self.level = torch.ones(n, dtype=torch.long, device=self.dev)    # realistic

    # ================================================================== helpers
    def _rand(self, *shape):
        return self._tiled(torch.rand, shape)

    def _randn(self, *shape):
        return self._tiled(torch.randn, shape)

    def _tiled(self, fn, shape):
        if self.tile == 1:
            return fn(*shape, generator=self.g, device=self.dev, dtype=self.f)
        x = fn(shape[0] // self.tile, *shape[1:], generator=self.g, device=self.dev, dtype=self.f)
        return x.repeat(self.tile, *([1] * (len(shape) - 1)))

    def _full(self, v, dtype=None):
        return torch.full((self.N,), v, device=self.dev, dtype=dtype or self.f)

    def _P(self, s, key):
        """Kart parameter: per-kart tensor if randomised, else the nominal float."""
        return s["P_" + key] if "P_" + key in s else self.kn[key]

    def _time(self, s):
        return s["n"].to(self.f) * self.dt

    def _wait(self, per_minute, k):
        """Exponential wait (s) until the next event at per_minute * k events per minute."""
        rate = per_minute * k / 60.0
        u = self._rand(self.N)
        w = -torch.log1p(-u) / torch.where(rate > 0, rate, torch.ones_like(rate))
        return torch.where(rate > 0, w, torch.full_like(w, math.inf))

    # ================================================================== reset
    def reset(self, mask=None, level=None):
        """New laps for the karts in mask (all if None). level: per-kart level index
        (0 perfect, 1 realistic, 2 harsh) or a name; default = their current level."""
        if isinstance(level, str):
            level = torch.full((self.N,), LEVELS.index(level), device=self.dev, dtype=torch.long)
        if self.graph is not None:
            self._rmask.copy_(mask if mask is not None else torch.ones_like(self._rmask))
            self._rlevel.copy_(level if level is not None else self.level)
            self.reset_graph.replay()
            return self._rout
        if level is not None:
            self.level = torch.where(mask, level, self.level) if mask is not None else level.clone()
        fresh = self._fresh(self.level)
        if not self.s:
            self.s = fresh
        else:                                   # in place: the buffers a captured graph uses stay put
            new = dict(_leaves(fresh))
            for name, old in _leaves(self.s):
                v = new[name]
                old.copy_(v if mask is None else torch.where(mask.view(-1, *([1] * (v.dim() - 1))), v, old))
        return self._est(self.s)

    def _fresh(self, level):
        """A just-started lap for every kart: the CPU LapEnv.reset() up to and including
        the step its first GNSS fix arrives on (the 'ready' step)."""
        N, dt, sc = self.N, self.dt, self.sc
        s = {}
        k = self.lv_scale[level]
        lat = self.lv_lat[level]
        s["k"], s["lat"] = k, lat
        # ---- kart parameters
        for key, frac in self.rand.items():
            s["P_" + key] = self.kn[key] * (1.0 - frac + 2.0 * frac * self._rand(N))
        m = self._P(s, "mass") * torch.ones(N, device=self.dev, dtype=self.f)
        wf = self.kn["front_weight"]
        s["fz0"] = torch.stack([m * G * wf / 2] * 2 + [m * G * (1 - wf) / 2] * 2, 1)
        mu = self._P(s, "mu") * torch.ones_like(m)
        s["mu4"] = torch.stack([mu * self._P(s, "front_grip")] * 2 + [mu] * 2, 1)
        s["steer_rate"] = math.radians(1.0) * self._P(s, "steer_rate_deg") * torch.ones_like(m)
        dly = self._P(s, "steer_delay") * torch.ones_like(m)
        s["D"] = torch.round(dly / dt).long().clamp(1, RING - 1)
        # ---- kart on the grid, rear axle on the start pose
        sx, sy, syaw = self.start
        z = torch.zeros(N, device=self.dev, dtype=self.f)
        s.update(x=z + sx + self.lr * math.cos(syaw), y=z + sy + self.lr * math.sin(syaw), yaw=z + syaw,
                 vx=z.clone(), vy=z.clone(), r=z.clone(), omega=z.clone(), delta=z.clone(),
                 f_motor=z.clone(), ax=z.clone(), ay=z.clone(),
                 brake_pos=z + (0.0 if self.held else -self.kn["brake_dead_time"]),
                 rolled=torch.zeros(N, dtype=torch.bool, device=self.dev),
                 ring=torch.zeros(N, RING, device=self.dev, dtype=self.f))
        # ---- sensor errors for this lap
        g, mo, im, wh = sc["gnss"], sc["map_offset"], sc["imu"], sc["wheel"]
        s["shift"] = (self._rand(N, 2) * 2 - 1) * mo["shift"] * k[:, None]
        s["rot"] = math.radians(1.0) * (self._rand(N) * 2 - 1) * mo["rotate_deg"] * k
        s["gyro_bias"] = self._randn(N) * im["gyro_bias"] * k
        s["accel_bias"] = self._randn(N, 2) * im["accel_bias"] * k[:, None]
        s["wheel_scale"] = 1.0 + self._randn(N) * wh["scale_error"] * k
        s["wander"] = torch.zeros(N, 2, device=self.dev, dtype=self.f)
        s["float_until"], s["drop_until"] = z - 1.0, z - 1.0
        s["float_off"] = torch.zeros(N, 2, device=self.dev, dtype=self.f)
        s["next_float"] = self._wait(g["float_rate"], k)
        s["next_drop"] = self._wait(g["dropout_rate"], k)
        # ---- GNSS at t = 0, and (with latency) the one taken on the ready step
        s["n"] = torch.zeros(N, dtype=torch.long, device=self.dev)
        g0 = self._gnss_sample(s)
        n_ready = torch.where(lat, self.lat_steps, 0)
        s["n"] = n_ready
        g1 = self._gnss_sample(s)                 # perfect karts: k = 0, nothing changes
        s["pend"] = {kk: (v & lat if kk == "valid" else v) for kk, v in g1.items()}
        imu = self._imu_sample(s)
        wheel = self._wheel_sample(s)
        t = self._time(s)
        # ---- EKF starts at the first fix, facing along the track
        X = torch.stack([g0["x"], g0["y"], z + syaw, z, z, z], 1)
        s.update(X=X, P=torch.diag_embed(self.P0.expand(N, 6)).clone(), ekf_t=t.clone(),
                 gyro=z.clone(), acc0=z.clone(), acc1=z.clone(),
                 last_fix=t.clone(), last_fixed=t.clone(), last_pos_ok=t.clone(),
                 fixed=torch.ones(N, dtype=torch.bool, device=self.dev))
        self._gnss_update(s, g0, s["X"][:, :4].clone(), t)        # history empty: h = X
        # the IMU and wheel readings of the ready step come after the fix (latency only;
        # without latency they arrive before the filter exists and are ignored)
        s["gyro"] = torch.where(lat, imu[0], z)
        s["acc0"] = torch.where(lat, imu[1], z)
        s["acc1"] = torch.where(lat, imu[2], z)
        self._wheel_update(s, wheel, lat)
        s["wheel_speed"] = torch.where(lat, wheel, z)
        # ---- low-level loop (reset, then the ready step's update) and the ready step
        s.update(v_target=z.clone(), curv=z.clone(), integ=z.clone(), brake_scale=z + 1.0, drive_scale=z + 1.0)
        s["cmd"] = self._lowlevel(s)
        self._kart_step(s, *s["cmd"])
        s["n"] = s["n"] + 1
        t0 = self._time(s)
        s["t0"] = t0
        self._predict(s, t0)                      # LapEnv.reset() returns ekf.state(t0)
        s["hist"] = s["X"][:, :4].clone()          # what the pending fix gets compared with
        # ---- lap bookkeeping
        s["k_idx"] = torch.full((N,), self.k0, dtype=torch.long, device=self.dev)
        s["s_last"] = z + self.s0
        s["progress"] = z.clone()
        s["min_clear"] = z + math.inf
        s["clear_last"] = z + math.inf
        s["max_lat_g"] = z.clone()
        s["slow_since"] = z - 1.0
        s["end"] = torch.zeros(N, dtype=torch.long, device=self.dev)
        s["t_end"] = z.clone()
        # driver B has no hint yet: global nearest on the line
        d = torch.cdist(s["X"][:, :2], self.l_pts, compute_mode="donot_use_mm_for_euclid_dist")
        s["bk"] = d.argmin(1)
        return s

    # ================================================================== sensors
    def _imu_sample(self, s):
        im, k = self.sc["imu"], s["k"]
        gz = s["r"] + s["gyro_bias"] + self._randn(self.N) * im["gyro_noise"] * k
        ax = s["ax"] + s["accel_bias"][:, 0] + self._randn(self.N) * im["accel_noise"] * k
        ay = s["ay"] + s["accel_bias"][:, 1] + self._randn(self.N) * im["accel_noise"] * k
        return gz, ax, ay

    def _wheel_sample(self, s):
        wh = self.sc["wheel"]
        v = s["omega"] * self.Rr * s["wheel_scale"] + self._randn(self.N) * wh["noise"] * s["k"]
        return v.clamp(min=0.0)

    def _gnss_sample(self, s):
        """Sensors.sample()'s GNSS part at time n*dt; updates the error state in s."""
        g, k, N = self.sc["gnss"], s["k"], self.N
        t = self._time(s)
        dtg = 1.0 / g["rate_hz"]
        a = math.exp(-dtg / 10.0)
        s["wander"] = s["wander"] * a + self._randn(N, 2) * g["pos_wander"] * k[:, None] * math.sqrt(1 - a * a)
        hit = t >= s["next_float"]
        fu = t + g["float_duration"][0] + (g["float_duration"][1] - g["float_duration"][0]) * self._rand(N)
        ang = self._rand(N) * (2 * math.pi)
        fo = g["float_error"] * k[:, None] * torch.stack([torch.cos(ang), torch.sin(ang)], 1)
        nf = fu + self._wait(g["float_rate"], k)
        s["float_until"] = torch.where(hit, fu, s["float_until"])
        s["float_off"] = torch.where(hit[:, None], fo, s["float_off"])
        s["next_float"] = torch.where(hit, nf, s["next_float"])
        hit = t >= s["next_drop"]
        du = t + g["dropout_duration"][0] + (g["dropout_duration"][1] - g["dropout_duration"][0]) * self._rand(N)
        nd = du + self._wait(g["dropout_rate"], k)
        s["drop_until"] = torch.where(hit, du, s["drop_until"])
        s["next_drop"] = torch.where(hit, nd, s["next_drop"])
        valid = ~(t < s["drop_until"])
        fl = t < s["float_until"]
        rx, ry = self._rear(s)
        noise = self._randn(N, 2) * g["pos_noise"] * k[:, None] * torch.where(fl, 3.0, 1.0)[:, None]
        err = s["shift"] + s["wander"] + noise + torch.where(fl[:, None], s["float_off"], torch.zeros_like(noise))
        cr, sr = torch.cos(s["rot"]), torch.sin(s["rot"])
        c, sn = torch.cos(s["yaw"]), torch.sin(s["yaw"])
        vy_r = s["vy"] - s["r"] * self.lr
        ve = s["vx"] * c - vy_r * sn + self._randn(N) * g["vel_noise"] * k
        vn = s["vx"] * sn + vy_r * c + self._randn(N) * g["vel_noise"] * k
        return {"valid": valid, "fixed": ~fl, "x": cr * rx - sr * ry + err[:, 0],
                "y": sr * rx + cr * ry + err[:, 1], "ve": ve, "vn": vn}

    # ================================================================== EKF
    def _predict(self, s, t):
        dt = (t - s["ekf_t"]).clamp(min=0.0)
        X = s["X"]
        x, y, yaw, v, bg, ba = X.unbind(1)
        w = s["gyro"] - bg
        a = s["acc0"] - ba
        c, sn = torch.cos(yaw), torch.sin(yaw)
        s["X"] = torch.stack([x + v * c * dt, y + v * sn * dt, _wrap(yaw + w * dt),
                              (v + a * dt).clamp(min=0.0), bg, ba], 1)
        F = self.eye6.expand(self.N, 6, 6).clone()
        F[:, 0, 2], F[:, 0, 3] = -v * sn * dt, c * dt
        F[:, 1, 2], F[:, 1, 3] = v * c * dt, sn * dt
        F[:, 2, 4] = -dt
        F[:, 3, 5] = -dt
        one = torch.ones_like(v)
        q = torch.stack([0.05 + 0.02 * v, 0.05 + 0.02 * v, 0.01 * one, 0.5 * one, 1e-4 * one, 1e-2 * one], 1) ** 2 \
            * dt[:, None]
        s["P"] = F @ s["P"] @ F.transpose(1, 2) + torch.diag_embed(q)
        s["ekf_t"] = torch.where(dt > 0, t, s["ekf_t"])

    def _update(self, s, idx, z, h, R, mask, angle=False):
        """Ekf._update for the karts in mask: measurement of the state components in slice idx."""
        X, P = s["X"], s["P"]
        m = idx.stop - idx.start
        y = z - h
        if angle:
            y = torch.cat([_wrap(y[:, :1]), y[:, 1:]], 1)
        S = P[:, idx][:, :, idx] + torch.diag_embed(R)
        if m == 1:
            Si = 1.0 / S
        else:
            det = S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] * S[:, 1, 0]
            Si = torch.stack([torch.stack([S[:, 1, 1], -S[:, 0, 1]], 1),
                              torch.stack([-S[:, 1, 0], S[:, 0, 0]], 1)], 1) / det[:, None, None]
        d2 = (y[:, None, :] @ Si @ y[:, :, None])[:, 0, 0]
        ok = mask & (d2 <= GATE ** 2 * m)
        K = P[:, :, idx] @ Si
        Xn = X + (K @ y[:, :, None])[:, :, 0]
        Xn = torch.cat([Xn[:, :2], _wrap(Xn[:, 2:3]), Xn[:, 3:4].clamp(min=0.0), Xn[:, 4:]], 1)
        KH = torch.zeros_like(P)
        KH[:, :, idx] = K
        A = self.eye6 - KH
        Pn = A @ P @ A.transpose(1, 2) + K @ torch.diag_embed(R) @ K.transpose(1, 2)
        s["X"] = torch.where(ok[:, None], Xn, X)
        s["P"] = torch.where(ok[:, None, None], Pn, P)
        return ok

    def _wheel_update(self, s, speed, mask):
        v = s["X"][:, 3]
        sig = 0.05 + 0.02 * v + 0.3 * (s["acc0"].abs() - 2.0).clamp(min=0.0)
        self._update(s, slice(3, 4), speed[:, None], v[:, None], (sig ** 2)[:, None], mask)

    def _gnss_update(self, s, msg, hist, t):
        """A GNSS message (msg, valid where it exists) arriving at t; hist = the filter's
        (x, y, yaw, v) at the time the fix was taken."""
        d, fx = msg["valid"], msg["fixed"]
        s["fixed"] = torch.where(d, fx, s["fixed"])
        s["last_fix"] = torch.where(d, t, s["last_fix"])
        s["last_fixed"] = torch.where(d & fx, t, s["last_fixed"])
        px, py, pyaw, pv = hist.unbind(1)
        sig = torch.where(fx, SIG_POS_FIXED, SIG_POS_FLOAT).to(self.f)
        z = torch.stack([msg["x"], msg["y"]], 1)
        ok = self._update(s, slice(0, 2), z, torch.stack([px, py], 1), torch.stack([sig ** 2] * 2, 1), d)
        # a long run of rejected fixes: trust the GNSS again
        relock = d & ~ok & fx & (t - s["last_pos_ok"] > 1.0)
        X = s["X"]
        X0 = torch.where(relock, msg["x"] + (X[:, 0] - px), X[:, 0])
        X1 = torch.where(relock, msg["y"] + (X[:, 1] - py), X[:, 1])
        s["X"] = torch.cat([X0[:, None], X1[:, None], X[:, 2:]], 1)
        P = s["P"]
        P00 = torch.where(relock, sig ** 2, P[:, 0, 0])
        P11 = torch.where(relock, sig ** 2, P[:, 1, 1])
        P = P.clone()
        P[:, 0, 0], P[:, 1, 1] = P00, P11
        s["P"] = P
        ok = ok | relock
        s["last_pos_ok"] = torch.where(ok, t, s["last_pos_ok"])
        speed = torch.hypot(msg["ve"], msg["vn"])
        fast = d & (speed > 3.0)
        cog = torch.atan2(msg["vn"], msg["ve"])
        sig_cog = torch.maximum(torch.full_like(speed, 0.04), 0.1 / speed.clamp(min=1e-6))
        self._update(s, slice(2, 4), torch.stack([cog, speed], 1), torch.stack([pyaw, pv], 1),
                     torch.stack([sig_cog ** 2, torch.full_like(speed, 0.01)], 1), fast, angle=True)

    def _est(self, s):
        """Ekf.state() without the predict (the filter is already at the current time)."""
        t = self._time(s)
        X, P = s["X"], s["P"]
        return {"x": X[:, 0], "y": X[:, 1], "yaw": X[:, 2], "v": X[:, 3],
                "yaw_rate": s["gyro"] - X[:, 4], "ax": s["acc0"] - X[:, 5], "ay": s["acc1"],
                "pos_std": torch.sqrt(torch.maximum(P[:, 0, 0], P[:, 1, 1]).clamp(min=0.0)),
                "rtk_fixed": s["fixed"] & (t - s["last_fixed"] < 0.5),
                "gnss_age": t - s["last_fix"]}

    # ================================================================== low-level control
    def _lowlevel(self, s):
        """LowLevel.update (nominal kart numbers) -> (motor N, brake 0..1, steering rad)."""
        k, c = self.kn, self.cc
        m = k["mass"]
        dt = self.ll_every * self.dt
        v, axle, lat = s["X"][:, 3], s["wheel_speed"], s["acc1"]
        vt = s["v_target"]
        err = vt - v
        a = _clip(c["speed_kp"] * err + s["integ"], -c["max_brake_decel"], 4.0)
        free = (a > -c["max_brake_decel"]) & (a < 4.0)
        integ = torch.where(free, _clip(s["integ"] + c["speed_ki"] * err * dt, -1.0, 1.0), s["integ"])
        stop = (vt < 0.05) & (v < 0.3)
        s["integ"] = torch.where(stop, torch.zeros_like(integ), integ)
        resist = k["rolling_resistance"] * m * G + 0.5 * AIR * k["drag_area"] * v * v
        force = m * a + resist
        slip = (axle - v) / v.clamp(min=1.0)
        s["brake_scale"] = torch.where(slip < -c["abs_slip"], (s["brake_scale"] * 0.6).clamp(min=0.2),
                                       (s["brake_scale"] + 0.1).clamp(max=1.0))
        s["drive_scale"] = torch.where(slip > c["abs_slip"], (s["drive_scale"] * 0.6).clamp(min=0.2),
                                       (s["drive_scale"] + 0.1).clamp(max=1.0))
        room = torch.sqrt((c["grip_accel"] ** 2 - lat * lat).clamp(min=0.0)) * m
        motor_d = torch.minimum(force, resist + room) * s["drive_scale"]
        regen_max = torch.minimum(torch.full_like(axle, k["regen_force"]),
                                  k["regen_power"] / axle.clamp(min=0.1)) * axle.clamp(max=1.0)
        motor_b = -torch.minimum(-force, regen_max) * s["brake_scale"]
        rest = -force - regen_max
        hold = (vt < 0.05) & (v < 1.0)
        brake_b = torch.where((rest > BRAKE_DEADBAND) | hold,
                              (rest.clamp(min=0.0) / k["brake_force"]).clamp(max=1.0) * s["brake_scale"],
                              torch.zeros_like(rest))
        brake_b = torch.where(hold, torch.ones_like(brake_b), brake_b)
        drive = force >= 0.0
        motor = torch.where(drive, motor_d, motor_b)
        brake = torch.where(drive, torch.zeros_like(brake_b), brake_b)
        steer = _clip(torch.atan(self.L * s["curv"]), -self.max_delta, self.max_delta)
        return motor, brake, steer

    # ================================================================== kart physics
    def _rear(self, s):
        return s["x"] - self.lr * torch.cos(s["yaw"]), s["y"] - self.lr * torch.sin(s["yaw"])

    def _wheel_angles(self, delta):
        t = torch.tan(delta)
        L, kh = self.L, self.kh
        return torch.atan(L * t / (L - kh * t)), torch.atan(L * t / (L + kh * t))

    def _tyre(self, fz, kappa, tan_a, mu, fz0):
        kn = self.kn
        sx = kappa / kn["slip_ratio_peak"]
        sy = tan_a / math.tan(math.radians(kn["slip_angle_peak_deg"]))
        sl = torch.sqrt(sx * sx + sy * sy)
        small = sl < 1e-9
        ss = torch.where(small, torch.ones_like(sl), sl)
        mu_l = mu * (1.0 - kn["load_sensitivity"] * (fz / fz0 - 1.0))
        f = mu_l * fz * torch.sin(self.Cs * torch.atan(self.Bs * sl))
        f = torch.where(small | (fz <= 0.0), torch.zeros_like(f), f)
        return f * sx / ss, f * sy / ss

    def _kart_step(self, s, motor_cmd, brake_cmd, steer_cmd):
        """Kart.step for every kart; s['n'] = this physics step's index."""
        p, dt = self.kn, self.dt
        m = self._P(s, "mass")
        # ---- actuators
        cmd = _clip(steer_cmd, -self.max_delta, self.max_delta)
        n = s["n"]
        ring = s["ring"].scatter(1, (n % RING)[:, None], cmd[:, None])
        s["ring"] = ring
        target = ring.gather(1, ((n - s["D"]) % RING)[:, None])[:, 0]
        st = s["steer_rate"] * dt
        s["delta"] = s["delta"] + torch.maximum(-st, torch.minimum(st, target - s["delta"]))
        delta = s["delta"]
        omega = s["omega"]
        v_wheel = omega * self.Rr
        f_drive_max = self._P(s, "drive_force") * ((p["top_speed"] - v_wheel) / p["top_speed_taper"]).clamp(0.0, 1.0)
        f_regen_max = torch.minimum(torch.full_like(v_wheel, p["regen_force"]),
                                    self._P(s, "regen_power") / v_wheel.clamp(min=0.1)) * (v_wheel / 1.0).clamp(max=1.0)
        f_cmd = torch.maximum(-f_regen_max, torch.minimum(f_drive_max, motor_cmd))
        s["f_motor"] = s["f_motor"] + (f_cmd - s["f_motor"]) * min(1.0, dt / p["motor_lag"])
        lo = 0.0 if self.held else -p["brake_dead_time"]
        apply_t = self._P(s, "brake_apply_time")
        b_target = torch.where(brake_cmd <= 0.0, torch.full_like(brake_cmd, lo), brake_cmd.clamp(max=1.0) * apply_t)
        s["brake_pos"] = s["brake_pos"] + (b_target - s["brake_pos"]).clamp(-dt, dt)
        f_brake = (s["brake_pos"] / apply_t).clamp(0.0, 1.0) * self._P(s, "brake_force")

        # ---- tyre loads
        L, h, wf, rs = self.L, self.h, p["front_weight"], p["rear_roll_share"]
        ax, ay = s["ax"], s["ay"]
        fz_front = m * G * wf - m * ax * h / L
        fz_rear = m * G * (1 - wf) + m * ax * h / L
        dfz_f = m * ay * h * (1 - rs) / self.tf
        dfz_r = m * ay * h * rs / self.tr
        warp = self._P(s, "steer_jacking") * delta
        fz = torch.stack([fz_front / 2 - dfz_f + warp, fz_front / 2 + dfz_f - warp,
                          fz_rear / 2 - dfz_r - warp, fz_rear / 2 + dfz_r + warp], 1).clamp(min=0.0)
        s["rolled"] = s["rolled"] | (ay.abs() > p["tip_lateral_g"] * G)

        # ---- wheel-frame velocities and slip angles
        dl, dr = self._wheel_angles(delta)
        zero = torch.zeros_like(dl)
        ang = torch.stack([dl, dr, zero, zero], 1)
        ca, sa = torch.cos(ang), torch.sin(ang)
        vx, vy, r = s["vx"], s["vy"], s["r"]
        vxi = vx[:, None] - r[:, None] * self.wy
        vyi = vy[:, None] + r[:, None] * self.wx
        vxw = vxi * ca + vyi * sa
        vyw = -vxi * sa + vyi * ca
        den = vxw.abs().clamp(min=V_LOW)
        tan_a = -vyw / den
        fz0, mu4 = s["fz0"], s["mu4"]

        # ---- rear axle, implicit
        def rear_fx(om):
            kap = (om[:, None] * self.Rr - vxw[:, 2:]) / den[:, 2:]
            return self._tyre(fz[:, 2:], kap, tan_a[:, 2:], mu4[:, 2:], fz0[:, 2:])[0].sum(1)
        eps = 0.01
        f0 = rear_fx(omega)
        dfdw = (rear_fx(omega + eps) - f0) / eps
        i_eff = p["axle_inertia"] + dt * self.Rr * dfdw.clamp(min=0.0)
        omega = (omega + dt * (s["f_motor"] * self.Rr - f_brake * self.Rr - self.Rr * f0) / i_eff).clamp(min=0.0)
        s["omega"] = omega

        # ---- tyre forces -> body frame
        kap = torch.cat([torch.zeros_like(vxw[:, :2]), (omega[:, None] * self.Rr - vxw[:, 2:]) / den[:, 2:]], 1)
        fxw, fyw = self._tyre(fz, kap, tan_a, mu4, fz0)
        fxw = torch.cat([-p["rolling_resistance"] * fz[:, :2] * torch.sign(vxw[:, :2]), fxw[:, 2:]], 1)
        fxi = fxw * ca - fyw * sa
        fyi = fxw * sa + fyw * ca
        fx_b = fxi.sum(1)
        fy_b = fyi.sum(1)
        mz = (self.wx * fyi - self.wy * fxi).sum(1)
        resist = p["rolling_resistance"] * (fz[:, 2] + fz[:, 3]) + 0.5 * AIR * p["drag_area"] * vx * vx
        fx_b = fx_b - torch.sign(vx) * torch.minimum(resist, m * vx.abs() / dt)

        # ---- body dynamics (semi-implicit Euler)
        ax, ay = fx_b / m, fy_b / m
        s["ax"], s["ay"] = ax, ay
        vx = vx + (ax + vy * r) * dt
        vy = vy + (ay - vx * r) * dt
        r = r + mz / p["yaw_inertia"] * dt
        vx = torch.where((vx < 0.0) & (omega == 0.0), torch.zeros_like(vx), vx)
        w = ((vx - 0.5) / 1.5).clamp(0.0, 1.0)
        r_kin = vx * torch.tan(delta) / L
        r = w * r + (1 - w) * r_kin
        vy = w * vy + (1 - w) * r_kin * self.lr
        yaw = s["yaw"]
        c, sn = torch.cos(yaw), torch.sin(yaw)
        s["x"] = s["x"] + (vx * c - vy * sn) * dt
        s["y"] = s["y"] + (vx * sn + vy * c) * dt
        s["yaw"] = _wrap(yaw + r * dt)
        s["vx"], s["vy"], s["r"] = vx, vy, r

    def _tyre_corners(self, s):
        """(N, 16, 2) tyre corners seen from above (Kart.tyre_corners)."""
        dl, dr = self._wheel_angles(s["delta"])
        zero = torch.zeros_like(dl)
        ang = torch.stack([dl, dr, zero, zero], 1)[:, :, None]          # (N, 4, 1)
        ca, sa = torch.cos(ang), torch.sin(ang)
        bx = self.wx[None, :, None] + self.ca_ * ca - self.cb_ * sa      # (N, 4, 4)
        by = self.wy[None, :, None] + self.ca_ * sa + self.cb_ * ca
        c, sn = torch.cos(s["yaw"])[:, None, None], torch.sin(s["yaw"])[:, None, None]
        wx = s["x"][:, None, None] + bx * c - by * sn
        wy = s["y"][:, None, None] + bx * sn + by * c
        return torch.stack([wx, wy], -1).reshape(self.N, 16, 2)

    # ================================================================== track
    def _nearest(self, pts, n, x, y, hint):
        """Nearest of pts (n, 2) to (x, y) within +-60 samples of hint."""
        idx = torch.remainder(hint[:, None] + self.win60, n)
        q = pts[idx]                                                     # (N, 121, 2)
        d = (q[..., 0] - x[:, None]) ** 2 + (q[..., 1] - y[:, None]) ** 2
        return idx.gather(1, d.argmin(1, keepdim=True))[:, 0]

    def _clearance(self, pts, k):
        """TrackMap.clearance: (N, P, 2) points, centreline index k -> (N,) worst signed distance."""
        out = None
        for poly, near, side in ((self.outer, self.near_out, self.side_out), (self.inner, self.near_in, self.side_in)):
            M = len(poly)
            idx = torch.remainder(near[k][:, None] + self.win16, M)      # (N, 32)
            a = poly[idx]
            ab = poly[torch.remainder(idx + 1, M)] - a                     # (N, 32, 2)
            ap = pts[:, :, None, :] - a[:, None]                           # (N, P, 32, 2)
            t = ((ap * ab[:, None]).sum(-1) / (ab * ab).sum(-1)[:, None]).clamp(0.0, 1.0)
            dvec = ap - t[..., None] * ab[:, None]
            dist = torch.sqrt((dvec * dvec).sum(-1))                       # (N, P, 32)
            j = dist.argmin(-1, keepdim=True)
            dj = dist.gather(-1, j)[..., 0]
            abj = ab[:, None].expand(-1, pts.shape[1], -1, -1).gather(2, j[..., None].expand(-1, -1, -1, 2))[:, :, 0]
            apj = ap.gather(2, j[..., None].expand(-1, -1, -1, 2))[:, :, 0]
            cross = abj[..., 0] * apj[..., 1] - abj[..., 1] * apj[..., 0]
            sd = dj * torch.where(cross * side >= 0, 1.0, -1.0)
            out = sd if out is None else torch.minimum(out, sd)
        return out.min(1).values

    def _track_check(self, s):
        """LapEnv.step's check after every low-level period: progress, clearance, ends."""
        alive = s["end"] == 0
        rx, ry = self._rear(s)
        k = self._nearest(self.t_pts, self.t_n, rx, ry, s["k_idx"])
        rel = torch.stack([rx, ry], 1) - self.t_pts[k]
        along = (rel * self.t_tan[k]).sum(1)
        sl = torch.remainder(self.t_s[k] + along, self.t_len)
        ds = torch.remainder(sl - s["s_last"] + self.t_len / 2, self.t_len) - self.t_len / 2
        prog = s["progress"] + ds
        clear = self._clearance(self._tyre_corners(s), k)
        end = torch.where(clear < 0.0, 2, torch.where(s["rolled"], 3, torch.where(prog >= self.t_len, 1, 0)))
        W = lambda new, old: torch.where(alive, new, old)
        s["k_idx"] = W(k, s["k_idx"])
        s["s_last"] = W(sl, s["s_last"])
        s["progress"] = W(prog, s["progress"])
        s["clear_last"] = W(clear, s["clear_last"])
        s["min_clear"] = W(torch.minimum(s["min_clear"], clear), s["min_clear"])
        s["max_lat_g"] = W(torch.maximum(s["max_lat_g"], s["ay"].abs() / 9.81), s["max_lat_g"])
        s["t_end"] = torch.where(alive & (end > 0), self._time(s), s["t_end"])
        s["end"] = W(end, s["end"])

    # ================================================================== one decision
    def step(self, speed, curvature):
        """Hold (target speed, curvature) for one decision (1/20 s) on every kart.
        -> (estimate dict, done (N,) bool, info dict of (N,) tensors). With use_graph()
        the returned tensors are overwritten by the next step: copy what you keep."""
        if self.graph is None:
            return self._step(speed, curvature)
        self._spd.copy_(speed)
        self._crv.copy_(curvature)
        self.graph.replay()
        return self._out

    def use_graph(self):
        """Record one decision step as a CUDA graph and replay it from now on (one launch
        instead of thousands; much faster). Restarts every lap."""
        self._spd = torch.zeros(self.N, device=self.dev, dtype=self.f)
        self._crv = torch.zeros(self.N, device=self.dev, dtype=self.f)
        if not self.s:
            self.reset()
        side = torch.cuda.Stream(self.dev)
        side.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(side):
            for _ in range(2):
                self._graph_body()
        torch.cuda.current_stream(self.dev).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        self.graph.register_generator_state(self.g)
        with torch.cuda.graph(self.graph):
            self._out = self._graph_body()
        # restarts as a graph too: mask and level are its inputs
        self._rmask = torch.ones(self.N, dtype=torch.bool, device=self.dev)
        self._rlevel = self.level.clone()
        with torch.cuda.stream(side):
            self._reset_body()
        torch.cuda.current_stream(self.dev).wait_stream(side)
        self.reset_graph = torch.cuda.CUDAGraph()
        self.reset_graph.register_generator_state(self.g)
        with torch.cuda.graph(self.reset_graph):
            self._rout = self._reset_body()
        return self.reset()

    def _reset_body(self):
        m = self._rmask
        self.level.copy_(torch.where(m, self._rlevel, self.level))
        new = dict(_leaves(self._fresh(self.level)))
        for name, old in _leaves(self.s):
            v = new[name]
            old.copy_(torch.where(m.view(-1, *([1] * (v.dim() - 1))), v, old))
        return self._est(self.s)

    def _graph_body(self):
        saved = self.s
        self.s = {k: (dict(v) if isinstance(v, dict) else v) for k, v in saved.items()}
        out = self._step(self._spd, self._crv)
        new = dict(_leaves(self.s))
        for name, old in _leaves(saved):
            if new[name] is not old:
                old.copy_(new[name])
        self.s = saved
        return out

    def _step(self, speed, curvature):
        s = self.s
        s["v_target"] = speed.to(self.f).clamp(min=0.0)
        s["curv"] = curvature.to(self.f)
        s["end"] = torch.zeros_like(s["end"])
        for i in range(self.n_sub):
            sense = (i + 1) % self.ll_every == 0
            if sense:
                self._sense(s, gnss=(i == self.n_sub - 1))
            self._kart_step(s, *s["cmd"])
            s["n"] = s["n"] + 1
            if sense:
                self._track_check(s)
        # stuck / time limit, then the estimate at the end of the decision
        t = self._time(s)
        going = s["end"] == 0
        slow = (torch.hypot(s["vx"], s["vy"]) < 0.2) & (t > 5.0)
        since = torch.where(slow & (s["slow_since"] < 0), t, s["slow_since"])
        since = torch.where(slow, since, torch.full_like(since, -1.0))
        s["slow_since"] = torch.where(going, since, s["slow_since"])
        end = torch.where(going & slow & (t - since > 5.0), 4, s["end"])
        end = torch.where(going & (t > TIME_LIMIT), 5, end)
        s["t_end"] = torch.where(going & (end > 0), t, s["t_end"])
        s["end"] = end
        self._predict(s, t)
        done = end > 0
        rx, ry = self._rear(s)
        info = {"end": end, "lap_time": s["t_end"] - s["t0"], "progress": s["progress"],
                "min_clearance": s["min_clear"], "clear": s["clear_last"], "max_lat_g": s["max_lat_g"],
                "t": t, "true_x": rx, "true_y": ry, "true_v": torch.hypot(s["vx"], s["vy"])}
        return self._est(s), done, info

    def _sense(self, s, gnss):
        """One low-level period's sensing: readings -> EKF -> low-level loop (LapEnv._physics_step)."""
        t = self._time(s)
        imu = self._imu_sample(s)
        wheel = self._wheel_sample(s)
        self._predict(s, t)
        hist_now = s["X"][:, :4].clone()
        s["gyro"], s["acc0"], s["acc1"] = imu
        self._wheel_update(s, wheel, torch.ones_like(s["lat"]))
        if gnss:
            new = self._gnss_sample(s)
            lat = s["lat"]
            msg = {k: torch.where(lat, s["pend"][k], v) for k, v in new.items()}
            hist = torch.where(lat[:, None], s["hist"], hist_now)
            self._gnss_update(s, msg, hist, t)
            s["pend"] = {k: (v & lat if k == "valid" else v) for k, v in new.items()}
            s["hist"] = hist_now
        s["wheel_speed"] = wheel
        s["cmd"] = self._lowlevel(s)

    # ================================================================== driver B
    def _ahead(self, k, dist):
        target = torch.remainder(self.l_s[k] + dist, self.l_len)
        return torch.remainder(torch.searchsorted(self.l_s, target.contiguous()), self.l_n)

    def drive_b(self, est):
        """PurePursuit.act (driver B, the optimised line) for every kart -> (speed, curvature)."""
        s, c = self.s, self.pp
        x, y, yaw, v = est["x"], est["y"], est["yaw"], est["v"]
        k = self._nearest(self.l_pts, self.l_n, x, y, s["bk"])
        s["bk"].copy_(k)                          # in place: the graphs hold this buffer
        ld = (c["gain"] * v + c["base"]).clamp(c["min"], c["max"])
        j = self._ahead(k, ld)
        dx, dy = self.l_pts[j, 0] - x, self.l_pts[j, 1] - y
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        lx, ly = cy * dx + sy * dy, -sy * dx + cy * dy
        curv = 2.0 * ly / (lx * lx + ly * ly).clamp(min=1e-6)
        speed = self.l_speed[self._ahead(k, v * c["speed_preview"])]
        speed = torch.where(est["gnss_age"] > 0.5, speed * c["lost_slowdown"],
                            torch.where(~est["rtk_fixed"], speed * c["float_slowdown"], speed))
        return speed, curv


def _leaves(s, pre=""):
    """(name, tensor) for every tensor in the state dict (nested dicts and tuples)."""
    for k, v in s.items():
        if isinstance(v, dict):
            yield from _leaves(v, pre + k + ".")
        elif isinstance(v, tuple):
            for i, a in enumerate(v):
                yield f"{pre}{k}.{i}", a
        else:
            yield pre + k, v


def end_names(end):
    """(N,) end codes -> list of names."""
    return [END[int(e)] for e in np.asarray(end.cpu())]
