"""Simulated kart sensors: RTK GNSS (ZED-F9P), IMU (BNO055) and the rear-axle encoder.

Readings come out as messages with the time they were measured and the time they
arrive, like the ROS topics on the real kart:
  {"type": "gnss", "stamp", "x", "y", "ve", "vn", "fix": "fixed" | "float"}
  {"type": "imu", "stamp", "gyro_z", "ax", "ay"}
  {"type": "wheel", "stamp", "speed"}

The noise level (perfect / realistic / harsh) scales every noise, bias, offset and
event rate in gps.yaml 'sensors'. GNSS positions are in the MAP frame but include the
per-run map offset (where the kart's GPS thinks the map is vs where it really is).
"""
import heapq
import math

import numpy as np


class Sensors:
    def __init__(self, cfg, level=None):
        """cfg: gps.yaml 'sensors' dict. level: perfect | realistic | harsh."""
        self.c = cfg
        self.level = level or cfg["level"]
        lv = cfg["levels"][self.level]
        self.k = float(lv["scale"])
        self.latency = cfg["gnss"]["latency"] if lv["latency"] else 0.0

    def reset(self, rng, t=0.0):
        c, k = self.c, self.k
        self.rng = rng
        g = c["gnss"]
        mo = c["map_offset"]
        self.shift = rng.uniform(-1, 1, 2) * mo["shift"] * k
        self.rot = math.radians(rng.uniform(-1, 1) * mo["rotate_deg"] * k)
        self.gyro_bias = rng.normal() * c["imu"]["gyro_bias"] * k
        self.accel_bias = rng.normal(size=2) * c["imu"]["accel_bias"] * k
        self.wheel_scale = 1.0 + rng.normal() * c["wheel"]["scale_error"] * k
        self.wander = np.zeros(2)
        self.float_until = self.drop_until = -1.0
        self.float_offset = np.zeros(2)
        self.next_float = t + self._wait(g["float_rate"])
        self.next_drop = t + self._wait(g["dropout_rate"])
        self.t_next = {"gnss": t, "imu": t, "wheel": t}
        self.queue = []          # (arrival time, seq, message)
        self.seq = 0

    def _wait(self, per_minute):
        rate = per_minute * self.k / 60.0
        return self.rng.exponential(1.0 / rate) if rate > 0 else math.inf

    def _send(self, arrive, msg):
        heapq.heappush(self.queue, (arrive, self.seq, msg))
        self.seq += 1

    def sample(self, kart):
        """Call every physics step. Takes readings that are due from the kart's true state."""
        t, c, k, rng = kart.t, self.c, self.k, self.rng
        if t + 1e-9 >= self.t_next["imu"]:
            dt = 1.0 / c["imu"]["rate_hz"]
            self.t_next["imu"] += dt
            i = c["imu"]
            self._send(t, {"type": "imu", "stamp": t,
                           "gyro_z": kart.r + self.gyro_bias + rng.normal() * i["gyro_noise"] * k,
                           "ax": kart.ax + self.accel_bias[0] + rng.normal() * i["accel_noise"] * k,
                           "ay": kart.ay + self.accel_bias[1] + rng.normal() * i["accel_noise"] * k})
        if t + 1e-9 >= self.t_next["wheel"]:
            self.t_next["wheel"] += 1.0 / c["wheel"]["rate_hz"]
            v = kart.axle_speed * self.wheel_scale + rng.normal() * c["wheel"]["noise"] * k
            self._send(t, {"type": "wheel", "stamp": t, "speed": max(0.0, v)})
        if t + 1e-9 >= self.t_next["gnss"]:
            g = c["gnss"]
            dt = 1.0 / g["rate_hz"]
            self.t_next["gnss"] += dt
            tau = 10.0
            a = math.exp(-dt / tau)
            self.wander = self.wander * a + rng.normal(size=2) * g["pos_wander"] * k * math.sqrt(1 - a * a)
            if t >= self.next_float:
                self.float_until = t + rng.uniform(*g["float_duration"])
                ang = rng.uniform(0, 2 * math.pi)
                self.float_offset = g["float_error"] * k * np.array([math.cos(ang), math.sin(ang)])
                self.next_float = self.float_until + self._wait(g["float_rate"])
            if t >= self.next_drop:
                self.drop_until = t + rng.uniform(*g["dropout_duration"])
                self.next_drop = self.drop_until + self._wait(g["dropout_rate"])
            if t < self.drop_until:
                return
            fl = t < self.float_until
            x, y = getattr(kart, "antenna", kart.rear_axle)   # MjKart: the real antenna position
            cr, sr = math.cos(self.rot), math.sin(self.rot)
            noise = rng.normal(size=2) * g["pos_noise"] * k * (3.0 if fl else 1.0)
            err = self.shift + self.wander + noise + (self.float_offset if fl else 0.0)
            # velocity of the rear axle point (the antenna)
            c_, s_ = math.cos(kart.yaw), math.sin(kart.yaw)
            vy_r = kart.vy - kart.r * kart.lr
            ve = kart.vx * c_ - vy_r * s_
            vn = kart.vx * s_ + vy_r * c_
            self._send(t + self.latency, {
                "type": "gnss", "stamp": t, "fix": "float" if fl else "fixed",
                "x": cr * x - sr * y + err[0], "y": sr * x + cr * y + err[1],
                "ve": ve + rng.normal() * g["vel_noise"] * k, "vn": vn + rng.normal() * g["vel_noise"] * k})

    def receive(self, t):
        """Messages that have arrived by time t, oldest first."""
        out = []
        while self.queue and self.queue[0][0] <= t + 1e-9:
            out.append(heapq.heappop(self.queue)[2])
        return out
