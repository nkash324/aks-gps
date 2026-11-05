"""EKF that every driver uses: fuses RTK GNSS, IMU and the wheel encoder into the kart's
position, heading and speed. Runs unchanged on the Jetson (it only sees messages).

State: [x, y, yaw, v, gyro_bias, accel_bias]
  x, y   rear axle centre = the GNSS antenna point, map frame (m)
  yaw    heading (rad), v forward speed (m/s)
Prediction at the IMU rate (gyro for yaw, accelerometer for speed). Corrections:
  - wheel encoder speed (skipped while the rears slip under hard braking/acceleration)
  - GNSS position (float fixes trusted much less); the message is ~50 ms old, so it is
    compared with where the filter thought the kart was at that time
  - GNSS velocity: speed, and direction of travel = heading when moving > 3 m/s
"""
import math
from collections import deque

import numpy as np

SIG_POS_FIXED = 0.03
SIG_POS_FLOAT = 0.6
GATE = 4.0        # reject corrections more than this many sigma away (outliers)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Ekf:
    def __init__(self):
        self.ready = False

    def reset(self, x, y, yaw, t=0.0, sig_yaw=math.radians(3.0)):
        """Start at a GNSS fix, facing yaw (the kart sits on the grid facing along the track)."""
        self.X = np.array([x, y, yaw, 0.0, 0.0, 0.0])
        self.P = np.diag([0.05, 0.05, sig_yaw, 0.1, 0.01, 0.1]) ** 2
        self.t = t
        self.gyro = 0.0
        self.acc = (0.0, 0.0)
        self.hist = deque(maxlen=60)       # (t, x, y, yaw, v) for delayed GNSS
        self.last_fix = t
        self.last_fixed = t
        self.fix = "fixed"
        self._last_pos_ok = t
        self.ready = True

    # ------------------------------------------------------------------ predict
    def _predict(self, t):
        dt = t - self.t
        if dt <= 0:
            return
        x, y, yaw, v, bg, ba = self.X
        w = self.gyro - bg
        a = self.acc[0] - ba
        c, s = math.cos(yaw), math.sin(yaw)
        self.X = np.array([x + v * c * dt, y + v * s * dt, _wrap(yaw + w * dt),
                           max(0.0, v + a * dt), bg, ba])
        F = np.eye(6)
        F[0, 2], F[0, 3] = -v * s * dt, c * dt
        F[1, 2], F[1, 3] = v * c * dt, s * dt
        F[2, 4] = -dt
        F[3, 5] = -dt
        # process noise: sideslip of the rear axle, gyro and accelerometer noise, bias drift
        q = np.array([0.05 + 0.02 * v, 0.05 + 0.02 * v, 0.01, 0.5, 1e-4, 1e-2]) ** 2 * dt
        self.P = F @ self.P @ F.T + np.diag(q)
        self.t = t
        self.hist.append((t, *self.X[:4]))

    def _update(self, z, h, H, R, angle=False):
        y = z - h
        if angle:
            y[0] = _wrap(y[0])
        S = H @ self.P @ H.T + R
        d2 = float(y @ np.linalg.solve(S, y))
        if d2 > GATE ** 2 * len(y):
            return False
        K = self.P @ H.T @ np.linalg.inv(S)
        self.X = self.X + K @ y
        self.X[2] = _wrap(self.X[2])
        self.X[3] = max(0.0, self.X[3])
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return True

    def _at(self, stamp):
        """Filter state (x, y, yaw, v) at an earlier time, from the history."""
        best = None
        for h in reversed(self.hist):
            best = h
            if h[0] <= stamp + 1e-9:
                break
        return best[1:] if best else tuple(self.X[:4])

    # ------------------------------------------------------------------ messages
    def handle(self, msg, t):
        """Feed one sensor message that arrived at time t."""
        if not self.ready:
            return
        self._predict(t)
        typ = msg["type"]
        if typ == "imu":
            self.gyro = msg["gyro_z"]
            self.acc = (msg["ax"], msg["ay"])
        elif typ == "wheel":
            # the rears slip under hard braking/acceleration: trust the encoder less then
            sig = 0.05 + 0.02 * self.X[3] + 0.3 * max(0.0, abs(self.acc[0]) - 2.0)
            H = np.zeros((1, 6))
            H[0, 3] = 1.0
            self._update(np.array([msg["speed"]]), np.array([self.X[3]]), H, np.array([[sig ** 2]]))
        elif typ == "gnss":
            self.fix = msg["fix"]
            self.last_fix = t
            if msg["fix"] == "fixed":
                self.last_fixed = t
            px, py, pyaw, pv = self._at(msg["stamp"])
            sig = SIG_POS_FIXED if msg["fix"] == "fixed" else SIG_POS_FLOAT
            H = np.zeros((2, 6))
            H[0, 0] = H[1, 1] = 1.0
            ok = self._update(np.array([msg["x"], msg["y"]]), np.array([px, py]), H, np.eye(2) * sig ** 2)
            if not ok and msg["fix"] == "fixed" and t - self._last_pos_ok > 1.0:
                # a long run of rejected fixes means the filter is lost: trust the GNSS again
                self.X[0], self.X[1] = msg["x"] + (self.X[0] - px), msg["y"] + (self.X[1] - py)
                self.P[0, 0] = self.P[1, 1] = sig ** 2
                ok = True
            if ok:
                self._last_pos_ok = t
            speed = math.hypot(msg["ve"], msg["vn"])
            if speed > 3.0:
                H = np.zeros((2, 6))
                H[0, 2] = 1.0
                H[1, 3] = 1.0
                cog = math.atan2(msg["vn"], msg["ve"])
                sig_cog = max(0.04, 0.1 / speed)      # also covers the rear axle's slip angle
                self._update(np.array([cog, speed]), np.array([pyaw, pv]), H,
                             np.diag([sig_cog ** 2, 0.1 ** 2]), angle=True)

    def state(self, t):
        """Estimate at time t (predicted forward from the last message)."""
        self._predict(t)
        x, y, yaw, v, bg, ba = self.X
        return {"x": x, "y": y, "yaw": yaw, "v": v, "yaw_rate": self.gyro - bg,
                "ax": self.acc[0] - ba, "ay": self.acc[1],
                "pos_std": math.sqrt(max(self.P[0, 0], self.P[1, 1])),
                "rtk_fixed": self.fix == "fixed" and t - self.last_fixed < 0.5,
                "gnss_age": t - self.last_fix}
