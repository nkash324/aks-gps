"""Racing lines and speed plans, made once from the map before driving.

  - centreline: the middle of the track
  - optimised line: the minimum-curvature line that keeps every tyre at least `buffer`
    (0.3 m) inside the surveyed edges. Found as a convex least-squares problem on the
    sideways offset from the centreline, solved with ADMM (numpy only, ~1 s).
  - speed plan along a line: as fast as the cornering limit allows, then forward/backward
    passes for how fast the kart can actually speed up and slow down (friction circle:
    less braking/acceleration is available while cornering).
"""
import math

import numpy as np

G = 9.81


class Line:
    """A closed driving line with its speed plan."""
    def __init__(self, pts, name):
        self.name = name
        self.pts = np.asarray(pts, dtype=float)
        self.n = len(self.pts)
        seg = np.linalg.norm(np.roll(self.pts, -1, axis=0) - self.pts, axis=1)
        self.s = np.r_[0.0, np.cumsum(seg)[:-1]]
        self.length = float(seg.sum())
        tang = np.roll(self.pts, -1, axis=0) - np.roll(self.pts, 1, axis=0)
        self.tangent = tang / np.linalg.norm(tang, axis=1, keepdims=True)
        self.heading = np.arctan2(self.tangent[:, 1], self.tangent[:, 0])
        self.curvature = _curvature(self.pts)
        self.speed = None

    def nearest(self, x, y, hint=None, window=60):
        if hint is not None:
            idx = (hint + np.arange(-window, window + 1)) % self.n
            d = np.hypot(self.pts[idx, 0] - x, self.pts[idx, 1] - y)
            j = int(d.argmin())
            if d[j] < 5.0 and 0 < j < 2 * window:
                return int(idx[j])
        return int(np.argmin(np.hypot(self.pts[:, 0] - x, self.pts[:, 1] - y)))

    def ahead(self, k, dist):
        """Index of the point `dist` metres along the line after point k."""
        target = (self.s[k] + dist) % self.length
        j = int(np.searchsorted(self.s, target)) % self.n
        return j


def _curvature(P):
    """Signed curvature at each point of a closed polyline (circle through neighbours)."""
    a, b, c = np.roll(P, 1, axis=0), P, np.roll(P, -1, axis=0)
    ab, bc, ac = b - a, c - b, c - a
    cross = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]
    den = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1) * np.linalg.norm(ac, axis=1)
    return 2 * cross / np.maximum(den, 1e-9)


def kart_half_width(kart):
    """Kart centre to the outside of a rear tyre (the widest point)."""
    return max(kart["track_rear"] + kart["tyre_rear"]["width"],
               kart["track_front"] + kart["tyre_front"]["width"]) / 2


def centreline(track, step=0.5):
    idx = np.arange(0, track.n, max(1, int(round(step / (track.length / track.n)))))
    return Line(track.pts[idx], "centreline")


def min_curvature_line(track, kart, buffer=0.3, step=0.5, iters=5000):
    """Minimum-curvature line inside the edges, with `buffer` m between the tyres and edges."""
    idx = np.arange(0, track.n, max(1, int(round(step / (track.length / track.n)))))
    C, N = track.pts[idx], track.normal[idx]
    hw = kart_half_width(kart) + buffer
    lo, hi = -(track.hw_right[idx] - hw), track.hw_left[idx] - hw
    n = len(C)
    # second difference of the points = curvature * ds^2 (spacing is even):
    #   p[i-1] - 2 p[i] + p[i+1],  p = C + offset * N  -> linear in the offsets
    D = np.zeros((2 * n, n))
    d0 = np.zeros(2 * n)
    for i in range(n):
        im, ip = (i - 1) % n, (i + 1) % n
        for ax in (0, 1):
            r = 2 * i + ax
            D[r, im] += N[im, ax]
            D[r, i] += -2 * N[i, ax]
            D[r, ip] += N[ip, ax]
            d0[r] = C[im, ax] - 2 * C[i, ax] + C[ip, ax]
    Q = D.T @ D
    q = D.T @ d0
    # ADMM: least-squares step, clamp to the track, dual update. Q is badly conditioned
    # (slow, wide moves barely change curvature), which a small rho handles well.
    rho = 1e-3
    Linv = np.linalg.inv(Q + rho * np.eye(n))
    z = np.zeros(n)
    u = np.zeros(n)
    for _ in range(iters):
        w = Linv @ (-q + rho * (z - u))
        z = np.clip(w + u, lo, hi)
        u += w - z
    pts = C + z[:, None] * N
    line = Line(pts, "optimised")
    line.offset = z
    return line


def speed_plan(line, kart, lat_accel, brake_decel, top_speed, passes=2):
    """Fill line.speed (m/s at each point) for these limits (nominal kart numbers)."""
    m = kart["mass"]
    kappa = np.abs(line.curvature)
    v_corner = np.minimum(top_speed, np.sqrt(lat_accel / np.maximum(kappa, 1e-6)))
    ds = np.diff(np.r_[line.s, line.length])
    n = line.n

    def accel_max(v, k):
        f = kart["drive_force"] * min(1.0, max(0.0, (kart["top_speed"] - v) / kart["top_speed_taper"]))
        resist = kart["rolling_resistance"] * m * G + 0.6 * kart["drag_area"] * v * v
        a = (f - resist) / m
        lat = min(1.0, v * v * k / lat_accel)
        return a * math.sqrt(max(0.0, 1.0 - lat * lat))

    def decel_max(v, k):
        lat = min(1.0, v * v * k / lat_accel)
        return brake_decel * math.sqrt(max(0.0, 1.0 - lat * lat)) + 0.05

    v = v_corner.copy()
    for _ in range(passes):      # the loop is closed: go round twice so the ends agree
        for i in range(n):       # forward: how fast can it get there
            j = (i + 1) % n
            v[j] = min(v[j], math.sqrt(v[i] ** 2 + 2 * accel_max(v[i], kappa[i]) * ds[i]))
        for i in range(n - 1, -1, -1):   # backward: can it slow down in time
            j = (i + 1) % n
            v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2 * decel_max(v[j], kappa[j]) * ds[i]))
    line.speed = v
    line.lap_estimate = float(np.sum(ds / np.maximum(v, 0.1)))
    return v
