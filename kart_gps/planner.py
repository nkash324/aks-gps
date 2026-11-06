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


def track_bounds(track, kart, buffer=0.3, step=0.5):
    """Centreline samples and how far a line may sit from them.
    -> (C (n, 2) centreline points, N (n, 2) left normals, lo (n,), hi (n,)) where lo / hi
    are the sideways offset bounds (left +) that keep every tyre `buffer` m inside the edges."""
    idx = np.arange(0, track.n, max(1, int(round(step / (track.length / track.n)))))
    C, N = track.pts[idx], track.normal[idx]
    hw = kart_half_width(kart) + buffer
    lo, hi = -(track.hw_right[idx] - hw), track.hw_left[idx] - hw
    return C, N, lo, hi


def min_curvature_line(track, kart, buffer=0.3, step=0.5, iters=5000):
    """Minimum-curvature line inside the edges, with `buffer` m between the tyres and edges."""
    C, N, lo, hi = track_bounds(track, kart, buffer, step)
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


# ---------------------------------------------------------------------------------------
# Physics-based speed plan and the minimum-time line (the "formula" drivers C and MPC)
# ---------------------------------------------------------------------------------------
AIR = 1.2

# Starting values from scripts/gps/kart_checks.py on the 2D model: skidpad 1.15 g (11.3 m/s^2)
# and peak braking 4.81 m/s^2 (regen + hydraulic), each scaled by a margin for the
# per-lap randomisation (grip +-15 %, ...) and tracking errors.
GGV_DEFAULTS = {"lat": 0.65 * 11.3, "brake": 0.8 * 4.81, "top_speed": 11.5, "p": 2.0}


def speed_plan_ggv(line, kart, lat=None, brake=None, top_speed=None, p=None, passes=2,
                   hydraulic=1.0, store=True):
    """Speed plan from a simple point-mass model of the nominal kart (a 'GGV' diagram).

    - Friction ellipse: with lateral acceleration a_y = v^2 |curvature| the share of the
      longitudinal limit that is left is (1 - (a_y / lat)^p)^(1/p). p = 2 is the
      friction circle; p > 2 is a squarer diagram (corner and brake at the same time).
    - Accelerating: motor force (drive_force, fading over the last top_speed_taper m/s
      below the motor limit) minus rolling resistance and drag, times the ellipse share.
    - Braking: min(brake, (regen + hydraulic) / mass), where regen = min(regen_force,
      regen_power / v) and hydraulic = `hydraulic` x brake_force; times the ellipse share,
      plus the resistance that helps slow the kart.
    - Cornering speed: sqrt(lat / |curvature|), capped at top_speed.
    - Forward and backward passes round the closed loop, `passes` times.

    lat, brake in m/s^2 (the tunable limits; defaults GGV_DEFAULTS). Returns the speed (m/s) at
    every line point; with store=True also sets line.speed and line.lap_estimate (s).
    """
    d = GGV_DEFAULTS
    lat = d["lat"] if lat is None else lat
    brake = d["brake"] if brake is None else brake
    top_speed = d["top_speed"] if top_speed is None else top_speed
    p = d["p"] if p is None else p
    m = kart["mass"]
    kappa = np.abs(line.curvature)
    v_corner = np.minimum(top_speed, np.sqrt(lat / np.maximum(kappa, 1e-6)))
    ds = np.diff(np.r_[line.s, line.length])
    n = line.n
    f_drive, v_top, taper = kart["drive_force"], kart["top_speed"], kart["top_speed_taper"]
    roll = kart["rolling_resistance"] * m * G
    cda = 0.5 * AIR * kart["drag_area"]
    ip = 1.0 / p

    def share(v, k):
        r = min(1.0, v * v * k / lat)
        return (1.0 - r ** p) ** ip

    def accel_max(v, k):
        f = f_drive * min(1.0, max(0.0, (v_top - v) / taper))
        return (f - roll - cda * v * v) / m * share(v, k)

    def decel_max(v, k):
        f = min(kart["regen_force"], kart["regen_power"] / max(v, 0.5)) + hydraulic * kart["brake_force"]
        return min(brake, f / m) * share(v, k) + (roll + cda * v * v) / m

    v = v_corner.copy()
    for _ in range(passes):
        for i in range(n):
            j = (i + 1) % n
            v[j] = min(v[j], math.sqrt(v[i] ** 2 + 2 * max(accel_max(v[i], kappa[i]), 0.0) * ds[i]))
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2 * decel_max(v[j], kappa[j]) * ds[i]))
    if store:
        line.speed = v
        line.lap_estimate = float(np.sum(ds / np.maximum(v, 0.1)))
    return v


def bspline_basis(n, spacing_m, length):
    """(n, nb) periodic cubic B-spline bumps, one every ~spacing_m metres along n samples."""
    nb = max(8, int(round(length / spacing_m)))
    h = n / nb
    i = np.arange(n)[:, None]
    c = np.arange(nb)[None, :] * h
    d = np.abs(((i - c + n / 2) % n) - n / 2) / h
    return np.where(d < 1, 2 / 3 - d ** 2 + d ** 3 / 2, np.where(d < 2, (2 - d) ** 3 / 6, 0.0))


def mintime_line(track, kart, buffer=0.3, step=0.5, plan=None, spacing=5.0, sweeps=60,
                 step0=0.4, min_step=0.02, z0=None, verbose=False):
    """Minimum-time line: lowest lap time of the speed plan, inside the same edges.

    Starts from the minimum-curvature offsets z0 and moves z = z0 + B w, B a periodic cubic
    B-spline basis with a bump every `spacing` m. Pattern search on w: try +-step on each
    weight, keep a move if it stays inside [lo, hi] and the planned lap time drops; when a
    full sweep finds nothing the step halves, until it is below min_step.
    plan: dict of speed_plan_ggv keywords (the limits the line is optimised for).
    Returns a Line named 'mintime' (.offset, .lap_estimate set; speed plan stored).
    """
    plan = dict(plan or {})
    C, N, lo, hi = track_bounds(track, kart, buffer, step)
    if z0 is None:
        z0 = min_curvature_line(track, kart, buffer, step).offset
    z0 = np.clip(z0, lo, hi)
    B = bspline_basis(len(C), spacing, float(np.sum(np.linalg.norm(np.roll(C, -1, 0) - C, axis=1))))
    nb = B.shape[1]

    def make(z):
        return Line(C + z[:, None] * N, "mintime")

    def cost(z):
        ln = make(z)
        speed_plan_ggv(ln, kart, store=True, **plan)
        return ln.lap_estimate

    w = np.zeros(nb)
    best = cost(z0)
    t_start = best
    st = step0
    rng = np.random.default_rng(0)
    while st >= min_step and sweeps > 0:
        improved = False
        sweeps -= 1
        for j in rng.permutation(nb):
            for sgn in (1.0, -1.0):
                w2 = w.copy()
                w2[j] += sgn * st
                z = z0 + B @ w2
                if np.any(z < lo) or np.any(z > hi):
                    continue
                c = cost(z)
                if c < best - 1e-6:
                    best, w, improved = c, w2, True
                    break
        if verbose:
            print(f"  step {st:.3f} m: planned lap {best:.3f} s (start {t_start:.3f})", flush=True)
        if not improved:
            st /= 2
    z = z0 + B @ w
    line = make(z)
    speed_plan_ggv(line, kart, **plan)
    line.offset = z
    line.lap_estimate_start = t_start
    return line
