"""Track map for the GPS stack, read straight from a track world XML (no MuJoCo needed,
so the same code runs on the Jetson).

The world draws each surveyed edge as a chain of short boxes: line0_* (outer edge)
and line1_* (inner edge). Their centres are the surveyed edge. From them:
  - outer / inner: closed edge polygons
  - pts: centreline (0.25 m spacing), s = 0 at the start, increasing in race direction
  - hw_left / hw_right: distance from the centreline to each edge (m)
  - inside(points): True where a point is on the track (between the two edges)
  - project(x, y): progress along the lap s and the offset from the centreline
"""
import math
import re
import xml.etree.ElementTree as ET

import numpy as np

STEP = 0.25          # m between centreline samples


def _resample_closed(P, step):
    Pc = np.vstack([P, P[:1]])
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(Pc, axis=0), axis=1))]
    n = max(3, int(round(s[-1] / step)))
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, s, Pc[:, 0]), np.interp(t, s, Pc[:, 1])]), s[-1]


def _smooth_closed(P, k):
    pad = np.vstack([P[-k:], P, P[:k]])
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    return np.column_stack([np.convolve(pad[:, j], ker, "valid") for j in range(P.shape[1])])


def points_in_polygon(pts, poly):
    """Even-odd rule for many points against one closed polygon. pts (N, 2), poly (M, 2)."""
    x, y = pts[:, 0:1], pts[:, 1:2]
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    crosses = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    return (np.count_nonzero(crosses & (x < xc), axis=1) % 2) == 1


def dist_to_polyline(pts, poly):
    """Distance from each point (N, 2) to a closed polyline (M, 2)."""
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a
    ap = pts[:, None, :] - a[None]
    t = np.clip(np.einsum("nmk,mk->nm", ap, ab) / np.einsum("mk,mk->m", ab, ab), 0.0, 1.0)
    d = ap - t[..., None] * ab[None]
    return np.sqrt(np.min(np.einsum("nmk,nmk->nm", d, d), axis=1))


class TrackMap:
    def __init__(self, xml_path):
        root = ET.parse(xml_path).getroot()
        chains = {"line0_": [], "line1_": []}
        for g in root.iter("geom"):
            m = re.fullmatch(r"(line[01]_)(\d+)", g.get("name", ""))
            if m:
                x, y = (float(v) for v in g.get("pos").split()[:2])
                chains[m.group(1)].append((int(m.group(2)), x, y))
        self.outer = np.array([[x, y] for _, x, y in sorted(chains["line0_"])])
        self.inner = np.array([[x, y] for _, x, y in sorted(chains["line1_"])])
        start = None
        for n in root.iter("numeric"):
            if n.get("name") == "start_pose":
                start = [float(v) for v in n.get("data").split()]
            if n.get("name") == "gnss_origin":
                self.gnss_origin = tuple(float(v) for v in n.get("data").split())
        self.start_pose = tuple(start)

        # Centreline: midpoint between each outer point and its nearest inner point, smoothed.
        d = np.linalg.norm(self.outer[:, None] - self.inner[None], axis=2)
        mid = (self.outer + self.inner[d.argmin(axis=1)]) / 2
        mid, _ = _resample_closed(mid, 0.5)
        mid = _smooth_closed(mid, 6)
        pts, _ = _resample_closed(mid, STEP)
        sx, sy, syaw = self.start_pose
        i = int(np.argmin(np.hypot(pts[:, 0] - sx, pts[:, 1] - sy)))
        t = pts[(i + 1) % len(pts)] - pts[i]
        if t[0] * math.cos(syaw) + t[1] * math.sin(syaw) < 0:
            pts = pts[::-1].copy()
        i = int(np.argmin(np.hypot(pts[:, 0] - sx, pts[:, 1] - sy)))
        self.pts = np.roll(pts, -i, axis=0)
        self.n = len(self.pts)
        seg = np.linalg.norm(np.diff(np.vstack([self.pts, self.pts[:1]]), axis=0), axis=1)
        self.s = np.r_[0.0, np.cumsum(seg)[:-1]]
        self.length = float(seg.sum())

        nxt = np.roll(self.pts, -1, axis=0)
        prv = np.roll(self.pts, 1, axis=0)
        tang = nxt - prv
        self.tangent = tang / np.linalg.norm(tang, axis=1, keepdims=True)
        self.normal = np.column_stack([-self.tangent[:, 1], self.tangent[:, 0]])   # points left
        self.heading = np.arctan2(self.tangent[:, 1], self.tangent[:, 0])
        dh = np.angle(np.exp(1j * (np.roll(self.heading, -1) - self.heading)))
        self.curvature = _smooth_closed((dh / seg)[:, None], 4)[:, 0]

        # Half widths: distance from each centreline point to the edge on each side.
        # The outer edge is on the right when driving counter-clockwise.
        d_out = dist_to_polyline(self.pts, self.outer)
        d_in = dist_to_polyline(self.pts, self.inner)
        ccw = _signed_area(self.outer) > 0
        self.hw_left, self.hw_right = (d_in, d_out) if ccw else (d_out, d_in)
        self.left_edge = self.pts + self.normal * self.hw_left[:, None]
        self.right_edge = self.pts - self.normal * self.hw_right[:, None]
        # For the fast local clearance check: nearest edge vertex to each centreline
        # sample, and which side of each edge (seen along its point order) is the track.
        self._near_out = np.argmin(np.linalg.norm(self.pts[:, None] - self.outer[None], axis=2), axis=1)
        self._near_in = np.argmin(np.linalg.norm(self.pts[:, None] - self.inner[None], axis=2), axis=1)
        self._side_out = 1.0 if _signed_area(self.outer) > 0 else -1.0    # track = left of a CCW outer edge
        self._side_in = -1.0 if _signed_area(self.inner) > 0 else 1.0     # track = right of a CCW inner edge

    # ------------------------------------------------------------ queries
    def nearest(self, x, y, hint=None, window=60):
        """Index of the nearest centreline sample. Searches near `hint` first."""
        if hint is not None:
            idx = (hint + np.arange(-window, window + 1)) % self.n
            d = np.hypot(self.pts[idx, 0] - x, self.pts[idx, 1] - y)
            j = int(d.argmin())
            if d[j] < 8.0 and 0 < j < 2 * window:
                return int(idx[j])
        return int(np.argmin(np.hypot(self.pts[:, 0] - x, self.pts[:, 1] - y)))

    def project(self, x, y, hint=None):
        """-> (index, s along the lap in m, lateral offset in m, + = left of centre)."""
        k = self.nearest(x, y, hint)
        rel = np.array([x - self.pts[k, 0], y - self.pts[k, 1]])
        along = float(rel @ self.tangent[k])
        lat = float(rel @ self.normal[k])
        return k, (self.s[k] + along) % self.length, lat

    def inside(self, pts):
        """True for each point (N, 2) that is on the track surface (between the edges)."""
        pts = np.asarray(pts, dtype=float)
        return points_in_polygon(pts, self.outer) & ~points_in_polygon(pts, self.inner)

    def edge_distance(self, pts):
        """Distance (m) from each point to the nearest surveyed edge."""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        return np.minimum(dist_to_polyline(pts, self.outer), dist_to_polyline(pts, self.inner))

    def clearance(self, pts, k, window=16):
        """Signed distance (m) from each point (N, 2) to the nearest edge, + = on the track,
        - = past the edge. Only looks at edge pieces near centreline sample k (fast)."""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        out = np.full(len(pts), np.inf)
        for poly, near, side in ((self.outer, self._near_out[k], self._side_out),
                                 (self.inner, self._near_in[k], self._side_in)):
            idx = (near + np.arange(-window, window)) % len(poly)
            a = poly[idx]
            ab = poly[(idx + 1) % len(poly)] - a
            ap = pts[:, None, :] - a[None]
            t = np.clip(np.einsum("nmk,mk->nm", ap, ab) / np.einsum("mk,mk->m", ab, ab), 0.0, 1.0)
            d = ap - t[..., None] * ab[None]
            dist = np.sqrt(np.einsum("nmk,nmk->nm", d, d))
            j = dist.argmin(axis=1)
            r = np.arange(len(pts))
            cross = ab[j, 0] * ap[r, j, 1] - ab[j, 1] * ap[r, j, 0]
            out = np.minimum(out, dist[r, j] * np.where(cross * side >= 0, 1.0, -1.0))
        return out

    def index_at(self, s):
        return np.round(np.asarray(s) / self.length * self.n).astype(int) % self.n


def _signed_area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
