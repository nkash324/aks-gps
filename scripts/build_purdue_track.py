#!/usr/bin/env python3
"""Build a MuJoCo track world from the RTK edge survey of the Purdue kart track.

  python scripts/build_purdue_track.py
  python scripts/build_purdue_track.py --survey "C:/AKS RL/data/purdue_rtk_survey" --out tracks/purdue_rtk

Reads "purdue left.csv" and "purdue right.csv" (RTK points along each track edge,
2021-06-01) and writes <out>/track_world.xml in the same layout as the team's world:
line0_* = outer edge, line1_* = inner edge, a "start_pose" and the kart from
line_robot.xml (copied from the team sim, which is only read). Use it with
track_world: tracks/purdue_rtk/track_world.xml in config.local.yaml.

Frame: x = east, y = north (m), origin = the start point on the start line.
The kart starts on the start line, centred, facing counter-clockwise (seen from above).
The pit lane files are not used (the race does not use the outer loop).
"""
import argparse
import csv
import math
import pathlib
import shutil
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
ROOT = pathlib.Path(__file__).resolve().parent.parent

EARTH_R = 6378137.0
# Centre of the painted checkered start/finish line on the long bottom straight (next to the
# pit lane), located on aerial imagery aligned to the survey.
START_LATLON = (40.4376892, -86.9444293)
PIECE = 0.5                              # m between line pieces (same as the team world)
LINE_HALF_WIDTH = 0.075                  # 15 cm painted line, like the team world


def read_edge(path):
    """-> (N, 2) lat/lon of the RTK FIX points, in survey order."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    pts = [(float(r["Latitude"]), float(r["Longitude"])) for r in rows if r["Solution status"] == "FIX"]
    dropped = len(rows) - len(pts)
    if dropped:
        print(f"{pathlib.Path(path).name}: dropped {dropped} non-FIX point(s)")
    return np.array(pts)


def to_local(latlon, origin):
    """Lat/lon -> local east/north metres around origin (flat-earth; < 1 mm error over 200 m)."""
    lat0, lon0 = origin
    x = np.radians(latlon[:, 1] - lon0) * EARTH_R * math.cos(math.radians(lat0))
    y = np.radians(latlon[:, 0] - lat0) * EARTH_R
    return np.column_stack([x, y])


def drop_backtracks(P, name):
    """Remove survey points recorded out of order (the edge doubles back on itself there).
    At each reversal, drop the point lying furthest from the chord between its neighbours."""
    def seg_dist(q, a, b):
        ab = b - a
        t = np.clip(np.dot(q - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
        return np.linalg.norm(q - (a + t * ab))
    while True:
        n = len(P)
        cand = []
        for i in range(n):
            a, b = P[i] - P[i - 1], P[(i + 1) % n] - P[i]
            if np.dot(a, b) < -0.5 * np.linalg.norm(a) * np.linalg.norm(b):      # turns back > 120 deg
                cand.append((seg_dist(P[i], P[i - 1], P[(i + 1) % n]), i))
        if not cand:
            return P
        d, i = max(cand)
        print(f"{name}: dropped out-of-order point at ({P[i][0]:.1f}, {P[i][1]:.1f}), {d:.1f} m off the edge")
        P = np.delete(P, i, axis=0)


def signed_area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def catmull_rom_closed(P, per_m=10):
    """Centripetal Catmull-Rom through every point of a closed loop (passes through the
    survey points, so corners are not cut). Returns a dense polyline."""
    n = len(P)
    out = []
    for i in range(n):
        p0, p1, p2, p3 = P[i - 1], P[i], P[(i + 1) % n], P[(i + 2) % n]
        t0 = 0.0
        t1 = t0 + np.linalg.norm(p1 - p0) ** 0.5
        t2 = t1 + np.linalg.norm(p2 - p1) ** 0.5
        t3 = t2 + np.linalg.norm(p3 - p2) ** 0.5
        k = max(2, int(np.linalg.norm(p2 - p1) * per_m))
        for t in np.linspace(t1, t2, k, endpoint=False):
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    return np.array(out)


def resample_closed(P, step):
    Pc = np.vstack([P, P[:1]])
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(Pc, axis=0), axis=1))]
    n = max(3, int(round(s[-1] / step)))
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, s, Pc[:, 0]), np.interp(t, s, Pc[:, 1])]), s[-1]


def line_geoms(P, prefix):
    """One flat box per edge segment, like the team world's line chains."""
    out = []
    nxt = np.roll(P, -1, axis=0)
    for i, (a, b) in enumerate(zip(P, nxt)):
        c, d = (a + b) / 2, b - a
        half = np.linalg.norm(d) / 2 + 0.01           # tiny overlap so the line has no cracks
        yaw = math.degrees(math.atan2(d[1], d[0]))
        out.append(f'    <geom name="{prefix}{i}" class="line" pos="{c[0]:.3f} {c[1]:.3f} 0.002" '
                   f'size="{half:.3f} {LINE_HALF_WIDTH} 0.002" euler="0 0 {yaw:.2f}"/>')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default=str(ROOT / "tracks" / "purdue_rtk" / "line_robot.xml"),
                    help="line_robot.xml the world includes (default: the copy in tracks/purdue_rtk)")
    ap.add_argument("--survey", default=str(ROOT.parent / "data" / "purdue_rtk_survey"),
                    help="folder with 'purdue left.csv' and 'purdue right.csv'")
    ap.add_argument("--out", default="tracks/purdue_rtk", help="output folder (relative to AKS-RL)")
    args = ap.parse_args()

    survey = pathlib.Path(args.survey)
    out = pathlib.Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    edges = [catmull_rom_closed(drop_backtracks(to_local(read_edge(survey / f"purdue {side}.csv"), START_LATLON),
                                                f"purdue {side}.csv"))
             for side in ("left", "right")]
    # Outer edge = the one enclosing more area. Both are put in counter-clockwise order,
    # which is the race direction.
    edges.sort(key=lambda P: -abs(signed_area(P)))
    edges = [P if signed_area(P) > 0 else P[::-1] for P in edges]
    (outer, l_out), (inner, l_in) = (resample_closed(P, PIECE) for P in edges)

    # Start pose: the start point, facing along the counter-clockwise outer edge there.
    i = int(np.argmin(np.linalg.norm(outer, axis=1)))
    t = outer[(i + 2) % len(outer)] - outer[i - 2]
    yaw = math.atan2(t[1], t[0])

    allp = np.vstack([outer, inner])
    lo, hi = allp.min(axis=0) - 10.0, allp.max(axis=0) + 10.0
    centre, half = (lo + hi) / 2, (hi - lo) / 2
    extent = float(np.linalg.norm(hi - lo))
    tex = np.maximum(1, np.round(half)).astype(int)

    header = f"""<!--
  Purdue kart track, built by scripts/build_purdue_track.py from the RTK edge survey
  of 2021-06-01 ({survey.name}). Do not edit by hand; re-run the script.

  - Frame: x = east, y = north (m). Origin (0, 0) = the start point on the start line,
    lat/lon {START_LATLON[0]}, {START_LATLON[1]} (also stored as "gnss_origin" below).
  - line0_* = outer edge ({l_out:.0f} m), line1_* = inner edge ({l_in:.0f} m), one piece
    every {PIECE} m, drawn along the surveyed edge (15 cm wide). Visual only, no collision.
  - start_pose = rear axle on the start line, centred, heading counter-clockwise.
  - The kart comes from line_robot.xml, copied from the team sim.
-->"""
    xml = [header, '<mujoco model="purdue_rtk_track">',
           '  <compiler angle="degree"/>',
           '  <option timestep="0.002" integrator="implicitfast"/>', '',
           '  <visual>',
           '    <headlight ambient="0.4 0.4 0.4" diffuse="0.5 0.5 0.5" specular="0 0 0"/>',
           f'    <map znear="{0.02 / extent:.6f}" zfar="{300.0 / extent:.4f}"/>',
           '    <global offwidth="640" offheight="480"/>',
           '  </visual>', '',
           f'  <statistic extent="{extent:.1f}" center="{centre[0]:.2f} {centre[1]:.2f} 0"/>', '',
           '  <custom>',
           f'    <numeric name="start_pose" data="0.000 0.000 {yaw:.4f}"/>',
           f'    <numeric name="gnss_origin" data="{START_LATLON[0]} {START_LATLON[1]}"/>',
           '  </custom>', '',
           '  <asset>',
           '    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.55 0.7 0.9" rgb2="0.95 0.95 1.0" width="256" height="256"/>',
           '    <texture name="asphalt" type="2d" builtin="checker" rgb1="0.20 0.20 0.21" rgb2="0.23 0.23 0.24" width="64" height="64"/>',
           f'    <material name="asphalt_mat" texture="asphalt" texrepeat="{tex[0]} {tex[1]}"/>',
           '    <material name="line_mat" rgba="0.95 0.95 0.95 1"/>',
           '  </asset>', '',
           '  <default>',
           '    <default class="line">',
           '      <geom type="box" material="line_mat" contype="0" conaffinity="0"/>',
           '    </default>',
           '  </default>', '',
           '  <worldbody>',
           '    <light name="sun" directional="true" pos="0 0 10" dir="-0.3 0.2 -1" diffuse="0.6 0.6 0.6" castshadow="false"/>',
           f'    <geom name="floor" type="plane" pos="{centre[0]:.2f} {centre[1]:.2f} 0" '
           f'size="{half[0]:.1f} {half[1]:.1f} 0.1" material="asphalt_mat"/>']
    xml += line_geoms(outer, "line0_") + line_geoms(inner, "line1_")
    xml += ['  </worldbody>', '', '  <include file="line_robot.xml"/>', '</mujoco>', '']
    (out / "track_world.xml").write_text("\n".join(xml), encoding="utf-8")

    robot = pathlib.Path(args.robot)
    if robot.resolve() != (out / "line_robot.xml").resolve():
        shutil.copyfile(robot, out / "line_robot.xml")
    print(f"outer edge {l_out:.1f} m ({len(outer)} pieces), inner edge {l_in:.1f} m ({len(inner)} pieces)")
    print(f"start pose (0, 0), heading {math.degrees(yaw):.1f} deg")
    print(f"Wrote {out / 'track_world.xml'} (+ line_robot.xml from {robot.parent})")


if __name__ == "__main__":
    main()
