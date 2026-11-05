import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from kart_gps.config import load_config, track_path  # noqa: E402
from kart_gps.env import LapEnv  # noqa: E402
from kart_gps.track import TrackMap  # noqa: E402

CFG = load_config()
TRACK = TrackMap(track_path(CFG))


def test_track_map():
    assert 425 < TRACK.length < 435
    width = TRACK.hw_left + TRACK.hw_right
    assert 4.9 < width.min() and width.max() < 6.8
    assert TRACK.inside(TRACK.pts).all()


def test_clearance_matches_polygon_test():
    rng = np.random.default_rng(0)
    for k in rng.integers(0, TRACK.n, 200):
        pts = TRACK.pts[k] + TRACK.normal[k] * rng.uniform(-4, 4, (16, 1))
        assert ((TRACK.clearance(pts, k) >= 0) == TRACK.inside(pts)).all()


def _pursuit(env, est, k, speed=6.0, ahead=4.0):
    tr = env.track
    k, _, _ = tr.project(est["x"], est["y"], hint=k)
    j = (k + int(ahead / 0.25)) % tr.n
    dx, dy = tr.pts[j] - [est["x"], est["y"]]
    c, s = math.cos(est["yaw"]), math.sin(est["yaw"])
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    return k, (speed, 2 * ly / (lx * lx + ly * ly))


def test_slow_lap_completes_and_ekf_tracks():
    env = LapEnv(CFG, level="realistic", seed=2, track=TRACK)
    est, k, errs = env.reset(), None, []
    while True:
        k, cmd = _pursuit(env, est, k)
        est, done, info = env.step(*cmd)
        t = info["true"]
        errs.append(math.hypot(est["x"] - t["x"], est["y"] - t["y"]))
        if done:
            break
    assert info["end"] == "lap"
    assert 70 < info["lap_time"] < 80
    # error = map offset (<= 0.42 m) + GNSS noise
    assert np.percentile(errs, 95) < 0.5


def test_driving_straight_goes_off_track():
    env = LapEnv(CFG, level="perfect", seed=0, track=TRACK)
    env.reset()
    for _ in range(400):
        est, done, info = env.step(6.0, 0.0)
        if done:
            break
    assert info["end"] == "offtrack"
    assert info["min_clearance"] < 0


def test_same_seed_same_run():
    a = LapEnv(CFG, level="harsh", seed=5, track=TRACK)
    b = LapEnv(CFG, level="harsh", seed=5, track=TRACK)
    ea, eb = a.reset(), b.reset()
    for _ in range(40):
        ea, _, _ = a.step(5.0, 0.05)
        eb, _, _ = b.step(5.0, 0.05)
    assert ea == eb
