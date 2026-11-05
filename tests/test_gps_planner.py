import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from kart_gps.config import load_config, track_path  # noqa: E402
from kart_gps.drivers import build_line  # noqa: E402
from kart_gps.planner import kart_half_width  # noqa: E402
from kart_gps.track import TrackMap  # noqa: E402

CFG = load_config()
TRACK = TrackMap(track_path(CFG))


def test_optimised_line_is_shorter_and_keeps_the_buffer():
    c = build_line("centreline", TRACK, CFG)
    o = build_line("optimised", TRACK, CFG)
    assert o.length < c.length - 10
    assert o.lap_estimate < c.lap_estimate
    need = kart_half_width(CFG["kart"]) + CFG["planner"]["buffer"]
    assert TRACK.edge_distance(o.pts).min() > need - 0.02


def test_speed_plan_respects_limits():
    o = build_line("optimised", TRACK, CFG)
    pl = CFG["planner"]
    assert o.speed.max() <= pl["top_speed"] + 1e-9
    assert np.all(o.speed ** 2 * np.abs(o.curvature) <= pl["lat_accel"] + 1e-6)
