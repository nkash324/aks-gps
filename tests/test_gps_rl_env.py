import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from kart_gps.config import load_config, track_path  # noqa: E402
from kart_gps.drivers import build_line, make_driver  # noqa: E402
from kart_gps.env import LapEnv  # noqa: E402
from kart_gps.rl_env import ResidualEnv  # noqa: E402
from kart_gps.track import TrackMap  # noqa: E402

CFG = load_config()
TRACK = TrackMap(track_path(CFG))
LINE = build_line("optimised", TRACK, CFG)


def test_zero_action_is_driver_b():
    """With a zero correction the residual env drives exactly like plain driver B."""
    env = ResidualEnv(CFG, track=TRACK, line=LINE, level="realistic", stress=False)
    obs, _ = env.reset(seed=0, options={"lap_seed": 7})
    lap = LapEnv(CFG, level="realistic", seed=7, track=TRACK)
    drv = make_driver("pp_optimised", TRACK, CFG, {"optimised": LINE})
    est = lap.reset()
    drv.reset()
    for _ in range(300):
        est, done_b, info_b = lap.step(*drv.act(est))
        obs, r, done, _, info = env.step(np.zeros(2))
        assert info["progress"] == info_b["progress"]
        assert np.isfinite(obs).all() and obs.shape == env.observation_space.shape
        assert done == done_b
        if done:
            break


def test_lap_return_tracks_lap_time():
    """A zero-correction lap returns about: track length - time * steps + lap bonus."""
    rw = CFG["residual"]["reward"]
    env = ResidualEnv(CFG, track=TRACK, line=LINE, level="realistic", stress=False)
    env.reset(seed=1, options={"lap_seed": 3})
    total, done, n = 0.0, False, 0
    while not done:
        _, r, done, _, info = env.step(np.zeros(2))
        total += r
        n += 1
    assert info["end"] == "lap"
    no_edge = rw["progress"] * info["progress"] - rw["time"] * n + rw["lap"]
    assert no_edge - 20 < total <= no_edge + 1e-6
