#!/usr/bin/env python3
"""Watch a GPS-stack driver race on the MuJoCo kart (kart_gps/mj_kart.py, aks_kart.xml on
the surveyed Purdue track). Everything except the physics is the real stack: sensors with
noise, EKF, low-level controller, driver B and the network's corrections.

  python scripts/gps/play_mujoco.py                         # best model (G2 460.1M), realistic, seed 0
  python scripts/gps/play_mujoco.py --driver b              # driver B alone (no network)
  python scripts/gps/play_mujoco.py --level harsh --seed 4 --laps 3
  python scripts/gps/play_mujoco.py --ckpt runs/gpu/<run>/ckpt/<steps>
  python scripts/gps/play_mujoco.py --raw                   # untuned MuJoCo tyre contact
  python scripts/gps/play_mujoco.py --no-view               # no window, just the lap numbers
  python scripts/gps/play_mujoco.py --ghosts                # + drivers A and B as see-through karts
  python scripts/gps/play_mujoco.py --level harsh --sensor-ghosts   # + the same driver at the other levels

Before each lap it drives the same lap (same seed = same kart and sensor errors) in the
2D sim the model was trained in, so the overlay shows MuJoCo vs 2D. The dots on the track
are the optimised line, coloured by its planned speed (blue slow, red fast).
--ghosts also drives the same lap with driver A (centreline, blue ghost) and driver B
(optimised line, green ghost), each on its own MuJoCo kart, and draws them over the live one.
--sensor-ghosts does the same with the live driver itself at the other sensor levels
(perfect white, realistic yellow, harsh red). Then every kart, the live one too, is drawn
the same way: in its level's colour, mostly solid. The two can be combined.
Camera: --cam chase (behind the kart), top (from above) or free (mouse); in the window 8 = chase,
9 = top, Esc = free, 6 = follow the next kart (live, then each ghost). 7 hides / shows the
info panel. Close the window to stop.
"""
import argparse
import math
import pathlib
import pickle
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kart_gps.config import load_config, track_path  # noqa: E402
from kart_gps.drivers import build_line, make_driver  # noqa: E402
from kart_gps.env import LapEnv  # noqa: E402
from kart_gps.rl_env import ResidualEnv  # noqa: E402
from kart_gps.track import TrackMap  # noqa: E402

BEST = ROOT / "runs" / "gpu" / "20261029-234306-run2" / "ckpt" / "460062720"     # G2 460.1M
RAW = {"cone": "pyramidal", "front_friction_scale": 1.0}


def load_policy(ckpt):
    from stable_baselines3 import PPO
    model = PPO.load(pathlib.Path(ckpt) / "model.zip", device="cpu",
                     custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0})
    with open(pathlib.Path(ckpt) / "vecnormalize.pkl", "rb") as f:
        norm = pickle.load(f)
    return lambda obs: np.clip(model.predict(norm.normalize_obs(obs[None]), deterministic=True)[0][0], -1, 1)


def run_lap(env, pol, seed, on_step=None):
    """One lap; on_step(env, action, info) after every 1/20 s. -> last info."""
    obs, _ = env.reset(seed=0, options={"lap_seed": seed})
    while True:
        a = pol(obs)
        obs, _, done, _, info = env.step(a)
        if on_step:
            on_step(env, a, info)
        if done:
            return info


def ghost_lap(env, drv, seed):
    """Driver A or B alone on its own kart. -> (qpos after every 1/20 s, last info)."""
    est = env.reset(seed=seed)
    drv.reset()
    poses = []
    while True:
        est, done, info = env.step(*drv.act(est))
        poses.append(env.kart.d.qpos.copy())
        if done:
            return np.array(poses), info


def ghost_policy_lap(env, pol, seed):
    """The live driver on its own kart (another sensor level). -> (qpos after every 1/20 s, last info)."""
    poses = []
    info = run_lap(env, pol, seed, lambda env, a, info: poses.append(env.env.kart.d.qpos.copy()))
    return np.array(poses), info


def fmt_end(info):
    return f"{info['lap_time']:.2f} s" if info["end"] == "lap" else f"{info['end']} at {info['progress']:.0f} m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", choices=("model", "b"), default="model")
    ap.add_argument("--ckpt", default=str(BEST), help="checkpoint folder (model.zip + vecnormalize.pkl)")
    ap.add_argument("--level", choices=("perfect", "realistic", "harsh"), default="realistic")
    ap.add_argument("--seed", type=int, default=0, help="first lap's seed (kart + sensor errors)")
    ap.add_argument("--laps", type=int, default=1)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    ap.add_argument("--cam", choices=("chase", "top", "free"), default="chase")
    ap.add_argument("--raw", action="store_true", help="untuned tyre contact (pyramidal cone, front grip x1)")
    ap.add_argument("--no-view", action="store_true", help="no window; print MuJoCo vs 2D lap numbers")
    ap.add_argument("--ghosts", action="store_true", help="also show drivers A and B as see-through karts")
    ap.add_argument("--sensor-ghosts", action="store_true",
                    help="also show the same driver at the other sensor levels as see-through karts")
    args = ap.parse_args()

    mj_over = {"physics": "mujoco", "mujoco": dict(RAW if args.raw else {}, lines=not args.no_view)}
    cfg2d, cfgmj = load_config(), load_config(overrides=mj_over)
    track = TrackMap(track_path(cfg2d))
    line = build_line("optimised", track, cfg2d)
    env2d = ResidualEnv(cfg2d, track=track, line=line, level=args.level, stress=False)
    envmj = ResidualEnv(cfgmj, track=track, line=line, level=args.level, stress=False)
    if args.driver == "b":
        name, pol = "driver B (no network)", (lambda obs: np.zeros(2))
    else:
        name, pol = f"model {pathlib.Path(args.ckpt).name}", load_policy(args.ckpt)
        if pathlib.Path(args.ckpt).resolve() == BEST.resolve():
            name = "model G2 460.1M"
    phys = "MuJoCo " + ("raw" if args.raw else "tuned")

    if args.no_view:
        for seed in range(args.seed, args.seed + args.laps):
            i2 = run_lap(env2d, pol, seed)
            t0 = time.time()
            stats = {"edge": math.inf, "lat": 0.0}

            def track_stats(env, a, info):
                stats["edge"] = min(stats["edge"], info["clear"])
                stats["lat"] = max(stats["lat"], abs(env.env.kart.ay) / 9.81)
            im = run_lap(envmj, pol, seed, track_stats)
            print(f"{name}, {args.level}, seed {seed}: 2D {fmt_end(i2)} | {phys} {fmt_end(im)} "
                  f"(closest edge {stats['edge']:.2f} m, max lateral {stats['lat']:.2f} g, "
                  f"computed in {time.time() - t0:.0f} s)", flush=True)
        return

    import mujoco
    import mujoco.viewer

    envmj.reset(seed=0, options={"lap_seed": args.seed})       # builds the scene
    kart = envmj.env.kart
    m, d = kart.m, kart.d
    hud = {"2d": "-", "last": "-"}
    # ghosts run on a scene without the edge markers, so they get their own MjModel / MjData
    ghost_runs = []                    # (label, rgba, seed -> (poses, last info))
    alpha = 0.8 if args.sensor_ghosts else 0.45
    levels = {"perfect": ("white", [0.95, 0.95, 0.95]), "realistic": ("yellow", [1.0, 0.85, 0.1]),
              "harsh": ("red", [1.0, 0.2, 0.2])}
    live_rgba = np.array(levels[args.level][1] + [alpha], dtype=np.float32) if args.sensor_ghosts else None
    cfgg = load_config(overrides=dict(mj_over, mujoco=dict(mj_over["mujoco"], lines=False)))
    if args.ghosts:
        genv = LapEnv(cfgg, level=args.level, track=track)
        drv_a = make_driver("pp_centre", track, cfgg)
        ghost_runs.append(("Driver A (blue)", np.array([0.2, 0.5, 1.0, alpha], dtype=np.float32),
                           lambda seed: ghost_lap(genv, drv_a, seed)))
        if args.driver != "b":
            drv_b = make_driver("pp_optimised", track, cfgg, {"optimised": line})
            ghost_runs.append(("Driver B (green)", np.array([0.2, 0.9, 0.3, alpha], dtype=np.float32),
                               lambda seed: ghost_lap(genv, drv_b, seed)))
    if args.sensor_ghosts:
        who = "Driver B" if args.driver == "b" else "Model"
        for lvl, (colour, rgb) in levels.items():
            if lvl != args.level:
                lenv = ResidualEnv(cfgg, track=track, line=line, level=lvl, stress=False)
                ghost_runs.append((f"{who}, {lvl} sensors ({colour})", np.array(rgb + [alpha], dtype=np.float32),
                                   lambda seed, lenv=lenv: ghost_policy_lap(lenv, pol, seed)))
    font, grid = mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT

    # 8 = chase, 9 = top, Esc = free, 6 = next kart to follow, 7 = info panel on / off (every
    # letter and 0-5 are taken by the viewer's render toggles).
    # The camera is a free camera whose look-at point is moved onto the followed kart each frame
    # (ghosts aren't bodies, so a tracking camera can't follow them); scroll zooms and drag
    # rotates. Chase turns with the kart by adding its yaw change, which keeps the user's angle.
    cams = {"chase": (5.5, -18.0), "top": (25.0, -89.0)}
    cam_keys = {ord("8"): "chase", ord("9"): "top"}
    view = {"cam": args.cam, "yaw": None, "ghosts": [], "i": 0, "hud": True, "target": 0}
    rate = cfgmj["control"]["rate_hz"]
    q0 = kart.q_free

    def target_pose():
        """Position and heading of the followed kart: 0 = live, 1.. = ghosts in info-panel order."""
        t = view["target"]
        if 0 < t <= len(view["ghosts"]):
            poses = view["ghosts"][t - 1][2]
            q = poses[min(view["i"], len(poses)) - 1]
        else:
            q = d.qpos
        w, x, y, z = q[q0 + 3:q0 + 7]
        return q[q0:q0 + 3].copy(), math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def set_cam(viewer, which):
        pos, yaw = target_pose()
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = pos
        viewer.cam.distance, viewer.cam.elevation = cams[which]
        viewer.cam.azimuth = math.degrees(yaw) if which == "chase" else 90.0
        view["cam"], view["yaw"] = which, yaw

    def on_key(key):
        if key == 256:                     # Esc: the viewer frees the camera, so stop following
            view["cam"] = "free"
        elif key == ord("7"):
            view["hud"] = not view["hud"]
        elif key == ord("6"):
            view["target"] = (view["target"] + 1) % (len(view["ghosts"]) + 1)
            if view["cam"] == "chase" and "v" in view:
                set_cam(view["v"], "chase")
        elif key in cam_keys and "v" in view:
            set_cam(view["v"], cam_keys[key])

    with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False,
                                      key_callback=on_key) as viewer:
        view["v"] = viewer
        # optimised line, coloured by planned speed
        scn = viewer.user_scn
        idx = np.arange(0, len(line.pts), max(1, len(line.pts) // min(scn.maxgeom - 10, 500)))
        vlo, vhi = float(line.speed.min()), float(line.speed.max())
        for n, i in enumerate(idx):
            f = (line.speed[i] - vlo) / max(vhi - vlo, 1e-6)
            mujoco.mjv_initGeom(scn.geoms[n], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.06, 0, 0]),
                                np.array([line.pts[i, 0], line.pts[i, 1], 0.02]), np.eye(3).ravel(),
                                np.array([f, 0.3, 1 - f, 0.9], dtype=np.float32))
        scn.ngeom = n_dots = len(idx)
        if args.cam != "free":
            set_cam(viewer, args.cam)
        gd, gopt, gpert = mujoco.MjData(m), mujoco.MjvOption(), mujoco.MjvPerturb()
        if live_rgba is not None:
            # hide the real kart (its shown geoms move to group 5, which the window doesn't draw)
            # and draw it like the ghosts instead
            mine = (m.body_rootid[m.geom_bodyid] == kart.body) & gopt.geomgroup[np.clip(m.geom_group, 0, 5)].astype(bool)
            sites = (m.body_rootid[m.site_bodyid] == kart.body) & gopt.sitegroup[np.clip(m.site_group, 0, 5)].astype(bool)
            with viewer.lock():
                m.geom_group[mine] = 5
                m.site_group[sites] = 5
                viewer.opt.geomgroup[5] = viewer.opt.sitegroup[5] = 0
            gopt.geomgroup[5] = gopt.sitegroup[5] = 1

        def draw_ghosts():
            scn.ngeom = n_dots
            if live_rgba is not None:
                n0 = scn.ngeom
                mujoco.mjv_addGeoms(m, d, gopt, gpert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
                for g in range(n0, scn.ngeom):
                    scn.geoms[g].rgba = live_rgba
            for _, rgba, poses, _ in view["ghosts"]:
                gd.qpos[:] = poses[min(view["i"], len(poses)) - 1]
                mujoco.mj_kinematics(m, gd)
                n0 = scn.ngeom
                mujoco.mjv_addGeoms(m, gd, gopt, gpert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
                for g in range(n0, scn.ngeom):
                    scn.geoms[g].rgba = rgba

        def show_texts():
            if view["hud"]:
                viewer.set_texts(view["texts"])
            else:
                viewer.clear_texts()

        def show(env, a, info, lap_t):
            k = env.env.kart
            est = env.est
            sb, kb = env.base
            ll = env.env.ll
            left = "\n".join(["Camera (8 / 9 / Esc, 6 = next kart)", "Driver", "Physics", "Lap seed / sensors", "Lap time", "Same lap in 2D sim",
                              "Speed (true / EKF)", "B's target speed", "Network speed change", "Network curvature change",
                              "Steering (wheels)", "Lateral accel", "Closest tyre to edge", "GNSS",
                              "Position error (EKF)", "Previous lap (MuJoCo)"]
                             + [g[0] for g in view["ghosts"]])
            rx, ry = k.rear_axle
            karts = [f"{name}, {args.level}"] + [g[0] for g in view["ghosts"]]
            right = "\n".join([
                f"{view['cam']}" + ("" if view["cam"] == "free" else f" on {karts[min(view['target'], len(karts) - 1)]}"),
                name, phys, f"{env.info0['lap_seed']} / {args.level}"
                + (f" ({levels[args.level][0]})" if live_rgba is not None else ""), f"{lap_t:6.2f} s", hud["2d"],
                f"{k.speed:5.2f} / {est['v']:5.2f} m/s", f"{sb:5.2f} m/s",
                f"{a[0] * env.rc['max_dv']:+5.2f} m/s", f"{a[1] * env.rc['max_dk']:+6.3f} 1/m",
                f"{math.degrees(k.delta):+5.1f} deg", f"{k.ay / 9.81:+5.2f} g", f"{info['clear']:5.2f} m",
                f"{'RTK fixed' if est['rtk_fixed'] else 'float / lost'}, age {est['gnss_age']:.2f} s",
                f"{math.hypot(est['x'] - rx, est['y'] - ry) * 100:4.1f} cm", hud["last"]]
                + [fmt_end(g[3]) if view["i"] >= len(g[2]) else f"{view['i'] / rate:6.2f} s"
                   for g in view["ghosts"]])
            view["texts"] = [(font, grid, left, right)]
            show_texts()
            if view["ghosts"] or live_rgba is not None:
                with viewer.lock():
                    draw_ghosts()
            if view["cam"] != "free":
                pos, yaw = target_pose()
                viewer.cam.lookat[:] = pos
                if view["cam"] == "chase" and view["yaw"] is not None:
                    viewer.cam.azimuth += math.degrees((yaw - view["yaw"] + math.pi) % (2 * math.pi) - math.pi)
                view["yaw"] = yaw
            viewer.sync()

        seed = args.seed
        while viewer.is_running() and seed < args.seed + args.laps:
            hud["2d"] = "computing..."
            viewer.set_texts([(font, grid, "Same lap in 2D sim", "computing...")])
            viewer.sync()
            hud["2d"] = fmt_end(run_lap(env2d, pol, seed))
            if ghost_runs:
                viewer.set_texts([(font, grid, "Ghost laps", "computing...")])
                viewer.sync()
                view["ghosts"] = [(label, rgba, *run(seed)) for label, rgba, run in ghost_runs]
            obs, _ = envmj.reset(seed=0, options={"lap_seed": seed})
            view["i"] = 0
            t_wall = time.time()
            sim_t = 0.0
            n_end = max((len(g[2]) for g in view["ghosts"]), default=0)
            done = False
            while viewer.is_running():
                if not done:                   # once finished, the kart stays where it stopped
                    a = pol(obs)
                    with viewer.lock():
                        obs, _, done, _, info = envmj.step(a)
                    if done:
                        print(f"{name}, {args.level}, seed {seed}: 2D {hud['2d']} | {phys} {fmt_end(info)}",
                              flush=True)
                view["i"] += 1
                sim_t += 1.0 / rate
                show(envmj, a, info, envmj.env.kart.t - envmj.env.t0)
                wait = t_wall + sim_t / args.speed - time.time()
                if wait > 0:
                    time.sleep(wait)
                if done and view["i"] >= n_end:    # every ghost has finished too
                    hud["last"] = f"seed {seed}: {fmt_end(info)}"
                    show(envmj, a, info, envmj.env.kart.t - envmj.env.t0)
                    break
            seed += 1
        while viewer.is_running():            # keep the last frame up until the window is closed
            show_texts()
            viewer.sync()
            time.sleep(0.1)


if __name__ == "__main__":
    main()
