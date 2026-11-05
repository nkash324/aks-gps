#!/usr/bin/env python3
"""Train the residual driver (PPO): corrections on top of driver B. Settings: gps.yaml 'residual'.

  python scripts/gps/train_residual.py --name overnight --eval-after
  python scripts/gps/train_residual.py --resume runs/gps/residual/<run>/last

Curriculum: realistic sensors only until --curriculum-at, then the harsh share of laps
ramps linearly to gps.yaml 'mix.harsh' over --ramp steps and holds there.
The learning rate decays linearly to 0 at --steps (a paused / resumed run keeps the schedule).

Every --eval-every steps it saves ckpt/<steps>/. scripts/gps/eval_residual.py evaluates saved
checkpoints in a separate batch job (--eval-after starts it when training ends, so training
never waits). With --eval-envs N it instead drives the fixed eval laps itself (realistic, + harsh once training sees harsh; never used in training) with no
exploration noise, and keeps the best checkpoint (most laps finished, then fastest).
--pause-at N saves and stops at N steps; --resume continues in the same run folder.
Stops and saves when free RAM drops under --min-free-gb. Output: runs/gps/residual/<time>-<name>/
"""
import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from kart_gps.config import ROOT, load_config  # noqa: E402

EVAL_SEEDS = {"realistic": list(range(500000, 500020)), "harsh": list(range(600000, 600010))}


def make_env(i, level=None, stress=None, seed_base=0):
    def f():
        from kart_gps.config import load_config
        from kart_gps.rl_env import ResidualEnv
        env = ResidualEnv(load_config(), level=level, stress=stress)
        env.reset(seed=seed_base + 10_000 * (i + 1))
        return env
    return f


def linear_lr(lr0):
    return lambda progress_remaining: lr0 * progress_remaining


def harsh_share(steps, args, mix):
    """Curriculum: 0 until --curriculum-at, then a linear ramp to mix['harsh'] over --ramp."""
    f = (steps - args.curriculum_at) / max(args.ramp, 1)
    return mix["harsh"] * float(np.clip(f, 0.0, 1.0))


def eval_levels(steps, args):
    """Harsh laps join the eval only once training has seen harsh laps: from the first eval
    after --curriculum-at (the eval at the pause point itself stays realistic-only)."""
    return tuple(EVAL_SEEDS) if steps >= args.curriculum_at + args.eval_every // 2 else ("realistic",)


def evaluate(venv, policy_fn, levels=tuple(EVAL_SEEDS)):
    """Drive every EVAL_SEEDS lap of these levels once. policy_fn(raw obs batch) -> actions."""
    return drive(venv, policy_fn, [(lv, s) for lv in levels for s in EVAL_SEEDS[lv]])


def drive(venv, policy_fn, jobs):
    """Drive one lap per (level, lap seed) job, num_envs at a time."""
    n = venv.num_envs
    rows = []
    for b in range(0, len(jobs), n):
        batch = jobs[b:b + n]
        obs = np.zeros((n,) + venv.observation_space.shape, np.float32)
        for i, (lv, s) in enumerate(batch):
            venv.env_method("set_level", lv, indices=[i])
            obs[i] = venv.env_method("reset", options={"lap_seed": s}, indices=[i])[0][0]
        live = [i < len(batch) for i in range(n)]
        act_abs = np.zeros(n)
        steps = np.zeros(n)
        while any(live):
            a = policy_fn(obs)
            obs, _, dones, infos = venv.step(a)
            for i in range(len(batch)):
                if live[i]:
                    act_abs[i] += np.abs(a[i]).mean()
                    steps[i] += 1
                    if dones[i]:
                        live[i] = False
                        inf = infos[i]
                        rows.append({"level": batch[i][0], "seed": batch[i][1], "end": inf["end"],
                                     "lap_time": inf["lap_time"], "progress": inf["progress"],
                                     "min_clearance": inf["min_clearance"],
                                     "mean_abs_action": act_abs[i] / steps[i]})
    return rows


def summary(rows):
    out = {}
    for lv in dict.fromkeys(r["level"] for r in rows):
        g = [r for r in rows if r["level"] == lv]
        laps = [r["lap_time"] for r in g if r["end"] == "lap"]
        out[lv] = {"finished": len(laps), "of": len(g),
                   "lap_mean": float(np.mean(laps)) if laps else None,
                   "min_clearance_worst": float(min(r["min_clearance"] for r in g)),
                   "mean_abs_action": float(np.mean([r["mean_abs_action"] for r in g]))}
    return out


def score(s):
    """Higher is better: every finished lap first, then lap time."""
    s = {lv: s[lv] for lv in EVAL_SEEDS if lv in s}
    fin = sum(v["finished"] for v in s.values())
    t = np.mean([v["lap_mean"] for v in s.values() if v["lap_mean"] is not None] or [999])
    return fin * 1000 - t


def fmt(s):
    return "  ".join(f"{lv}: {v['finished']}/{v['of']} laps, {v['lap_mean'] or float('nan'):.2f} s, "
                     f"closest edge {v['min_clearance_worst']:+.2f} m, |a| {v['mean_abs_action']:.2f}"
                     for lv, v in s.items())


class Trainer(BaseCallback):
    def __init__(self, run, eval_env, args, mix, history):
        super().__init__()
        self.run, self.eval_env, self.args, self.mix = run, eval_env, args, mix
        self.history = history
        self.next_eval = None
        self.stop = False
        self.paused = False
        self.ep = []
        self.t0 = time.time()
        self.steps0 = None

    def _save(self, name):
        d = self.run / name
        d.mkdir(parents=True, exist_ok=True)
        self.model.save(d / "model.zip")
        self.training_env.save(str(d / "vecnormalize.pkl"))
        (d / "saved").write_text(str(self.num_timesteps))   # complete: eval_residual.py may read it

    def _on_training_start(self):
        self.steps0 = self.num_timesteps
        self.next_eval = (self.num_timesteps // self.args.eval_every + 1) * self.args.eval_every

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.ep.append(info)
        return not self.stop

    def _on_rollout_end(self):
        if self.ep:
            ends = [e["end"] for e in self.ep]
            laps = [e["lap_time"] for e in self.ep if e["end"] == "lap"]
            self.logger.record("laps/finished_share", ends.count("lap") / len(ends))
            self.logger.record("laps/offtrack_share", ends.count("offtrack") / len(ends))
            if laps:
                self.logger.record("laps/lap_time", float(np.mean(laps)))
            self.ep = []
        self.logger.record("time/steps_per_s", (self.num_timesteps - self.steps0) / (time.time() - self.t0))
        h = harsh_share(self.num_timesteps, self.args, self.mix)
        self.training_env.env_method("set_mix", h, self.mix["stress"])
        self.logger.record("curriculum/harsh_share", h)
        if psutil.virtual_memory().available / 1e9 < self.args.min_free_gb:
            print(f"free RAM under {self.args.min_free_gb} GB: saving and stopping", flush=True)
            self.stop = True
            return
        if self.args.pause_at and self.num_timesteps >= self.args.pause_at:
            print(f"[{self.num_timesteps}] pause point reached: saving and stopping", flush=True)
            self.stop = self.paused = True
            return
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.args.eval_every
            self.checkpoint()

    def checkpoint(self):
        """Eval here (--eval-envs > 0), or just save ckpt/<steps> for eval_residual.py."""
        if self.eval_env is not None:
            self.run_eval()
            return
        if (self.run / "ckpt" / str(self.num_timesteps) / "saved").exists():
            return                                   # the run ended right on a checkpoint
        self._save("last")
        self._save(f"ckpt/{self.num_timesteps}")
        print(f"[{self.num_timesteps} steps, {(time.time() - self.t0) / 60:.1f} min] saved ckpt/{self.num_timesteps}",
              flush=True)

    def run_eval(self):
        norm = self.training_env

        def pol(obs):
            return self.model.predict(norm.normalize_obs(obs), deterministic=True)[0]
        levels = eval_levels(self.num_timesteps, self.args)
        s = summary(evaluate(self.eval_env, pol, levels))
        # best/ is only compared against evals on the same levels (realistic-only evals come first)
        best = max((score(h) for h in self.history if set(EVAL_SEEDS) & set(h) == set(levels)),
                   default=-np.inf)
        self.history.append({"steps": self.num_timesteps, "minutes": (time.time() - self.t0) / 60, **s})
        for lv, v in s.items():
            self.logger.record(f"eval/{lv}_finished", v["finished"])
            if v["lap_mean"]:
                self.logger.record(f"eval/{lv}_lap", v["lap_mean"])
        tag = ""
        if score(s) > best:
            self._save("best")
            tag = "  <- best"
        self._save("last")
        self._save(f"ckpt/{self.num_timesteps}")
        print(f"[{self.num_timesteps} steps, {(time.time() - self.t0) / 60:.1f} min] {fmt(s)}{tag}", flush=True)
        (self.run / "eval.json").write_text(json.dumps(self.history, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=6, help="parallel training sims (one CPU thread each)")
    ap.add_argument("--eval-envs", type=int, default=0,
                    help="sims for eval inside this process; 0 = only save ckpt/ for eval_residual.py")
    ap.add_argument("--steps", type=int, default=20_000_000, help="total; the learning rate reaches 0 here")
    ap.add_argument("--eval-every", type=int, default=100_000, help="save ckpt/<steps> this often")
    ap.add_argument("--eval-after", action="store_true", help="run eval_residual.py on every checkpoint at the end")
    ap.add_argument("--curriculum-at", type=int, default=1_000_000, help="steps on realistic sensors only")
    ap.add_argument("--ramp", type=int, default=1_000_000, help="steps to ramp the harsh share up")
    ap.add_argument("--pause-at", type=int, default=None, help="save and stop at this many steps")
    ap.add_argument("--min-free-gb", type=float, default=2.5)
    ap.add_argument("--name", default="residual")
    ap.add_argument("--resume", default=None, help="checkpoint folder (model.zip + vecnormalize.pkl)")
    args = ap.parse_args()

    torch.set_num_threads(1)
    cfg = load_config()
    rc = cfg["residual"]
    p = rc["ppo"]
    lr = linear_lr(p["learning_rate"])
    if args.resume:                      # continue in the same run folder
        ckpt = pathlib.Path(args.resume)
        run = ckpt.parent
        ev = run / "eval.json"
        history = json.loads(ev.read_text()) if ev.exists() else []
        seed_base = int(time.time()) % 1_000_000     # new laps, not a replay of the first ones
    else:
        run = ROOT / "runs" / "gps" / "residual" / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.name}"
        run.mkdir(parents=True, exist_ok=True)
        history, seed_base = [], 0
    (run / "finished").unlink(missing_ok=True)       # tells eval_residual.py training is running
    stamp = time.strftime('%Y%m%d-%H%M%S')
    (run / f"settings-{stamp}.json").write_text(
        json.dumps({"args": vars(args), "seed_base": seed_base, "residual": rc}, indent=1))

    venv = VecMonitor(SubprocVecEnv([make_env(i, seed_base=seed_base) for i in range(args.envs)],
                                    start_method="spawn"))
    if args.resume:
        venv = VecNormalize.load(str(ckpt / "vecnormalize.pkl"), venv)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=p["gamma"])
    eval_env = None
    if args.eval_envs:
        eval_env = SubprocVecEnv([make_env(100 + i, level="realistic", stress=False)
                                  for i in range(args.eval_envs)], start_method="spawn")

    if args.resume:
        model = PPO.load(ckpt / "model.zip", env=venv, device="cpu", custom_objects={"learning_rate": lr})
    else:
        model = PPO("MlpPolicy", venv, device="cpu", verbose=1, seed=0,
                    learning_rate=lr, n_steps=p["n_steps"], batch_size=p["batch_size"],
                    n_epochs=p["n_epochs"], gamma=p["gamma"], gae_lambda=p["gae_lambda"],
                    clip_range=p["clip_range"], ent_coef=p["ent_coef"], vf_coef=p["vf_coef"],
                    max_grad_norm=p["max_grad_norm"],
                    policy_kwargs={"net_arch": {"pi": p["net"], "vf": p["net"]},
                                   "log_std_init": p["log_std_init"], "activation_fn": torch.nn.Tanh})
    venv.env_method("set_mix", harsh_share(model.num_timesteps, args, rc["mix"]), rc["mix"]["stress"])
    model.set_logger(configure(str(run / "tb"), ["stdout", "csv", "tensorboard"]))
    cb = Trainer(run, eval_env, args, rc["mix"], history)

    if eval_env is not None and not args.resume:
        base = summary(evaluate(eval_env, lambda o: np.zeros((len(o), 2), np.float32)))
        (run / "baseline_B.json").write_text(json.dumps(base, indent=1))
        print(f"driver B on the eval laps: {fmt(base)}", flush=True)

    left = args.steps - model.num_timesteps
    print(f"training {left} steps (to {args.steps}) on {args.envs} sims -> {run}", flush=True)
    try:
        model.learn(left, callback=cb, reset_num_timesteps=False)
    except KeyboardInterrupt:
        print("stopped by Ctrl+C", flush=True)
    cb.checkpoint()
    venv.close()
    if eval_env is not None:
        eval_env.close()
    (run / "finished").write_text(str(model.num_timesteps))
    if cb.paused:
        print(f"paused -> resume with: python scripts/gps/train_residual.py --resume {run / 'last'}", flush=True)
    print(f"done -> {run}", flush=True)
    if args.eval_after:
        import subprocess
        subprocess.run([sys.executable, "-u", str(pathlib.Path(__file__).with_name("eval_residual.py")), str(run)])


if __name__ == "__main__":
    main()
