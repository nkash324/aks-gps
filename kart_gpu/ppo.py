"""PPO for the GPU env: the same algorithm as Stable-Baselines3's PPO (and its VecNormalize),
written for rollouts that live on the GPU. The network is SB3's own ActorCriticPolicy, so
every checkpoint is a normal SB3 model.zip + vecnormalize.pkl that the CPU scripts load
(benchmark_residual.py, eval_residual.py). scripts/gpu/test_ppo.py checks it against SB3.

    algo = GpuPPO(n_obs=45, n_act=2, n_envs=4096, ppo=cfg, total_steps=100_000_000)
    obs = algo.normalise_reset(env.reset())
    while ...:
        batch, obs = algo.collect(env, obs, n_steps)
        stats = algo.train(batch)
    algo.save(folder)
"""
import pathlib
import warnings

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# SB3 warns that PPO on the GPU is slow; that is about its own CPU-side rollouts, not ours
warnings.filterwarnings("ignore", message="You are trying to run PPO on the GPU")

EPS = 1e-8          # VecNormalize epsilon
CLIP_OBS = 10.0
CLIP_REW = 10.0


class RunningMeanStd:
    """SB3 RunningMeanStd on the GPU (float64)."""
    def __init__(self, shape=(), device="cuda", epsilon=1e-4):
        self.mean = torch.zeros(shape, dtype=torch.float64, device=device)
        self.var = torch.ones(shape, dtype=torch.float64, device=device)
        self.count = epsilon

    def update(self, x):
        x = x.double()
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        m2 = self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot
        self.var = m2 / tot
        self.count = tot


class _Spaces(gym.Env):
    """Only carries the spaces, so SB3 can build the policy."""
    def __init__(self, n_obs, n_act):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (n_act,), np.float32)

    def reset(self, seed=None, options=None):
        return np.zeros(self.observation_space.shape, np.float32), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, np.float32), 0.0, False, False, {}


class GpuPPO:
    def __init__(self, n_obs, n_act, n_envs, ppo, total_steps, device="cuda", seed=0, resume=None, init=None):
        """ppo: dict with gamma, gae_lambda, learning_rate, n_epochs, minibatches, clip_range,
        ent_coef, vf_coef, max_grad_norm, log_std_init, net, and optionally lr_schedule:
          'linear'   (default) learning_rate decays linearly to 0 at total_steps
          'adaptive' starts at learning_rate, then after every gradient step x1.5 / /1.5 to keep
                     the policy change (KL) near desired_kl, within [lr_min, lr_max]
        resume: checkpoint folder to continue a run from (keeps its step count).
        init: checkpoint folder to start a NEW run from (its weights and normaliser, step 0)."""
        self.p, self.N, self.dev = ppo, n_envs, torch.device(device)
        self.total = total_steps
        lr0 = ppo["learning_rate"]
        dummy = DummyVecEnv([lambda: _Spaces(n_obs, n_act)])
        resume = resume or init
        if resume:
            resume = pathlib.Path(resume)
            self.model = PPO.load(resume / "model.zip", env=dummy, device=self.dev,
                                  custom_objects={"learning_rate": lr0, "lr_schedule": lambda _: lr0})
        else:
            self.model = PPO("MlpPolicy", dummy, device=self.dev, seed=seed, learning_rate=lr0,
                             gamma=ppo["gamma"], gae_lambda=ppo["gae_lambda"], n_epochs=ppo["n_epochs"],
                             clip_range=ppo["clip_range"], ent_coef=ppo["ent_coef"], vf_coef=ppo["vf_coef"],
                             max_grad_norm=ppo["max_grad_norm"],
                             policy_kwargs={"net_arch": {"pi": ppo["net"], "vf": ppo["net"]},
                                            "log_std_init": ppo["log_std_init"], "activation_fn": torch.nn.Tanh})
        self.policy = self.model.policy
        self.opt = self.policy.optimizer
        self.obs_rms = RunningMeanStd((n_obs,), self.dev)
        self.ret_rms = RunningMeanStd((), self.dev)
        if resume:
            import pickle
            with open(resume / "vecnormalize.pkl", "rb") as f:
                vn = pickle.load(f)
            for mine, theirs in ((self.obs_rms, vn.obs_rms), (self.ret_rms, vn.ret_rms)):
                mine.mean = torch.as_tensor(theirs.mean, dtype=torch.float64, device=self.dev)
                mine.var = torch.as_tensor(theirs.var, dtype=torch.float64, device=self.dev)
                mine.count = float(theirs.count)
        self.returns = torch.zeros(n_envs, dtype=torch.float64, device=self.dev)
        if init:
            self.model.num_timesteps = 0
        self.steps = int(self.model.num_timesteps)
        self.cur_lr = lr0

    # ------------------------------------------------------------------ normalisation (VecNormalize)
    def norm_obs(self, obs):
        o = (obs.double() - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + EPS)
        return o.clamp(-CLIP_OBS, CLIP_OBS).float()

    def normalise_reset(self, obs):
        self.returns.zero_()
        self._start = torch.ones(self.N, dtype=torch.bool, device=self.dev)    # SB3 _last_episode_starts
        self.obs_rms.update(obs)
        return self.norm_obs(obs)

    def normalise_step(self, obs, reward, done):
        self.obs_rms.update(obs)
        o = self.norm_obs(obs)
        self.returns = self.returns * self.p["gamma"] + reward.double()
        self.ret_rms.update(self.returns)
        r = (reward.double() / torch.sqrt(self.ret_rms.var + EPS)).clamp(-CLIP_REW, CLIP_REW).float()
        self.returns = torch.where(done, torch.zeros_like(self.returns), self.returns)
        return o, r

    # ------------------------------------------------------------------ rollout
    @torch.no_grad()
    def collect(self, env, obs, n_steps, on_step=None):
        """n_steps on every env. obs = the current normalised observation.
        -> (batch dict of (T, N, ...) tensors, next obs)."""
        T, N = n_steps, self.N
        B = {"obs": [], "act": [], "rew": [], "start": [], "val": [], "logp": []}
        start = self._start
        for _ in range(T):
            act, val, logp = self.policy(obs)
            raw, rew, done, info = env.step(act.clamp(-1.0, 1.0))
            nobs, nrew = self.normalise_step(raw, rew, done)
            for k, v in (("obs", obs), ("act", act), ("rew", nrew), ("start", start),
                         ("val", val.flatten()), ("logp", logp)):
                B[k].append(v)
            if on_step is not None:
                on_step(rew, done, info)
            obs, start = nobs, done
            self.steps += N
        B = {k: torch.stack(v) for k, v in B.items()}
        B["last_val"] = self.policy.predict_values(obs).flatten()
        B["last_done"] = start
        self._start = start
        self.gae(B)
        return B, obs

    def gae(self, B):
        """SB3 RolloutBuffer.compute_returns_and_advantage."""
        g, lam = self.p["gamma"], self.p["gae_lambda"]
        T = B["rew"].shape[0]
        adv = torch.zeros_like(B["rew"])
        last = torch.zeros_like(B["rew"][0])
        for t in reversed(range(T)):
            if t == T - 1:
                nnt = 1.0 - B["last_done"].float()
                nv = B["last_val"]
            else:
                nnt = 1.0 - B["start"][t + 1].float()
                nv = B["val"][t + 1]
            delta = B["rew"][t] + g * nv * nnt - B["val"][t]
            last = delta + g * lam * nnt * last
            adv[t] = last
        B["adv"] = adv
        B["ret"] = adv + B["val"]

    # ------------------------------------------------------------------ update
    def lr(self):
        if self.p.get("lr_schedule", "linear") == "adaptive":
            return self.cur_lr
        return self.p["learning_rate"] * max(0.0, 1.0 - self.steps / self.total)

    def train(self, B, lr=None):
        """SB3 PPO.train(): n_epochs over shuffled minibatches, clipped surrogate,
        value MSE, entropy bonus, gradient clipping. -> stats dict."""
        p = self.p
        lr = self.lr() if lr is None else lr
        for g in self.opt.param_groups:
            g["lr"] = lr
        flat = {k: B[k].reshape(-1, *B[k].shape[2:]) for k in ("obs", "act", "logp", "adv", "ret", "val")}
        n = flat["obs"].shape[0]
        mb = n // p["minibatches"]
        clip = p["clip_range"]
        st = {"pg": [], "vf": [], "ent": [], "kl": [], "clipfrac": []}
        for _ in range(p["n_epochs"]):
            perm = torch.randperm(n, device=self.dev)
            for i in range(0, n - mb + 1, mb):
                idx = perm[i:i + mb]
                obs, act, old, adv, ret = (flat[k][idx] for k in ("obs", "act", "logp", "adv", "ret"))
                val, logp, ent = self.policy.evaluate_actions(obs, act)
                val = val.flatten()
                if len(adv) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                ratio = torch.exp(logp - old)
                pg = -torch.min(adv * ratio, adv * ratio.clamp(1 - clip, 1 + clip)).mean()
                vf = torch.nn.functional.mse_loss(ret, val)
                ent_loss = -ent.mean() if ent is not None else -(-logp).mean()
                loss = pg + p["ent_coef"] * ent_loss + p["vf_coef"] * vf
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), p["max_grad_norm"])
                self.opt.step()
                with torch.no_grad():
                    lr_ = logp - old
                    kl = ((torch.exp(lr_) - 1) - lr_).mean()
                    st["kl"].append(kl)
                if p.get("lr_schedule", "linear") == "adaptive":       # rsl_rl's rule
                    k = float(kl)
                    if k > 2.0 * p["desired_kl"]:
                        self.cur_lr = max(p["lr_min"], self.cur_lr / 1.5)
                    elif k < 0.5 * p["desired_kl"]:
                        self.cur_lr = min(p["lr_max"], self.cur_lr * 1.5)
                    for g in self.opt.param_groups:
                        g["lr"] = self.cur_lr
                st["clipfrac"].append(((ratio - 1).abs() > clip).float().mean())
                st["pg"].append(pg.detach())
                st["vf"].append(vf.detach())
                st["ent"].append(ent_loss.detach())
        out = {k: float(torch.stack(v).mean()) for k, v in st.items()}
        v, r = flat["val"], flat["ret"]
        out["explained_var"] = float(1 - (r - v).var() / r.var().clamp(min=1e-12))
        out["std"] = float(torch.exp(self.policy.log_std.detach()).mean())
        out["lr"] = self.cur_lr if p.get("lr_schedule", "linear") == "adaptive" else lr
        return out

    # ------------------------------------------------------------------ checkpoints
    def save(self, folder):
        """model.zip + vecnormalize.pkl, loadable by the CPU scripts, then the 'saved' marker."""
        d = pathlib.Path(folder)
        d.mkdir(parents=True, exist_ok=True)
        self.model.num_timesteps = self.steps
        self.model.save(d / "model.zip")
        vn = VecNormalize(DummyVecEnv([lambda: _Spaces(len(self.obs_rms.mean), 2)]), norm_obs=True,
                          norm_reward=True, clip_obs=CLIP_OBS, clip_reward=CLIP_REW, gamma=self.p["gamma"],
                          epsilon=EPS)
        for mine, theirs in ((self.obs_rms, vn.obs_rms), (self.ret_rms, vn.ret_rms)):
            theirs.mean = mine.mean.cpu().numpy().astype(np.float64)
            theirs.var = mine.var.cpu().numpy().astype(np.float64)
            theirs.count = float(mine.count)
        vn.save(str(d / "vecnormalize.pkl"))
        (d / "saved").write_text(str(self.steps))
