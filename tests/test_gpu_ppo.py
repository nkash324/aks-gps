"""GPU PPO (kart_gpu/ppo.py) and GPU residual env (kart_gpu/residual.py) against their
references: Stable-Baselines3 (normaliser, GAE, one PPO update, checkpoint loading) and
the CPU ResidualEnv (observations and rewards on the same lap). Needs CUDA."""
import pickle

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

PPO_CFG = {"gamma": 0.995, "gae_lambda": 0.95, "learning_rate": 3e-4, "n_epochs": 1, "minibatches": 1,
           "clip_range": 0.2, "ent_coef": 0.001, "vf_coef": 0.5, "max_grad_norm": 0.5, "log_std_init": -1.0,
           "net": [32, 32]}


def test_running_mean_std():
    from stable_baselines3.common.running_mean_std import RunningMeanStd as Ref
    from kart_gpu.ppo import RunningMeanStd
    rng = np.random.default_rng(0)
    a, b = Ref(shape=(5,)), RunningMeanStd((5,))
    for _ in range(20):
        x = rng.normal(3.0, 2.0, (64, 5))
        a.update(x)
        b.update(torch.as_tensor(x, device="cuda"))
    assert np.allclose(a.mean, b.mean.cpu().numpy()) and np.allclose(a.var, b.var.cpu().numpy())
    assert abs(a.count - b.count) < 1e-9


def test_normaliser_matches_vecnormalize():
    from stable_baselines3.common.vec_env import VecEnv, VecNormalize
    from kart_gpu.ppo import GpuPPO, _Spaces
    N, T = 8, 30
    rng = np.random.default_rng(1)
    obs = rng.normal(0, 3, (T + 1, N, 4)).astype(np.float32)
    rew = rng.normal(0, 1, (T, N)).astype(np.float32)
    done = rng.random((T, N)) < 0.1
    sp = _Spaces(4, 2)

    class Scripted(VecEnv):
        def __init__(self):
            super().__init__(N, sp.observation_space, sp.action_space)
            self.t = 0
        def reset(self):
            return obs[0]
        def step_async(self, actions):
            pass
        def step_wait(self):
            self.t += 1
            return obs[self.t], rew[self.t - 1], done[self.t - 1], [{} for _ in range(N)]
        def close(self): pass
        def get_attr(self, *a, **k): return [None] * N
        def set_attr(self, *a, **k): pass
        def env_method(self, *a, **k): return [None] * N
        def env_is_wrapped(self, *a, **k): return [False] * N

    vn = VecNormalize(Scripted(), norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.995)
    algo = GpuPPO(4, 2, N, PPO_CFG, 1000)
    ref = [vn.reset()]
    mine = [algo.normalise_reset(torch.as_tensor(obs[0], device="cuda")).cpu().numpy()]
    for t in range(T):
        o, r, _, _ = vn.step(np.zeros((N, 2)))
        o2, r2 = algo.normalise_step(torch.as_tensor(obs[t + 1], device="cuda"),
                                     torch.as_tensor(rew[t], device="cuda"), torch.as_tensor(done[t], device="cuda"))
        assert np.allclose(o, o2.cpu().numpy(), atol=1e-5), t
        assert np.allclose(r, r2.cpu().numpy(), atol=1e-5), t


def test_gae_matches_rollout_buffer():
    from stable_baselines3.common.buffers import RolloutBuffer
    from kart_gpu.ppo import GpuPPO, _Spaces
    N, T = 6, 40
    rng = np.random.default_rng(2)
    sp = _Spaces(3, 2)
    buf = RolloutBuffer(T, sp.observation_space, sp.action_space, device="cpu", gae_lambda=0.95, gamma=0.995, n_envs=N)
    B = {k: [] for k in ("rew", "start", "val")}
    for t in range(T):
        r, s, v = rng.normal(size=N), rng.random(N) < 0.1, rng.normal(size=N)
        if t == 0:
            s[:] = True
        buf.add(np.zeros((N, 3)), np.zeros((N, 2)), r, s, torch.as_tensor(v), torch.zeros(N))
        B["rew"].append(r), B["start"].append(s), B["val"].append(v)
    last_v, last_d = rng.normal(size=N), rng.random(N) < 0.1
    buf.compute_returns_and_advantage(torch.as_tensor(last_v), last_d)
    algo = GpuPPO(3, 2, N, PPO_CFG, 1000)
    G = {k: torch.as_tensor(np.array(v), device="cuda", dtype=torch.float32 if k != "start" else torch.bool)
         for k, v in B.items()}
    G["last_val"] = torch.as_tensor(last_v, device="cuda", dtype=torch.float32)
    G["last_done"] = torch.as_tensor(last_d, device="cuda")
    algo.gae(G)
    assert np.allclose(buf.advantages, G["adv"].cpu().numpy(), atol=1e-4)
    assert np.allclose(buf.returns, G["ret"].cpu().numpy(), atol=1e-4)


def test_one_update_matches_sb3():
    """Same weights + same rollout -> one SB3 PPO.train() and one GpuPPO.train() give the same weights."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from kart_gpu.ppo import GpuPPO, _Spaces
    N, T = 8, 64
    p = PPO_CFG
    env = DummyVecEnv([lambda: _Spaces(5, 2)] * N)
    ref = PPO("MlpPolicy", env, device="cuda", seed=0, learning_rate=p["learning_rate"], n_steps=T,
              batch_size=T * N, n_epochs=1, gamma=p["gamma"], gae_lambda=p["gae_lambda"], clip_range=p["clip_range"],
              ent_coef=p["ent_coef"], vf_coef=p["vf_coef"], max_grad_norm=p["max_grad_norm"],
              policy_kwargs={"net_arch": {"pi": p["net"], "vf": p["net"]}, "log_std_init": p["log_std_init"],
                             "activation_fn": torch.nn.Tanh})
    algo = GpuPPO(5, 2, N, p, 1000)
    algo.policy.load_state_dict(ref.policy.state_dict())
    algo.opt.load_state_dict(ref.policy.optimizer.state_dict())
    torch.manual_seed(0)
    rng = np.random.default_rng(3)
    B = {k: [] for k in ("obs", "act", "rew", "start", "val", "logp")}
    buf = ref.rollout_buffer
    buf.reset()
    with torch.no_grad():
        for t in range(T):
            o = torch.as_tensor(rng.normal(size=(N, 5)), dtype=torch.float32, device="cuda")
            a, v, lp = ref.policy(o)
            r = rng.normal(size=N)
            s = (rng.random(N) < 0.05) | (t == 0)
            buf.add(o.cpu().numpy(), a.cpu().numpy(), r, s, v, lp)
            for k, x in (("obs", o), ("act", a), ("rew", torch.as_tensor(r, dtype=torch.float32, device="cuda")),
                         ("start", torch.as_tensor(s, device="cuda")), ("val", v.flatten()), ("logp", lp)):
                B[k].append(x)
        last_o = torch.as_tensor(rng.normal(size=(N, 5)), dtype=torch.float32, device="cuda")
        last_v = ref.policy.predict_values(last_o)
    last_d = rng.random(N) < 0.05
    buf.compute_returns_and_advantage(last_v, last_d)
    G = {k: torch.stack(v) for k, v in B.items()}
    G["last_val"], G["last_done"] = last_v.flatten(), torch.as_tensor(last_d, device="cuda")
    algo.gae(G)
    from stable_baselines3.common.logger import configure
    ref.set_logger(configure(None, []))
    ref.train()
    algo.train(G, lr=p["learning_rate"])
    for (n1, a), (n2, b) in zip(ref.policy.named_parameters(), algo.policy.named_parameters()):
        assert torch.allclose(a, b, atol=2e-6), n1
    # and the update did change something
    assert any(not torch.equal(a, b) for a, b in zip(ref.policy.parameters(),
               GpuPPO(5, 2, N, p, 1000).policy.parameters()))


def test_checkpoint_loads_on_cpu(tmp_path):
    """A GpuPPO checkpoint drives the same actions through benchmark_residual's loading code."""
    from stable_baselines3 import PPO
    from kart_gpu.ppo import GpuPPO
    algo = GpuPPO(45, 2, 16, dict(PPO_CFG, net=[128, 128]), 1000)
    raw = torch.randn(16, 45, device="cuda") * 4 + 1
    algo.normalise_reset(raw)
    algo.normalise_step(raw * 0.5, torch.randn(16, device="cuda"), torch.zeros(16, dtype=torch.bool, device="cuda"))
    algo.save(tmp_path)
    model = PPO.load(tmp_path / "model.zip", device="cpu",
                     custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0})
    with open(tmp_path / "vecnormalize.pkl", "rb") as f:
        norm = pickle.load(f)
    obs = raw.cpu().numpy()
    ref = model.predict(norm.normalize_obs(obs), deterministic=True)[0]
    with torch.no_grad():
        mine = algo.policy._predict(algo.norm_obs(raw), deterministic=True).clamp(-1, 1).cpu().numpy()
    assert np.allclose(ref, mine, atol=1e-5)
    assert (tmp_path / "saved").exists()


def test_residual_env_matches_cpu():
    """Perfect sensors, nominal kart: the same lap on both -> same observations and rewards."""
    from kart_gps.config import load_config
    from kart_gps.rl_env import ResidualEnv
    from kart_gpu.residual import BatchResidual
    cfg = load_config(overrides={"randomize": {"enabled": False}})
    cpu = ResidualEnv(cfg, level="perfect", stress=False)
    o_c, _ = cpu.reset(seed=0, options={"lap_seed": 0})
    gpu = BatchResidual(cfg, 2, dtype=torch.float64, graph=False, track=cpu.track, line=cpu.line)
    o_g = gpu.reset(level="perfect")
    rng = np.random.default_rng(4)
    worst_o = worst_r = 0.0
    for t in range(300):
        assert o_g.shape[1] == len(o_c)
        worst_o = max(worst_o, float(np.abs(o_g[0].cpu().numpy() - o_c).max()))
        a = np.clip(rng.normal(0, 0.3, 2), -1, 1)
        o_c, r_c, d_c, _, _ = cpu.step(a)
        o_g, r_g, d_g, _ = gpu.step(torch.as_tensor(np.tile(a, (2, 1)), device="cuda"), autoreset=False)
        worst_r = max(worst_r, abs(float(r_g[0]) - r_c))
        assert bool(d_g[0]) == d_c
        if d_c:
            break
    assert worst_o < 1e-3, worst_o
    assert worst_r < 1e-3, worst_r
