# AKS GPS driver: a learned racing driver for the AKS autonomous kart

A self-driving go-kart driver for the AKS race on the Purdue Grand Prix track. The kart
localises with RTK GPS, follows a racing line computed from a surveyed map of the track,
and a small neural network trained with reinforcement learning (PPO) nudges its speed and
steering to go faster. Everything here runs in simulation; no lap has been driven on the
real kart yet.

## Overview pages

Written summaries and interactive results (private claude.ai pages; share them from the
page if others need access):

| Page | What it is |
|---|---|
| [AKS Kart Driver Briefing](https://claude.ai/artifact/43FYGY8i3wHb8Z3XZ95Xxo) | Short team briefing: the result, lap replay at three sensor levels, how the network and PPO work, open questions |
| [AKS Kart Driver Overview](https://claude.ai/artifact/4ubgBm4Nxo1gXu8SWnNHVt) | Full technical write-up: track, kart model, sensors, EKF, planner, training, physics checks |
| [Purdue Kart GPU Run 2 at 500M](https://claude.ai/artifact/CJkfqUACmtU8NtWsaADRRQ) | Benchmark of the final training run (G2 460.1M) against drivers A and B |
| [Purdue Kart GPU Run 2](https://claude.ai/artifact/U13Py7R1zoukYqdwUP8KkE) | Run 2 to 200M steps and the GPU fine-tune |
| [Purdue Kart GPU Benchmark](https://claude.ai/artifact/PUCeUQ84i7JCiuGroVn3rD) | Every CPU- and GPU-trained checkpoint on the same 1,000 laps per sensor level |
| [Purdue GPS Kart Benchmark](https://claude.ai/artifact/4EbxeYrVweeKQWkrGbkmny) | First CPU benchmark of drivers A and B |

## Results

Mean lap on the same 1,000 simulated laps per driver, realistic sensor errors:

| Driver | Mean lap | Laps finished |
|---|---|---|
| A: centreline pure pursuit, cautious | 51.23 s | 99.8% |
| B: racing-line pure pursuit, best tuned settings | 46.47 s | 99.3% |
| **G2 460.1M: B + learned correction** | **43.46 s** | **99.3%** |

G2 is about 3 s a lap faster than the best tuned B, on every lap both finished. The best
checkpoint is `runs/gpu/20261029-234306-run2/ckpt/460062720/` (`model.zip` +
`vecnormalize.pkl`, Stable-Baselines3 format).

## What the code does

**1. The track: `tracks/purdue_rtk/`, `kart_gps/track.py`**
The track edges, built from the 2021 RTK survey, as a MuJoCo world. `track.py` reads them
and answers: how far around the lap is this point, how far off the centreline, how much
room to each edge, has the kart left the track? Plain numpy, so it can run on the kart.

**2. The kart: `kart_gps/kart.py`, `mj_kart.py`, `aks_kart.xml`**
- `kart.py`: a fast 2D kart model with mass, tyre grip (sliding gradually at the limit),
  weight shift under braking and cornering, steering speed and delay, motor and regen
  limits, and the hydraulic brake. Used for all training and benchmarks.
- `mj_kart.py`: the same kart as a 3D MuJoCo body, for cross-checks and watching laps.
- Every lap randomises the kart (grip, mass, motor, brakes, steering), because the real
  kart's numbers are still estimates.

**3. Sensors and "where am I": `kart_gps/sensors.py`, `estimator.py`**
- `sensors.py` simulates RTK GPS (20 Hz, 50 ms late, centimetre noise, occasional drops to
  "float" and short dropouts), the IMU (noise, bias) and the wheel encoder (noise, tyre
  size error), at three levels: perfect, realistic, harsh.
- `estimator.py` is the EKF. It blends those sensors into the kart's best estimate of
  position, heading and speed (the **pose**) that every driver uses.

**4. The plan: `kart_gps/planner.py`**
Before driving, it computes the racing line (the smoothest path, at least 0.30 m inside
the edges) and a speed plan (how fast at each point, from cornering and braking limits).

**5. The drivers: `kart_gps/drivers.py`, `control.py`, `rl_env.py`**
- **Driver A:** pure pursuit along the centreline, slow and safe.
- **Driver B:** pure pursuit along the racing line.
- **G2 (learned):** driver B plus a small neural network's correction to speed and turning
  (45 inputs → 128 → 128 → 2 outputs, capped at ±1 m/s and ±0.02 1/m). `rl_env.py`
  builds its 45 inputs.
- `control.py`: the low-level controller that turns "go this fast, turn this much" into
  motor, brake and steering commands.

**6. Training: `kart_gpu/`, `scripts/gpu/train.py`, `scripts/gps/train_residual.py`**
`kart_gpu/` is the whole simulator rewritten to run 4,096 karts at once on a GPU, plus a
PPO trainer that writes standard Stable-Baselines3 checkpoints. The best result, G2 460.1M,
is in `runs/gpu/20261029-234306-run2/ckpt/460062720/`.

**7. Testing and viewing: `scripts/gps/`, `scripts/gpu/`, `tests/`**
- Benchmarks where every driver drives the same laps (same kart, same sensor errors).
- Checkpoint screening, and `tune_b.py`, which finds driver B's best settings for a fair
  comparison.
- Kart spec checks (acceleration, braking, steering, skidpad) on both physics models.
- `play_mujoco.py`: watch G2 in the MuJoCo viewer, with A and B as see-through karts.
- HTML results pages, and 22 automated tests.

### One lap, end to end

```
map -> racing line + speed plan            (once, before driving)
every 10 ms:  sensors (noisy) -> EKF -> pose
every 50 ms:  driver (A, B or G2) -> target speed + turn
every 10 ms:  low-level control -> motor / brake / steering -> kart physics -> next step
```

### Not in this repository yet

- No ROS: a ROS 2 / Docker version of this loop exists separately.
- No camera input.
- No connection to real hardware.
- The kart numbers are estimates until the built kart is measured.

## File layout

```
gps.yaml               sim, sensors, planner, drivers, reward, network
gpu.yaml, gpu_run2.yaml, gpu_run2_ext.yaml, gpu_ft.yaml   GPU training settings
tracks/purdue_rtk/     track edges from the 2021 RTK survey (MuJoCo world)
kart_gps/              CPU reference stack
  track.py               map: centreline, edges, progress, offset (numpy only)
  kart.py                2D kart physics (bicycle model, combined-slip tyres, actuators)
  mj_kart.py, aks_kart.xml   the same kart in MuJoCo (3D cross-check and viewer)
  sensors.py             RTK GNSS, IMU, wheel encoder with noise, delay, dropouts
  estimator.py           EKF: position, heading, speed, IMU biases
  planner.py             minimum-curvature racing line and speed plan
  control.py             low-level speed / steering control
  drivers.py             drivers A and B (pure pursuit)
  env.py, rl_env.py      lap environment and the residual-RL environment
kart_gpu/              the same stack as batched PyTorch on the GPU
  sim.py, residual.py    thousands of karts at once
  ppo.py                 PPO with an adaptive learning rate (SB3-compatible checkpoints)
scripts/gps/           CPU: benchmark, training, evaluation, kart checks, MuJoCo viewer
scripts/gpu/           GPU: training, checkpoint screening and benchmark, sim check, tuning of B
scripts/build_purdue_track.py   rebuild tracks/purdue_rtk from the survey files
runs/gpu/20261029-234306-run2/ckpt/460062720/   best model, G2 460.1M (SB3 model.zip + vecnormalize.pkl)
tests/                 pytest: track, kart, planner, envs, GPU PPO vs Stable-Baselines3
notes/open_questions.md  questions for the organisers and the team
```

## Setup

Windows (CUDA GPU): `powershell -ExecutionPolicy Bypass -File setup.ps1`
Mac / Linux: `./setup.sh`

Then check it: `python -m pytest tests` (from the venv). Machine-specific settings go in
`gps.local.yaml` (same keys as `gps.yaml`, git-ignored).

## Run it

```bash
python scripts/gps/benchmark.py                 # drivers A and B, 3 sensor levels, CPU
python scripts/gps/play_mujoco.py               # watch G2 in the MuJoCo viewer
python scripts/gps/play_mujoco.py --ghosts      # ... with A and B as see-through karts
python scripts/gps/kart_checks.py --mujoco      # kart spec checks on both physics models
python scripts/gpu/train.py --config gpu_run2.yaml --name run2     # GPU training
python scripts/gpu/bench_all.py --run "G:GPU run 2:runs/gpu/<run>"  # screen + benchmark checkpoints
python scripts/gpu/results_page.py              # benchmark -> HTML results page
python scripts/gpu/tune_b.py                    # give driver B its best settings, then compare
```

## Notes

- The kart numbers in `gps.yaml` are bill-of-materials estimates until the built kart is
  measured; every lap randomises them so the driver copes with a range.
- The track files come from the 2021 RTK survey. Keep this repository private unless the
  survey's owners agree to sharing it.
- The overview pages are private claude.ai pages; share each one from its page for others
  to open it.
- This repository holds only the GPS/map work. The earlier camera-based driver lives in
  the original AKS-RL repository.
