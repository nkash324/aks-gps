# Strong non-ML drivers vs the learned driver G2

Question: does the learned residual driver (G2 460.1M) still win against a well-engineered
non-ML driver? Answer from this benchmark: **no, not on lap time.** A tuned feed-forward +
feedback "formula" driver (C) ties G2 on realistic laps and finishes more harsh laps. Only an
oracle that knows each lap's true grip is clearly faster (about 1 s).

All numbers are from the 2D sim, run on the **CPU** (no GPU in this session). CPU random draws
differ from CUDA, so the published GPU per-lap benchmark was not reused: every driver, G2 and B
included, was re-run on the same new laps (seed 20261029, 1,000 laps per sensor level;
`check_laps.py` confirms that separate runs with the same seed / N / levels drive identical laps,
max lap-time difference 0.000000 s). G2 reproduces its published number (43.49 s vs 43.46 s).

## What was built

| Piece | Where |
|---|---|
| GGV (friction-ellipse) speed plan, minimum-time line (B-spline offsets, pattern search on the planned lap time, same 0.30 m edge buffer) | `kart_gps/planner.py` (`speed_plan_ggv`, `mintime_line`) |
| Driver C, numpy (runs on the kart) | `kart_gps/drivers.py` `Formula` |
| Driver C, batched torch (knobs are per-kart tensors, so many candidates tune in one batch) | `kart_gpu/formula.py` `BatchFormula` |
| MPC driver, batched torch, any device | `kart_gpu/mpc.py` `BatchMPC` |
| Oracle C | `scripts/formula/bench.py --driver oracle`, `tune_oracle.py` |
| Scripts | `scripts/formula/` : `build_line, tune_c, tune_mpc, tune_oracle, bench, compare, check_laps, common` |
| Tests | `tests/test_formula.py` (mintime line inside the bounds, numpy and torch C give the same commands, MPC smoke test) |

**Minimum-time line.** Planned lap (GGV plan, lat 7.35 m/s^2, brake 3.85 m/s^2): min-curvature line
40.946 s, mintime line 39.560 s. Cached in `runs/formula/mintime_line.npz`. It was built once with
the default GGV limits and not rebuilt for the tuned limits.

**Driver C.** Predict the pose `tau` s ahead from v and yaw rate; nearest point on the mintime line:
lateral error e (left +), heading error epsi, curvature kappa_ref. `kappa_cmd = us*kappa_ref - k_e*e - k_psi*epsi`,
`k_e = w^2`, `k_psi = 2*zeta*w`, `w = min(w_space, w_time/max(v,1))`, clamped to the steering limit.
Speed = GGV plan at s + v*preview, times B's float / lost slowdowns.

**MPC.** 15 steps of 0.1 s (2 Euler substeps) in the path frame of the mintime line, state
(s, e, epsi, v, kappa_act), controls: curvature command and target speed per step. Cost: -progress,
e^2, soft barrier on the edge room (kart half width + margin), friction-ellipse excess, speed above
the plan, command smoothness, terminal over-plan-speed penalty. Adam on normalised variables
(one unit = 0.05 1/m or 1 m/s), warm-started from the previous solution shifted one step, first
command applied, safety slowdowns on top. Cost: about 1.25 s per decision for 3,000 karts on 4 CPU
cores (0.4 s for 128 karts: it is op-count bound, so big batches are cheap per kart); the 3,000-lap
benchmark took 21 min.

## Tuned settings (tuning laps: seed 7,000,000, never the benchmark laps)

**C** (random search, 40 + 24 candidates x 120 realistic laps, then the 3 best on 1000 realistic +
500 harsh fresh laps; bar: realistic finish >= 99.3%). Winner (99.3% finish, 43.47 s on the fresh
tuning laps): lat 6.92 m/s^2, brake 3.81 m/s^2, preview 0.561 s, w_space 0.430 /m, w_time 3.58 rad/s,
us 1.018, tau 0.131 s, zeta 1.276. The search converged to us = 1.0, the lower edge of its range
(the 1.29 understeer factor the kart needs at low speed is replaced by a longer tau and a
well-damped feedback), so a wider search might find a little more.

**MPC** (48 candidates x 100 laps, best 2 on 500 realistic + 250 harsh): lat 7.13, brake 4.35,
margin 0.070 m, w_b 194, w_g 4.55, w_e 0.89, w_sk 2.18, us 1.243, lag 0.03 s. Tuned with 12 solver
iterations (97.6% finish); 20 iterations raised it to 98.3% on a 240-lap check, so the benchmark
uses 20. **It never reached the 99.3% bar** (best 97.6-98.3% on tuning laps). Failures cluster
where it weaves at top speed on high-grip laps and then runs wide.

**Oracle C**: C's steering knobs, speed plan chosen per lap from the true grip:
lat = 0.69 x 11.3 x mu, brake = 0.95 x 4.81 x mu (grid on 300 tuning laps, 100% finish).
Sensors are as the level says (the oracle only knows the grip, not the state), so the "perfect"
row is its perfect-sensor result.

## Results (benchmark laps, 1,000 per sensor level, CPU)

Realistic sensors:

| Driver | Finished | Mean lap | Median lap | Off-track | Clearance median / 5th pct | Max lateral g (median) |
|---|---|---|---|---|---|---|
| G2 460.1M (learned) | 99.7% | 43.49 s | 43.33 s | 3 | 0.20 / 0.13 m | 0.79 |
| B (published) | 99.5% | 46.67 s | 46.46 s | 5 | 0.12 / 0.06 m | 0.67 |
| B tuned (6.1 / 5.0 / 0.35) | 99.5% | 46.17 s | 45.96 s | 5 | 0.12 / 0.06 m | 0.69 |
| **C tuned (formula)** | 99.6% | 43.51 s | 43.30 s | 4 | 0.22 / 0.14 m | 0.79 |
| MPC tuned | 97.9% | 43.85 s | 43.65 s | 21 | 0.16 / 0.05 m | 0.75 |
| Oracle C (true grip) | 99.6% | 42.45 s | 42.31 s | 4 | 0.23 / 0.15 m | 0.85 |

Perfect sensors: G2 100.0% 43.09 s, B 100.0% 46.24 s, B tuned 100.0% 45.75 s, C 99.9% 43.11 s, MPC 99.7% 43.38 s, oracle 100.0% 42.10 s.
Harsh sensors: G2 87.5% 43.76 s, B 84.1% 46.95 s, B tuned 84.3% 46.46 s, **C 91.6% 43.82 s**, MPC 86.5% 44.16 s, oracle 91.5% 42.81 s.
Full tables: `runs/formula/summary_1000.json`, per-lap rows: `runs/formula/bench_<driver>_1000.json`.

Versus G2, realistic laps both finished (gap = driver - G2, positive = G2 faster):

| Driver | Laps | Median gap | Mean gap | 95% CI (bootstrap) of mean | G2 faster on |
|---|---|---|---|---|---|
| B | 993 | +3.18 s | +3.19 s | [+3.17, +3.20] | 100% |
| B tuned | 992 | +2.69 s | +2.69 s | [+2.68, +2.71] | 100% |
| **C tuned** | 993 | +0.02 s | +0.03 s | [+0.02, +0.04] | 55.5% |
| MPC tuned | 977 | +0.13 s | +0.39 s | [+0.33, +0.44] | 56.4% |
| Oracle C | 993 | -1.16 s | -1.03 s | [-1.08, -0.99] | 8.6% |

C and G2 differ by 0.03 s on average, with G2 ahead on a bare majority of laps; that is a
statistically detectable but practically zero gap. Both have the same max lateral g (0.79).

## Where the gap to B comes from

G2 beats tuned B by 2.7 s; C closes all of it. Ablations (same 1,000 realistic laps):

| Variant | Mean lap | vs C |
|---|---|---|
| C (mintime line, GGV plan, formula steering) | 43.51 s | - |
| C on the min-curvature line (same plan limits) | 45.04 s | +1.53 s |
| C on the mintime line with B's speed plan (6.1 / 5.0) | 45.07 s | +1.56 s |
| B tuned (min-curvature line, B's plan, pure pursuit) | 46.17 s | +2.66 s |

- **Line**: about 1.5 s (mintime line vs min-curvature line).
- **Speed plan**: about 1.6 s (GGV ellipse plan with 6.9 / 3.8 vs B's circle plan at 6.1 / 5.0; the
  friction ellipse lets it carry speed through corners and brake earlier and harder where needed).
- **Tracking**: the formula steering was roughly neutral against pure pursuit in this decomposition
  (the two ablations do not add up exactly to 2.66 s, so interactions of about 0.4 s are unexplained).
- **Robustness**: C finishes the same share of realistic laps as G2 and 4 points more of harsh laps;
  its tyres run at the same lateral g. G2's learned correction found roughly what the physics-based
  line + plan gives. The learned residual rides on B, so it had to learn the line and plan gains
  that C gets by construction.
- **Grip knowledge**: the oracle (true mu per lap) is 1.0 s faster than C and G2. This is where a
  learned or adaptive driver could still win: C's speed limits carry a margin for the +-15% grip
  randomisation, and G2 evidently does not exploit grip information any better than that.

## Limitations and what was cut

- CPU laps and a CPU re-run of everything; numbers are close to, but not the same laps as, the GPU
  benchmark. One benchmark seed. Harsh finish rates (84-92%) are dominated by sensor events, so
  differences there are less meaningful than they look.
- MPC did **not** meet the 99.3% bar (97.9% on the benchmark). Its tuning was small (48 candidates x
  100 laps; three knobs sat on the edge of their ranges). I ran the benchmark at the full 1,000 laps per level.
  I also tried modelling the command delay (first substep uses the previous command): it was
  worse (95.8%), likely because the other knobs were tuned without it; not pursued.
  The MPC iterations (20) were chosen on a 240-lap check on tuning-seed laps, after tuning with 12.
- C's search hit the lower edge of `us` (1.0); the mintime line was not rebuilt with the tuned GGV
  limits; C was tuned on realistic sensors only (the 99.3% bar applies to realistic laps).
- Oracle C uses the lap's true mu only (steering knobs unchanged, front grip/mass not used).
- The ablations are not an exact decomposition (no pure-pursuit-on-mintime-line run).

## Reproduce

```bash
./setup.sh && .venv/bin/python -m pytest tests -q          # 19 passed, 6 CUDA tests skipped
cd scripts/formula
python check_laps.py --laps 20                             # laps identical across runs
python build_line.py                                       # mintime line -> runs/formula/mintime_line.npz
python tune_c.py --n1 40 --n2 24 --top 5 --screen 120 --shortlist 3 --laps2 1000 --out ../../runs/formula
python tune_mpc.py --n1 24 --n2 24 --top 4 --screen 100 --block 12 --shortlist 2 --laps2 500 --out ../../runs/formula
# (edit iters: 20 into the MPC winner in tune_mpc.json, as done here)
python tune_oracle.py --out ../../runs/formula --alphas 0.67 0.69 0.71 --betas 0.95 1.05
for d in g2 b btuned c mpc oracle c_mincurv c_bplan; do python bench.py --driver $d --per-level 1000; done
python compare.py --per-level 1000
```
On 4 CPU cores: C tuning 43 min, MPC tuning 52 min, each non-MPC benchmark 5 min per driver, MPC 21 min.
