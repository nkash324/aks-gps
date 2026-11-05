"""GPU version of the GPS stack's training sim: thousands of karts stepped at once (PyTorch).

A port of kart_gps (kart physics, low-level control, sensors, EKF, lap rules, driver B,
the residual env), step for step. The CPU sim in kart_gps stays the reference: every
benchmark on the results page runs there. scripts/gpu/check_sim.py compares the two.
Runs and checkpoints go to runs/gpu/, apart from the CPU runs.
"""
