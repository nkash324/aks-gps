#!/usr/bin/env bash
# Mac / Linux setup. GPU training (kart_gpu/, scripts/gpu/) needs an NVIDIA GPU with CUDA;
# everything else runs on the CPU.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch
.venv/bin/python -m pip install -r requirements.txt
echo "Done. Check it with: .venv/bin/python -m pytest tests"
