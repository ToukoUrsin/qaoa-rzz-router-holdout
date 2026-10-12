#!/usr/bin/env bash
set -euo pipefail
python -m pip install -q -r results/015/requirements.txt
python -m pip freeze > results/015/pip_freeze.txt
python results/015/run_holdout.py --map grid3 --out results/015/grid3_holdout.json > results/015/grid3_raw.log 2>&1
python results/015/run_holdout.py --map heavy5 --out results/015/heavy5_holdout.json > results/015/heavy5_raw.log 2>&1
