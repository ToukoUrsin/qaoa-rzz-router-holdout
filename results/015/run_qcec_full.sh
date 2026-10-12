#!/usr/bin/env bash
set -euo pipefail
python -m pip install -q -r results/015/requirements.txt mqt.qcec==3.11.0
python -m pip freeze > results/015/qcec_pip_freeze.txt
python results/015/verify_qcec.py --inputs results/015/grid3_holdout.json results/015/heavy5_holdout.json --out results/015/qcec_full.json > results/015/qcec_full.log 2>&1
