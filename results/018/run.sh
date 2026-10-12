#!/usr/bin/env bash
set -euo pipefail
python -m pip install -q -r results/015/requirements.txt 2>&1 | grep -v -i -E "warning|notice" || true
python results/018/run_c2q.py --map grid3 --out results/018/grid3_c2q.json > results/018/grid3_c2q.log 2>&1 &
python results/018/run_c2q.py --map heavy5 --out results/018/heavy5_c2q.json > results/018/heavy5_c2q.log 2>&1 &
wait
grep -h SUMMARY results/018/grid3_c2q.log results/018/heavy5_c2q.log
