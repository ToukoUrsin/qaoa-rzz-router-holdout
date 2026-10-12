#!/usr/bin/env bash
# Quick check: first 2 width-5 holdout inputs per map (compare with results/015/*_holdout.json rows).
set -euo pipefail
python -m pip install -q -r results/015/requirements.txt 2>&1 | grep -v -i -E "warning|notice" || true
for m in grid3 heavy5; do
  python results/015/run_holdout.py --map $m --widths 5 --per-width 2 --start-seed 2000 --out smoke-out/$m.json | grep SUMMARY
done
