# QAOA RZZ-layer router: 100-input holdout on MQT Bench QAOA

A router for QAOA cost layers that orders the commuting RZZ terms together with the SWAPs, and fuses an RZZ
followed by a SWAP on the same pair into one 3-CX block (`src/ising_router.py`, bounded beam search: beam 64,
state cap 30,000, region cap 12, all frozen before the holdout).

## Result (predeclared holdout, run 2026-10-12)

100 distinct MQT Bench 2.3.0 QAOA circuits (widths 5-9, 20 per width, graph seeds from 2000, deterministic
parameter binding). Score = two-qubit gate count after routing, SWAP = 3 CX. Incumbent per circuit = the lower of
Qiskit 2.5.2 `optimization_level=3` (Sabre, best of seeds 0-7) and pytket 2.18.5 / pytket-qiskit 0.78.0 offline
IBM level 3.

| map | valid | Qiskit L3 best-of-8 median | tket median | router median | median reduction | wins / ties / losses vs incumbent |
|---|---|---|---|---|---|---|
| 3x3 grid (`CouplingMap.from_grid(3, 3)`) | 100/100 | 54 | 67 | 48 | 11.1% | 89 / 10 / 1 |
| heavy-hex d=5 (`CouplingMap.from_heavy_hex(5)`) | 83/100 | 64 | 85 | 50 | 21.9% | 78 / 5 / 0 |

On heavy-hex, 17 inputs exceed the router's 12-qubit region cap and the router refuses them (they fall back to
the incumbent); they are counted as failures, not wins. Total CX over valid rows: 5,803 -> 4,960 (grid),
5,801 -> 4,389 (heavy-hex).

Every routed circuit is checked against the original input with layout-aware statevector probes (basis and random
complex inputs, plus a perturbed negative control); each changed layer carries an exhaustive local certificate.
MQT QCEC 3.11.0 (construction checker, 20 s cap per check) proves 115 of the 183 routed circuits equivalent to
the original (all of widths 5-7, two of width 8); the other 68 (widths 8-9) time out or give no information, and
none is reported non-equivalent (`results/015/qcec_full.json`, `results/015/audit_full.json`).

Qiskit's own commuting-gate flow (`FindCommutingPauliEvolutions` + `Commuting2qGateRouter` with a line
`SwapStrategy`, best of 32 line orderings, then level 3 on a physical path) gives medians 103 (grid) and 106
(heavy-hex) on the same inputs (`results/018/`), so Sabre L3 is the stronger baseline here.

## Reproduce

```bash
pip install -r results/015/requirements.txt          # qiskit 2.5.2, pytket 2.18.5, pytket-qiskit 0.78.0, mqt.bench 2.3.0
bash smoke.sh                                         # 2 inputs per map, ~1 min
bash results/015/run_full.sh                          # full holdout, both maps (~2 min on 4 vCPU)
bash results/018/run.sh                               # Commuting2qGateRouter baseline
pip install mqt.qcec==3.11.0 && bash results/015/run_qcec_full.sh
python results/015/audit_holdout.py --qcec results/015/qcec_full.json --out results/015/audit_full.json
```

Scope: QAOA circuits only, small widths (5-9) and two coupling maps. Not tested on other circuit families.
