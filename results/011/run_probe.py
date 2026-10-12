"""Pinned matched-input public screen of the exact phase automaton."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("XDG_CONFIG_HOME", str(ROOT / ".config"))

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Statevector
from pytket.extensions.qiskit import IBMQBackend, tk_to_qiskit
import comparator as baseline
from phase_automaton import PhaseAutomaton
from check_certificates import check


def physical(circuit, edges):
    return all(len(inst.qubits) <= 1 or
               (len(inst.qubits) == 2 and tuple(circuit.find_bit(q).index for q in inst.qubits) in edges)
               for inst in circuit.data)


def embedding(v, wires, width):
    out = np.zeros(1 << width, complex)
    for i, amplitude in enumerate(v):
        j = sum(((i >> bit) & 1) << position for bit, position in enumerate(wires))
        out[j] = amplitude
    return out


def original_check(source, compiled, cap=18):
    initial = list(compiled.layout.initial_index_layout(filter_ancillas=False))[:source.num_qubits]
    final = list(compiled.layout.final_index_layout(filter_ancillas=False))[:source.num_qubits]
    active = sorted({compiled.find_bit(q).index for inst in compiled.data for q in inst.qubits}
                    | set(initial) | set(final))
    if len(active) > cap:
        return {"status": "not_checked", "reason": f"active width {len(active)} > cap {cap}"}
    pos = {p: i for i, p in enumerate(active)}
    compact = QuantumCircuit(len(active))
    compact.global_phase = compiled.global_phase
    for inst in compiled.data:
        compact.append(inst.operation, [pos[compiled.find_bit(q).index] for q in inst.qubits], [])
    inp, out = [pos[p] for p in initial], [pos[p] for p in final]
    rng = np.random.default_rng(110)
    probes = []
    for i in (0, 1, (1 << source.num_qubits) - 1):
        v = np.zeros(1 << source.num_qubits, complex)
        v[i] = 1
        probes.append(v)
    v = rng.normal(size=1 << source.num_qubits) + 1j * rng.normal(size=1 << source.num_qubits)
    probes.append(v / np.linalg.norm(v))
    errors = []
    negatives = []
    phase = None
    for v in probes:
        expected = embedding(Statevector(v).evolve(source).data, out, len(active))
        got = Statevector(embedding(v, inp, len(active))).evolve(compact).data
        if phase is None:
            overlap = np.vdot(got, expected)
            phase = overlap / abs(overlap)
        errors.append(float(np.linalg.norm(expected - phase * got)))
        corruption = QuantumCircuit(1)
        corruption.rz(0.031, 0)
        bad = Statevector(got).evolve(corruption, qargs=[out[0]]).data
        overlap = np.vdot(bad, expected)
        negatives.append(float(np.linalg.norm(expected - overlap / abs(overlap) * bad)))
    # The random complex probe catches relative phase errors invisible on basis inputs.
    assert max(errors) < 1e-7, f"original-input mismatch: max_norm_error={max(errors):.12g}, errors={errors}"
    assert negatives[-1] > 1e-5, f"negative control invisible: errors={negatives}"
    return {"status": "checked", "active_wires": len(active), "probes": len(probes),
            "shared_global_phase": True, "max_norm_error": max(errors),
            "random_probe_negative_error": negatives[-1]}


def summary(rows):
    valid = [r for r in rows if r.get("status") == "ok"]
    if not valid:
        return {"valid": 0}
    inc = [r["incumbent"] for r in valid]
    raw = [r["automaton_cx"] for r in valid]
    selected = [min(i, c) for i, c in zip(inc, raw)]
    med = statistics.median(inc)
    return {"valid": len(valid), "incumbent_median": med,
            "automaton_median": statistics.median(raw),
            "selected_median": statistics.median(selected),
            "selected_ratio_gain": 1 - statistics.median(selected) / med if med else 0,
            "wins": sum(c < i for i, c in zip(inc, raw)),
            "ties": sum(c == i for i, c in zip(inc, raw)),
            "raw_losses": sum(c > i for i, c in zip(inc, raw)),
            "improved_qiskit_rows": sum(r["automaton_cx"] < r["qiskit_best"] for r in valid),
            "replacements": sum(len(r["certificates"]) for r in valid),
            "cx_saved_from_qiskit": sum(r["qiskit_best"] - r["automaton_cx"] for r in valid)}


def smoke(out):
    rows = []
    edges = {(0, 1), (1, 0), (1, 2), (2, 1)}
    solver = PhaseAutomaton(edges)
    rng = random.Random(110)
    for i in range(12):
        q = QuantumCircuit(3)
        for _ in range(8):
            c, t = rng.choice(sorted(edges))
            q.cx(c, t)
            q.rz(rng.uniform(-1, 1), rng.randrange(3))
        q.cx(0, 1)
        q.cx(0, 1)
        candidate, certs, stats = solver.optimize(q)
        checks = [check(c) for c in certs]
        assert certs and physical(candidate, edges)
        rows.append({"case": f"synthetic-{i}", "certificates": certs, "checks": checks,
                     "stats": stats, "status": "ok"})
    out.write_text(json.dumps({"scope": "synthetic smoke, not public benchmark", "rows": rows}, indent=2))
    print(json.dumps({"smoke_rows": len(rows), "certificates": sum(len(r["certificates"]) for r in rows)}))


def public(args):
    from mqt.bench import get_benchmark_alg
    versions = baseline.package_versions()
    assert versions["qiskit"] == "2.5.2" and versions["pytket"] == "2.18.4"
    assert versions["pytket-qiskit"] == "0.78.0" and versions["mqt.bench"] == "2.3.0"
    cm = baseline.coupling_map(args.map)
    edges = set(cm.get_edges())
    solver = PhaseAutomaton(edges, width=args.width, max_states=args.max_states)
    compiler = IBMQBackend.default_compilation_pass_offline(
        baseline.backend_config(cm, args.map), None, optimisation_level=3, timeout=20)
    if args.qasm_root:
        specs = [(str(path.relative_to(args.qasm_root)), "qasmbench", width)
                 for _, path, width in baseline.source_files(args.qasm_root, 9)]
    else:
        specs = baseline.mqt_cases(100)
    if args.families:
        specs = [s for s in specs if s[1] in args.families]
    if args.cases:
        specs = [s for s in specs if s[0] in args.cases]
    specs = specs[:args.limit]
    rows = []
    report = {"schema": "exact-edge-parity-screen-v1", "map": args.map, "versions": versions,
              "source_url": "https://github.com/pnnl/QASMBench" if args.qasm_root else "https://github.com/cda-tum/mqt-bench",
              "numpy_version": np.__version__, "seeds": list(range(args.seeds)),
              "objective": "CX count; phase parities visited and exact final GF(2) basis",
              "width": args.width, "max_states_per_block": args.max_states,
              "case_timeout": args.timeout, "rows": rows}
    signal.signal(signal.SIGALRM, baseline._case_alarm)
    for case, family, size in specs:
        started = time.monotonic()
        row = {"case": case, "family": family, "size": size, "map": args.map}
        try:
            signal.alarm(args.timeout)
            if args.qasm_root:
                input_path = args.qasm_root / case
                row["source_sha256"] = hashlib.sha256(input_path.read_bytes()).hexdigest()
                source = baseline.clean(QuantumCircuit.from_qasm_file(str(input_path)))
            else:
                source = baseline.mqt_clean(get_benchmark_alg(
                    family, circuit_size=size, random_parameters=False,
                    **({"seed": 10} if family == "graphstate" else {})))
            row["input_fingerprint"] = baseline.mqt_fingerprint(source)
            circuits = [baseline.qiskit_compile(source, cm, seed) for seed in range(args.seeds)]
            counts = [baseline.count_qiskit(q) for q in circuits]
            seed = min(range(args.seeds), key=lambda s: counts[s])
            before = circuits[seed]
            tk = baseline.tket_compile(source, compiler)
            tket = tk_to_qiskit(tk, replace_implicit_swaps=True, perm_warning=False)
            assert physical(before, edges) and physical(tket, edges)
            candidate, certs, stats = solver.optimize(before)
            assert physical(candidate, edges)
            checks = [check(c) for c in certs]
            after = baseline.count_qiskit(candidate)
            assert after == counts[seed] - sum(c["before_cx"] - c["after_cx"] for c in certs)
            row.update(status="ok", qiskit_counts=counts, qiskit_best=counts[seed], best_seed=seed,
                       tket_materialized=baseline.count_qiskit(tket), automaton_cx=after,
                       incumbent=min(counts[seed], baseline.count_qiskit(tket)),
                       certificates=certs, certificate_checks=checks, search=stats,
                       initial_layout=list(before.layout.initial_index_layout(filter_ancillas=False)),
                       final_layout=list(before.layout.final_index_layout(filter_ancillas=False)))
            if certs:
                artifact_case = case.replace("/", "_").replace(".qasm", "")
                artifact = args.out.parent / "circuits" / f"{args.map}-{artifact_case}.qpy"
                artifact.parent.mkdir(parents=True, exist_ok=True)
                with artifact.open("wb") as stream:
                    qpy.dump([source, before, candidate], stream)
                row["artifact"] = str(artifact.resolve().relative_to(ROOT))
                row["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
                row["original_input_check"] = original_check(source, candidate)
        except Exception as exc:
            row.update(status="failure", error=f"{type(exc).__name__}: {exc}")
        finally:
            signal.alarm(0)
        row["seconds"] = time.monotonic() - started
        rows.append(row)
        report["summary"] = summary(rows)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k not in {
            "certificates", "certificate_checks", "initial_layout", "final_layout"}}), flush=True)
    print("SUMMARY", json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "public"], default="public")
    p.add_argument("--map", choices=["grid3", "heavy5"], default="grid3")
    p.add_argument("--width", type=int, default=3)
    p.add_argument("--max-states", type=int, default=150_000)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--families", nargs="+")
    p.add_argument("--cases", nargs="+")
    p.add_argument("--qasm-root", type=Path)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--timeout", type=int, default=90)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.mode == "smoke":
        smoke(a.out)
    else:
        public(a)
