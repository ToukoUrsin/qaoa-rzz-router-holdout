#!/usr/bin/env python3
"""Reproducible public circuit-quality comparator.

The charter's live comparator is deliberately kept small and explicit:

* input is an original public QASMBench ``*.qasm`` file (transpiled copies are
  excluded),
* Qiskit's optimization level 3 and pytket's offline IBM level-3 pass compile the same cleaned
  circuit to ``rz/sx/x/cx`` on the same coupling map,
* a two-qubit gate is one CX/CZ/etc.; a SWAP is three CX-equivalents, and
* the incumbent for each circuit is the lower of the two baseline counts.

The primary acceptance statistic is the ratio of medians, ``1 - median(candidate)
/ median(incumbent)``.  A result can only pass the charter's gate-count test
when both heavy-hex and grid scopes have at least 100 valid public circuits,
the candidate is no worse than the incumbent on every valid circuit, and the
candidate median is at least 5% below the incumbent median on each map.  The
per-circuit relative gains are emitted as a secondary, paired statistic.

This module does not download data.  The QASMBench checkout is an explicit
input so its commit and file hashes can be retained in the result packet.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import signal
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

# pytket writes a small local config during import. Keep this harness
# self-contained and writable even when the caller has no HOME config dir.
os.environ.setdefault("XDG_CONFIG_HOME", str(Path.cwd() / ".config"))

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_ibm_runtime.models.backend_configuration import QasmBackendConfiguration
from pytket.extensions.qiskit import IBMQBackend, qiskit_to_tk
from pytket.circuit import OpType


BASIS = ["rz", "sx", "x", "cx"]
TWO_Q_OPS = {
    "cx", "cz", "ecr", "iswap", "dcx", "rzz", "rxx", "ryy", "crx",
    "cry", "crz", "ch", "cy", "cs", "csx", "ccx", "ccz", "cswap",
}

MQT_BENCHMARKS = (
    "ae", "bv", "dj", "ghz", "graphstate", "grover", "half_adder",
    "qaoa", "qft", "qftentangled", "qnn", "qpeexact", "qpeinexact",
    "qwalk", "randomcircuit", "vqe_real_amp", "vqe_su2", "vqe_two_local",
    "wstate", "multiplier",
)


class CaseTimeout(RuntimeError):
    """Raised when one public generator case exceeds its bounded budget."""


def _case_alarm(_signum: int, _frame: object) -> None:
    raise CaseTimeout("per-case compilation timeout")


def clean(circuit: QuantumCircuit) -> QuantumCircuit:
    """Remove measurement-only decoration before comparing gate quality."""
    out = circuit.remove_final_measurements(inplace=False)
    out.data = [inst for inst in out.data if inst.operation.name != "barrier"]
    return out


def count_qiskit(circuit: QuantumCircuit) -> int:
    return sum(
        value * (3 if name == "swap" else 1)
        for name, value in circuit.count_ops().items()
        if name in TWO_Q_OPS or name == "swap"
    )


def count_tket(circuit: Any) -> int:
    return (
        circuit.n_gates_of_type(OpType.CX)
        + 3 * circuit.n_gates_of_type(OpType.SWAP)
        + circuit.n_gates_of_type(OpType.CZ)
        + circuit.n_gates_of_type(OpType.TK2)
        + circuit.n_gates_of_type(OpType.ISWAP)
        + circuit.n_gates_of_type(OpType.FSim)
    )


def coupling_map(name: str) -> CouplingMap:
    if name == "heavy5":
        return CouplingMap.from_heavy_hex(5)
    if name == "grid3":
        return CouplingMap.from_grid(3, 3)
    raise ValueError(f"unknown map {name!r}; choose heavy5 or grid3")


def backend_config(cm: CouplingMap, name: str) -> QasmBackendConfiguration:
    return QasmBackendConfiguration(
        backend_name=f"public-comparator-{name}",
        backend_version="1",
        n_qubits=cm.size(),
        basis_gates=BASIS,
        gates=[],
        local=True,
        simulator=True,
        conditional=False,
        open_pulse=False,
        memory=False,
        coupling_map=cm.get_edges(),
        meas_levels=[1],
        meas_kernels=[],
        discriminators=[],
    )


def qiskit_compile(circuit: QuantumCircuit, cm: CouplingMap, seed: int) -> QuantumCircuit:
    return transpile(
        circuit,
        coupling_map=cm,
        basis_gates=BASIS,
        optimization_level=3,
        seed_transpiler=seed,
        layout_method="sabre",
        routing_method="sabre",
    )


def tket_compile(circuit: QuantumCircuit, compiler: Any) -> Any:
    # Rebase before conversion so the two routes count the same native basis.
    local = transpile(circuit, basis_gates=BASIS, optimization_level=0)
    tk = qiskit_to_tk(local)
    compiler.apply(tk)
    return tk


def source_files(root: Path, max_qubits: int, tier_filter: str = "all") -> list[tuple[str, Path, int]]:
    """List original QASMBench files that could fit the chosen map."""
    found: list[tuple[str, Path, int]] = []
    for tier in ("small", "medium", "large"):
        if tier_filter != "all" and tier != tier_filter:
            continue
        tier_root = root / tier
        if not tier_root.exists():
            continue
        for path in sorted(tier_root.glob("*/*.qasm")):
            if path.name.endswith("_transpiled.qasm"):
                continue
            header = path.read_text(errors="ignore")
            widths = [int(v) for v in re.findall(r"(?m)^qreg\s+[^\[]+\[(\d+)\]", header)]
            width = sum(widths)
            if width and width <= max_qubits:
                found.append((tier, path, width))
    return found


def mqt_cases(limit: int) -> list[tuple[str, str, int]]:
    """Return deterministic, documented MQT Bench generator cases."""
    cases: list[tuple[str, str, int]] = []
    for name in MQT_BENCHMARKS:
        for size in (4, 5, 6, 7, 8):
            if name == "half_adder" and size % 2 == 0:
                continue
            if name == "multiplier" and size % 4 != 0:
                continue
            cases.append((f"{name}-{size}", name, size))
            if len(cases) >= limit:
                return cases
    # The generators above provide 94 cases. These fixed 9-qubit fillers make
    # a 100-case panel on both nine-qubit maps.
    for name in ("ae", "bv", "ghz", "wstate", "graphstate", "qft", "dj", "grover"):
        if len(cases) >= limit:
            break
        cases.append((f"{name}-9", name, 9))
    return cases


def mqt_clean(circuit: QuantumCircuit) -> QuantumCircuit:
    """Bind MQT's deterministic symbolic parameters and remove decoration."""
    out = circuit.remove_final_measurements(inplace=False)
    out.data = [inst for inst in out.data if inst.operation.name != "barrier"]
    if out.parameters:
        values = {p: 0.173 * (i + 1) for i, p in enumerate(sorted(out.parameters, key=str))}
        out = out.assign_parameters(values, inplace=False)
    return out


def mqt_fingerprint(circuit: QuantumCircuit) -> str:
    """Hash the concrete cleaned input, retaining generator provenance."""
    # QASM3 export is not defined for every numeric payload emitted by MQT
    # (some generators carry NumPy arrays).  A canonical instruction listing
    # is sufficient here and avoids silently dropping those inputs.
    instructions = []
    for instruction in circuit.data:
        instructions.append({
            "name": instruction.operation.name,
            "params": [repr(value) for value in instruction.operation.params],
            "qargs": [circuit.find_bit(bit).index for bit in instruction.qubits],
            "cargs": [circuit.find_bit(bit).index for bit in instruction.clbits],
        })
    payload = {"qubits": circuit.num_qubits, "clbits": circuit.num_clbits, "data": instructions}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def package_versions() -> dict[str, str]:
    names = ("qiskit", "pytket", "pytket-qiskit", "mqt.bench")
    versions = {name: importlib.metadata.version(name) for name in names}
    versions["python"] = sys.version.split()[0]
    return versions


def relative_gain(incumbent: int, candidate: int) -> float:
    return (incumbent - candidate) / incumbent if incumbent else 0.0


def summarize(rows: Iterable[dict[str, Any]], candidate_field: str | None = None) -> dict[str, Any]:
    """Aggregate completed rows without silently treating failures as zeros."""
    valid = [row for row in rows if row.get("status", "ok") == "ok"]
    result: dict[str, Any] = {"valid": len(valid)}
    if not valid:
        return result
    incumbents = [min(row["qiskit_l3"], row["tket_ibm_l3"]) for row in valid]
    result.update(
        qiskit_l3_median=statistics.median(row["qiskit_l3"] for row in valid),
        tket_ibm_l3_median=statistics.median(row["tket_ibm_l3"] for row in valid),
        incumbent_median=statistics.median(incumbents),
    )
    if candidate_field and all(candidate_field in row for row in valid):
        result["candidate_field"] = candidate_field
        candidates = [row[candidate_field] for row in valid]
        gains = [relative_gain(i, c) for i, c in zip(incumbents, candidates)]
        result.update(
            candidate_median=statistics.median(candidates),
            ratio_of_medians_gain=1 - statistics.median(candidates) / statistics.median(incumbents),
            paired_relative_gain_median=statistics.median(gains),
            candidate_nonworse=sum(c <= i for i, c in zip(incumbents, candidates)),
            candidate_wins=sum(c < i for i, c in zip(incumbents, candidates)),
            candidate_ties=sum(c == i for i, c in zip(incumbents, candidates)),
        )
    return result


def acceptance(summary_by_map: dict[str, dict[str, Any]], minimum_circuits: int = 100) -> dict[str, Any]:
    """Apply the frozen charter predicate to two map summaries."""
    reasons: list[str] = []
    for name in ("heavy5", "grid3"):
        summary = summary_by_map.get(name, {})
        if summary.get("valid", 0) < minimum_circuits:
            reasons.append(f"{name}: fewer than {minimum_circuits} valid public circuits")
        if "candidate_median" not in summary:
            reasons.append(f"{name}: candidate counts are absent")
        elif summary["ratio_of_medians_gain"] < 0.05:
            reasons.append(f"{name}: ratio-of-medians gain is below 5%")
        if summary.get("candidate_nonworse", 0) != summary.get("valid", 0):
            reasons.append(f"{name}: candidate regresses on at least one circuit")
    return {"pass": not reasons, "reasons": reasons}


def run_qasmbench(root: Path, map_name: str, limit: int, seeds: int, timeout: int, tier: str = "all") -> dict[str, Any]:
    """Compile a bounded public panel and write hashes for every attempted source."""
    cm = coupling_map(map_name)
    compiler = IBMQBackend.default_compilation_pass_offline(
        backend_config(cm, map_name), None, optimisation_level=3, timeout=timeout
    )
    files = source_files(root, cm.size(), tier_filter=tier)[:limit]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    old_handler = signal.signal(signal.SIGALRM, _case_alarm)
    try:
        for tier, path, header_qubits in files:
            row: dict[str, Any] = {
                "status": "ok",
                "tier": tier,
                "path": str(path.relative_to(root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "header_qubits": header_qubits,
            }
            try:
                signal.alarm(timeout)
                circuit = clean(QuantumCircuit.from_qasm_file(str(path)))
                if circuit.num_qubits > cm.size():
                    raise ValueError(f"parsed width {circuit.num_qubits} > map width {cm.size()}")
                qiskit_counts = [count_qiskit(qiskit_compile(circuit, cm, seed)) for seed in range(seeds)]
                tk = tket_compile(circuit, compiler)
                row.update(
                    qubits=circuit.num_qubits,
                    qiskit_l3=qiskit_counts[0],
                    qiskit_multistart=min(qiskit_counts),
                    qiskit_best_seed=min(range(seeds), key=lambda seed: qiskit_counts[seed]),
                    tket_ibm_l3=count_tket(tk),
                )
                row["incumbent"] = min(row["qiskit_l3"], row["tket_ibm_l3"])
            except Exception as exc:  # keep malformed public files visible, never as zeros
                row.update(status="failure", error=f"{type(exc).__name__}: {exc}")
                failures.append(row)
            finally:
                signal.alarm(0)
            rows.append(row)
    finally:
        signal.signal(signal.SIGALRM, old_handler)
    return {
        "source": "QASMBench original QASM files",
        "source_url": "https://github.com/pnnl/QASMBench",
        "map": map_name,
        "tier": tier,
        "seeds": seeds,
        "attempted": len(files),
        "failures": failures,
        "rows": rows,
        "summary": summarize(rows),
    }


def run_mqt(map_name: str, limit: int, seeds: int, timeout: int) -> dict[str, Any]:
    """Run the same baseline routes on deterministic MQT Bench generators."""
    from mqt.bench import get_benchmark_alg

    cm = coupling_map(map_name)
    compiler = IBMQBackend.default_compilation_pass_offline(
        backend_config(cm, map_name), None, optimisation_level=3, timeout=timeout
    )
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    old_handler = signal.signal(signal.SIGALRM, _case_alarm)
    try:
        for case, benchmark, size in mqt_cases(limit):
            started = time.time()
            base: dict[str, Any] = {
                "case": case, "benchmark": benchmark, "size": size, "map": map_name,
            }
            try:
                signal.alarm(timeout)
                # MQT's graphstate generator uses an internal RNG even when
                # parameter randomisation is disabled. Pin it so both the
                # baseline and candidate replay the same public input.
                kwargs = {"seed": 10} if benchmark == "graphstate" else {}
                circuit = mqt_clean(get_benchmark_alg(benchmark, circuit_size=size, random_parameters=False, **kwargs))
                if circuit.num_qubits > cm.size():
                    raise ValueError(f"circuit width {circuit.num_qubits} > map width {cm.size()}")
                qiskit_counts = [count_qiskit(qiskit_compile(circuit, cm, seed)) for seed in range(seeds)]
                tk = tket_compile(circuit, compiler)
                qiskit_l3 = qiskit_counts[0]
                qiskit_best = min(qiskit_counts)
                row = {
                    **base,
                    "status": "ok",
                    "qubits": circuit.num_qubits,
                    "input_fingerprint": mqt_fingerprint(circuit),
                    "qiskit_l3": qiskit_l3,
                    "qiskit_multistart": qiskit_best,
                    "qiskit_best_seed": min(range(seeds), key=lambda seed: qiskit_counts[seed]),
                    "tket_ibm_l3": count_tket(tk),
                    "incumbent": min(qiskit_l3, count_tket(tk)),
                    "seconds": time.time() - started,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
            except Exception as exc:
                failure = {
                    **base, "status": "failure", "error": f"{type(exc).__name__}: {exc}",
                    "seconds": time.time() - started,
                }
                failures.append(failure)
                print("FAIL", json.dumps(failure), flush=True)
            finally:
                signal.alarm(0)
    finally:
        signal.signal(signal.SIGALRM, old_handler)
    return {
        "schema": "mqt-comparator-v1",
        "source": "MQT Bench deterministic generators",
        "source_url": "https://github.com/cda-tum/mqt-bench",
        "map": map_name,
        "seeds": seeds,
        "per_case_timeout_seconds": timeout,
        "generator_random_parameters": False,
        "parameter_binding": "0.173*(sorted parameter index + 1)",
        "versions": package_versions(),
        "cases": [list(case) for case in mqt_cases(limit)],
        "rows": rows,
        "failures": failures,
        "summary": summarize(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qasmbench-root", type=Path)
    parser.add_argument("--source", choices=("qasmbench", "mqt"), default="qasmbench")
    parser.add_argument("--map", dest="map_name", choices=("heavy5", "grid3"), required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--tier", choices=("small", "medium", "large", "all"), default="all")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.source == "qasmbench":
        if args.qasmbench_root is None:
            parser.error("--qasmbench-root is required for --source qasmbench")
        result = run_qasmbench(args.qasmbench_root, args.map_name, args.limit, args.seeds, args.timeout, args.tier)
    else:
        result = run_mqt(args.map_name, args.limit, args.seeds, args.timeout)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
