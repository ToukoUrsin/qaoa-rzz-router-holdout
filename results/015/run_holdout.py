"""Predeclared unique public QAOA holdout with fresh current baselines.

No tuning on heldout results. First K distinct fingerprints per width, starting
at a declared graph seed. All reported comparators compile the exact same input.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import signal
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for sub in ("src", "results/011"):
    sys.path.insert(0, str(ROOT / sub))
from run_probe import original_check, physical
sys.path.insert(0, str(ROOT / "results/014"))
sys.path.insert(0, str(ROOT / "results/013"))
from mqt.bench import get_benchmark_alg
from qiskit import transpile, qpy
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit, IBMQBackend
from pytket.predicates import CompilationUnit
import comparator
from ising_router import IsingRouter, normalize_qaoa
from check_layer_certificates import check_layer, check_regrouping
from run_bridge_probe import MappedCircuit


def check_fusions(fusions):
    # Import from a uniquely named module to avoid run_probe name collision.
    import importlib.util
    spec = importlib.util.spec_from_file_location("ising_screen", ROOT / "results/014/run_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_fusions(fusions)


def inputs(widths, per_width, start_seed):
    result = []
    seen = set()
    for width in widths:
        seed = start_seed
        admitted = 0
        while admitted < per_width:
            source = comparator.mqt_clean(get_benchmark_alg(
                "qaoa", circuit_size=width, random_parameters=False, seed=seed))
            fingerprint = comparator.mqt_fingerprint(source)
            if fingerprint not in seen:
                seen.add(fingerprint)
                result.append((f"qaoa-{width}-graphseed-{seed}", width, seed, fingerprint, source))
                admitted += 1
            seed += 1
    return result


def summary(rows):
    valid = [r for r in rows if r["status"] == "ok"]
    if not valid:
        return {"valid": 0, "attempted": len(rows)}
    inc = [r["incumbent"] for r in valid]
    raw = [r["candidate_cx"] for r in valid]
    selected = [min(i, c) for i, c in zip(inc, raw)]
    median = statistics.median(inc)
    return {"attempted": len(rows), "valid": len(valid), "failures": len(rows) - len(valid),
            "incumbent_median": median, "raw_candidate_median": statistics.median(raw),
            "selected_median": statistics.median(selected),
            "raw_ratio_gain": 1 - statistics.median(raw) / median,
            "selected_ratio_gain": 1 - statistics.median(selected) / median,
            "wins": sum(c < i for i, c in zip(inc, raw)), "ties": sum(c == i for i, c in zip(inc, raw)),
            "raw_losses": sum(c > i for i, c in zip(inc, raw)), "selected_losses": 0,
            "original_input_checks": 3 * len(valid)}


def tket_route(source, compiler):
    local = transpile(source, basis_gates=comparator.BASIS, optimization_level=0, qubits_initially_zero=False)
    logical = qiskit_to_tk(local)
    original_units = list(logical.qubits)
    unit = CompilationUnit(logical)
    compiler.apply(unit)
    circuit = tk_to_qiskit(unit.circuit, replace_implicit_swaps=True, perm_warning=False)
    initial = [unit.initial_map[q].index[0] for q in original_units]
    final = [unit.final_map[q].index[0] for q in original_units]
    circuit.metadata = {"logical_initial": initial, "logical_final": final}
    return circuit


def run(args):
    versions = comparator.package_versions()
    assert versions["qiskit"] == "2.5.2" and versions["pytket"] == "2.18.5"
    assert versions["pytket-qiskit"] == "0.78.0" and versions["mqt.bench"] == "2.3.0"
    cm = comparator.coupling_map(args.map)
    edges = set(cm.get_edges())
    compiler = IBMQBackend.default_compilation_pass_offline(
        comparator.backend_config(cm, args.map), None, optimisation_level=3, timeout=20)
    dataset = inputs(args.widths, args.per_width, args.start_seed)
    manifest = {"source_url": "https://github.com/cda-tum/mqt-bench", "generator": "qaoa",
                "random_parameters": False, "binding": "0.173*(sorted parameter index+1)",
                "selection": "first declared number of distinct fingerprints per width; no objective-dependent selection",
                "widths": args.widths, "per_width": args.per_width, "start_seed": args.start_seed,
                "rows": [{"case": c, "width": w, "seed": s, "input_fingerprint": f} for c, w, s, f, _ in dataset]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    (args.out.parent / f"{args.map}_input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = {"versions": versions, "map": args.map, "source": manifest,
              "comparator": "fresh best-of-eight Qiskit L3 seeds 0..7 and current pytket 2.18.5 offline IBM L3, materialized swaps",
              "scope": "new QAOA-family holdout, not the frozen mixed MQT+QASMBench acceptance scope",
              "parameters_frozen_before_holdout": {"beam": 64, "state_cap": 30000, "region_cap": 12},
              "source_code_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in [
                  "src/ising_router.py", "results/014/check_layer_certificates.py", "results/015/run_holdout.py"]},
              "rows": []}
    signal.signal(signal.SIGALRM, comparator._case_alarm)
    for case, width, seed, fingerprint, source in dataset:
        row = {"case": case, "width": width, "seed": seed, "input_fingerprint": fingerprint}
        started = time.monotonic()
        try:
            signal.alarm(90)
            routes = [comparator.qiskit_compile(source, cm, s) for s in range(8)]
            counts = [comparator.count_qiskit(q) for q in routes]
            best_seed = min(range(8), key=lambda s: counts[s])
            qiskit = routes[best_seed]
            tket = tket_route(source, compiler)
            assert physical(qiskit, edges) and physical(tket, edges)
            qiskit_check = original_check(source, qiskit)
            tket_check = original_check(source, MappedCircuit(tket))
            assert qiskit_check["status"] == tket_check["status"] == "checked"
            normalized, grouping = normalize_qaoa(source)
            grouping_check = check_regrouping(grouping)
            initial = list(qiskit.layout.initial_index_layout(filter_ancillas=False))[:source.num_qubits]
            candidate, certs, fusions = IsingRouter(edges, cm.size()).route(normalized, initial)
            certificate_checks = [check_layer(c, edges) for c in certs]
            fusion_checks = check_fusions(fusions)
            assert physical(candidate, edges)
            candidate_check = original_check(source, MappedCircuit(candidate))
            assert candidate_check["status"] == "checked"
            tket_count = comparator.count_qiskit(tket)
            incumbent = min(counts[best_seed], tket_count)
            candidate_count = comparator.count_qiskit(candidate)
            selected = candidate if candidate_count < incumbent else (qiskit if counts[best_seed] <= tket_count else tket)
            row.update(status="ok", qiskit_counts=counts, qiskit_best=counts[best_seed], best_seed=best_seed,
                       tket_cx=tket_count, incumbent=incumbent, candidate_cx=candidate_count,
                       selected_cx=min(incumbent, candidate_count), grouping=grouping, grouping_check=grouping_check,
                       certificates=certs, certificate_checks=certificate_checks, fusion_checks=fusion_checks,
                       qiskit_check=qiskit_check, tket_check=tket_check, candidate_check=candidate_check,
                       selected_route="candidate" if candidate_count < incumbent else ("qiskit" if counts[best_seed] <= tket_count else "tket"))
            artifact = args.out.parent / "circuits" / f"{args.map}-{case}.qpy"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            with artifact.open("wb") as stream:
                qpy.dump([source, qiskit, tket, candidate, selected], stream)
            row["artifact"] = str(artifact.resolve().relative_to(ROOT))
            row["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        except Exception as exc:
            row.update(status="failure", error=f"{type(exc).__name__}: {exc}")
        finally:
            signal.alarm(0)
        row["seconds"] = time.monotonic() - started
        report["rows"].append(row)
        report["summary"] = summary(report["rows"])
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k not in {
            "grouping", "grouping_check", "certificates", "certificate_checks", "fusion_checks"}}), flush=True)
    print("SUMMARY", json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--map", choices=["grid3", "heavy5"], required=True)
    p.add_argument("--widths", type=int, nargs="+", default=[5, 6, 7, 8, 9])
    p.add_argument("--per-width", type=int, default=20)
    p.add_argument("--start-seed", type=int, default=2000)
    p.add_argument("--out", type=Path, required=True)
    run(p.parse_args())
