"""New bridge-vs-SWAP routing screen on the already matched public inputs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import signal
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "results/011"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mqt.bench import get_benchmark_alg
from qiskit import QuantumCircuit, transpile, qpy
import comparator
from bridge_router import BridgeRouter, bridge_ladder
from run_probe import original_check, physical
from check_routing_certificates import check


class MappedCircuit:
    def __init__(self, circuit):
        self.circuit = circuit
        metadata = circuit.metadata
        self.layout = SimpleNamespace(
            initial_index_layout=lambda **_: metadata["logical_initial"],
            final_index_layout=lambda **_: metadata["logical_final"])

    def __getattr__(self, name):
        return getattr(self.circuit, name)


def summarize(rows):
    valid = [r for r in rows if r["status"] == "ok"]
    if not valid:
        return {"valid": 0, "failures": len(rows)}
    incumbents = [r["incumbent"] for r in valid]
    candidates = [r["bridge_cx"] for r in valid]
    selected = [min(i, c) for i, c in zip(incumbents, candidates)]
    median = statistics.median(incumbents)
    return {"valid": len(valid), "failures": len(rows) - len(valid),
            "incumbent_median": median, "raw_bridge_median": statistics.median(candidates),
            "selected_median": statistics.median(selected),
            "selected_ratio_gain": 1 - statistics.median(selected) / median if median else 0,
            "wins": sum(c < i for i, c in zip(incumbents, candidates)),
            "ties": sum(c == i for i, c in zip(incumbents, candidates)),
            "raw_losses": sum(c > i for i, c in zip(incumbents, candidates)),
            "bridge_vs_swap_wins": sum(r["bridge_cx"] < r["swap_only_cx"] for r in valid),
            "bridge_vs_swap_losses": sum(r["bridge_cx"] > r["swap_only_cx"] for r in valid)}


def smoke(out):
    rows = []
    edges = {(i, i + 1) for i in range(7)} | {(i + 1, i) for i in range(7)}
    for length in range(3, 9):
        path = list(range(length))
        for reverse in (False, True):
            pairs = bridge_ladder(path)
            if reverse:
                pairs = list(reversed(pairs))
            cert = {"kind": "bridge", "path": path, "physical_cx": pairs,
                    "before": path, "after": path, "control": 0, "target": length - 1, "swaps": []}
            result = check(cert, edges)
            rows.append({"case": f"path-{length}-reverse-{reverse}", "certificates": [cert], "check": result})
    out.write_text(json.dumps({"edges": sorted(edges), "rows": rows}, indent=2) + "\n")
    print(json.dumps({"smoke_remote_cx": len(rows), "all_pass": True}))


def run(args):
    cm = comparator.coupling_map(args.map)
    edges = set(cm.get_edges())
    reference_path = ROOT / "results/011" / f"{args.map}_mqt.json"
    reference = json.loads(reference_path.read_text())
    references = {r["case"]: r for r in reference["rows"] if r["status"] == "ok"}
    cases = [s for s in comparator.mqt_cases(100) if not args.families or s[1] in args.families][:args.limit]
    if args.cases:
        cases = [s for s in comparator.mqt_cases(100) if s[0] in args.cases]
    report = {"map": args.map, "edges": sorted(edges), "versions": comparator.package_versions(),
              "reference": str(reference_path.relative_to(ROOT)),
              "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
              "baseline_scope": "Previously reproduced same-input eight-seed Qiskit/tket screen, with fresh best-Qiskit route count assertion; baseline scores are reused explicitly.",
              "source_url": "https://github.com/cda-tum/mqt-bench", "case_timeout": args.timeout,
              "beam": 8, "next_cx_lookahead": 6, "distance_weight": 1.5, "rows": []}
    signal.signal(signal.SIGALRM, comparator._case_alarm)
    for case, family, size in cases:
        row = {"case": case, "map": args.map}
        started = time.monotonic()
        try:
            signal.alarm(args.timeout)
            prior = references[case]
            source = comparator.mqt_clean(get_benchmark_alg(
                family, circuit_size=size, random_parameters=False,
                **({"seed": 10} if family == "graphstate" else {})))
            fingerprint = comparator.mqt_fingerprint(source)
            assert fingerprint == prior["input_fingerprint"]
            row["input_fingerprint"] = fingerprint
            base = comparator.qiskit_compile(source, cm, prior["best_seed"])
            assert comparator.count_qiskit(base) == prior["qiskit_best"]
            native = transpile(source, basis_gates=comparator.BASIS, optimization_level=3,
                               seed_transpiler=0, qubits_initially_zero=False)
            base_initial = list(base.layout.initial_index_layout(filter_ancillas=False))[:native.num_qubits]
            if native.layout is None:
                native_initial = list(range(native.num_qubits))
                native_final = list(range(native.num_qubits))
            else:
                native_initial = list(native.layout.initial_index_layout(filter_ancillas=False))
                native_final = list(native.layout.final_index_layout(filter_ancillas=False))
            assert len(native_initial) == len(base_initial) == native.num_qubits
            initial = [None] * native.num_qubits
            for input_index, native_wire in enumerate(native_initial):
                initial[native_wire] = base_initial[input_index]
            candidate, certs, search = BridgeRouter(edges, cm.size(), bridges=True).route(native, initial)
            swap_only, _, swap_search = BridgeRouter(edges, cm.size(), bridges=False).route(native, initial)
            # Compose the native compiler's permutation with the router's.
            # QFT may have virtual final swaps even without a coupling map.
            routed_final = candidate.metadata["logical_final"]
            candidate.metadata.update(
                logical_initial=base_initial[:source.num_qubits],
                logical_final=[routed_final[native_final[i]] for i in range(source.num_qubits)],
                native_initial_layout=native_initial, native_final_layout=native_final)
            checks = [check(c, edges) for c in certs]
            assert physical(candidate, edges)
            row.update(incumbent=prior["incumbent"], qiskit_best=prior["qiskit_best"],
                       tket_materialized=prior["tket_materialized"], best_seed=prior["best_seed"],
                       native_cx=comparator.count_qiskit(native), bridge_cx=comparator.count_qiskit(candidate),
                       swap_only_cx=comparator.count_qiskit(swap_only), certificates=certs,
                       certificate_checks=checks, search=search, swap_only_search=swap_search,
                       native_initial_layout=native_initial, native_final_layout=native_final)
            row["original_input_check"] = original_check(source, MappedCircuit(candidate))
            assert row["original_input_check"]["status"] == "checked"
            row["status"] = "ok"
            if row["bridge_cx"] < row["incumbent"]:
                artifact = args.out.parent / "circuits" / f"{args.map}-{case}.qpy"
                artifact.parent.mkdir(parents=True, exist_ok=True)
                with artifact.open("wb") as stream:
                    qpy.dump([source, base, candidate], stream)
                row["artifact"] = str(artifact.resolve().relative_to(ROOT))
                row["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        except Exception as exc:
            row.update(status="failure", error=f"{type(exc).__name__}: {exc}")
        finally:
            signal.alarm(0)
        row["seconds"] = time.monotonic() - started
        report["rows"].append(row)
        report["summary"] = summarize(report["rows"])
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k not in {"certificates", "certificate_checks"}}), flush=True)
    print("SUMMARY", json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "public"], default="public")
    p.add_argument("--map", choices=["grid3", "heavy5"], default="grid3")
    p.add_argument("--families", nargs="+")
    p.add_argument("--cases", nargs="+")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.mode == "smoke":
        smoke(a.out)
    else:
        run(a)
