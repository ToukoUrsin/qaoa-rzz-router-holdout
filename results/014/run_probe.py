"""Joint commuting-Ising scheduling screen on pinned original MQT inputs."""
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "results/011"))
sys.path.insert(0, str(ROOT / "results/013"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from mqt.bench import get_benchmark_alg
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Operator
import comparator
from ising_router import IsingRouter, normalize_qaoa
from run_bridge_probe import MappedCircuit
from run_probe import original_check, physical
from check_layer_certificates import check_layer, check_regrouping


def verify_fusions(fusions):
    rows = []
    for angle, candidate in fusions.items():
        expected = QuantumCircuit(2)
        expected.rzz(angle, 0, 1)
        expected.swap(0, 1)
        a, b = Operator(expected).data, Operator(candidate).data
        z = np.vdot(b.ravel(), a.ravel())
        error = float(np.max(np.abs(a - z / abs(z) * b)))
        bad = candidate.copy()
        index = next(i for i, x in enumerate(bad.data) if x.operation.name == "cx")
        del bad.data[index]
        c = Operator(bad).data
        z = np.vdot(c.ravel(), a.ravel())
        negative = float(np.max(np.abs(a - (z / abs(z) if abs(z) else 1) * c)))
        assert error < 1e-8 and negative > 1e-5 and candidate.count_ops()["cx"] == 3
        rows.append({"angle": angle, "full_operator_error": error, "negative_error": negative, "cx": 3})
    return rows


def summary(rows):
    valid = [r for r in rows if r["status"] == "ok"]
    if not valid:
        return {"valid": 0, "failures": len(rows)}
    inc = [r["incumbent"] for r in valid]
    raw = [r["candidate_cx"] for r in valid]
    selected = [min(i, c) for i, c in zip(inc, raw)]
    return {"valid": len(valid), "failures": len(rows) - len(valid),
            "incumbent_median": statistics.median(inc), "raw_candidate_median": statistics.median(raw),
            "selected_median": statistics.median(selected),
            "selected_ratio_gain": 1 - statistics.median(selected) / statistics.median(inc),
            "wins": sum(c < i for i, c in zip(inc, raw)), "ties": sum(c == i for i, c in zip(inc, raw)),
            "raw_losses": sum(c > i for i, c in zip(inc, raw)),
            "fused_rzz_swaps": sum(r["fused_rzz_swaps"] for r in valid)}


def run(args):
    cm = comparator.coupling_map(args.map)
    edges = set(cm.get_edges())
    reference = ROOT / "results/011" / f"{args.map}_mqt.json"
    prior_rows = {r["case"]: r for r in json.loads(reference.read_text())["rows"]}
    report = {"source_url": "https://github.com/cda-tum/mqt-bench", "versions": comparator.package_versions(),
              "map": args.map, "scope": "QAOA sizes 4..8; component screen only",
              "reference": str(reference.relative_to(ROOT)), "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
              "source_code_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in [
                  "src/ising_router.py", "results/014/check_layer_certificates.py", "results/014/run_probe.py"]},
              "case_timeout": 90, "beam": 64, "state_cap_per_layer": 30000, "rows": []}
    signal.signal(signal.SIGALRM, comparator._case_alarm)
    for size in (4, 5, 6, 7, 8):
        case = f"qaoa-{size}"
        row = {"case": case, "map": args.map}
        started = time.monotonic()
        try:
            signal.alarm(90)
            prior = prior_rows[case]
            source = comparator.mqt_clean(get_benchmark_alg("qaoa", circuit_size=size, random_parameters=False))
            fingerprint = comparator.mqt_fingerprint(source)
            assert fingerprint == prior["input_fingerprint"] and prior["status"] == "ok"
            before = comparator.qiskit_compile(source, cm, prior["best_seed"])
            assert comparator.count_qiskit(before) == prior["qiskit_best"]
            initial = list(before.layout.initial_index_layout(filter_ancillas=False))[:source.num_qubits]
            regrouping = None
            regrouping_check = None
            routed_source = source
            if args.regroup:
                routed_source, regrouping = normalize_qaoa(source)
                regrouping_check = check_regrouping(regrouping)
            candidate, certificates, fusions = IsingRouter(edges, cm.size()).route(routed_source, initial)
            checks = [check_layer(c, edges) for c in certificates]
            fusion_checks = verify_fusions(fusions)
            assert physical(candidate, edges)
            replay = original_check(source, MappedCircuit(candidate))
            assert replay["status"] == "checked"
            row.update(status="ok", input_fingerprint=fingerprint, incumbent=prior["incumbent"],
                       qiskit_best=prior["qiskit_best"], tket_materialized=prior["tket_materialized"],
                       candidate_cx=comparator.count_qiskit(candidate), certificates=certificates,
                       certificate_checks=checks, fusion_checks=fusion_checks, original_input_check=replay,
                       fused_rzz_swaps=sum(c["fused_rzz_swaps"] for c in checks),
                       regrouping=regrouping, regrouping_check=regrouping_check)
            artifact = args.out.parent / "circuits" / f"{args.map}-{case}.qpy"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            with artifact.open("wb") as stream:
                qpy.dump([source, before, candidate], stream)
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
        print(json.dumps({k: v for k, v in row.items() if k not in {"certificates", "certificate_checks"}}), flush=True)
    print("SUMMARY", json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--map", choices=["grid3", "heavy5"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--regroup", action="store_true")
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    run(a)
