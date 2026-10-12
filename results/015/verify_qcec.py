"""Independent fixed-wire, full-unitary QCEC checks of the frozen Ising routes.

The reference embeds the original logical input on the recorded initial wires,
then applies every physical SWAP from the layer certificates. This includes the
permutation of spectator inputs. The candidate is copied into a fresh circuit
without compiler layout metadata; all compact wires are ordinary inputs.
Only definitive DD results count. Every check has a subprocess hard timeout.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qiskit import QuantumCircuit, qpy
import comparator


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def circuits(row, report):
    artifact = ROOT / row["artifact"]
    assert sha(artifact) == row["artifact_sha256"]
    with artifact.open("rb") as stream:
        source, qiskit, tket, candidate, selected = qpy.load(stream)
    assert comparator.mqt_fingerprint(source) == row["input_fingerprint"]
    assert [comparator.count_qiskit(c) for c in [qiskit, tket, candidate, selected]] == [
        row["qiskit_best"], row["tket_cx"], row["candidate_cx"], row["selected_cx"]]
    edges = set(comparator.coupling_map(report["map"]).get_edges())
    for circuit in [qiskit, tket, candidate, selected]:
        assert all(len(inst.qubits) <= 2 and (len(inst.qubits) < 2 or
                   tuple(circuit.find_bit(q).index for q in inst.qubits) in edges)
                   for inst in circuit.data)
    # Bind grouping and layer certificates to the actual original QPY input.
    events = [{"id": i, "name": inst.operation.name,
               "q": [source.find_bit(q).index for q in inst.qubits],
               "params": [float(p) for p in inst.operation.params]}
              for i, inst in enumerate(source.data)]
    assert row["grouping"]["events"] == events
    order = row["grouping"]["normalized_order"]
    assert sorted(order) == list(range(len(events)))
    for wire in range(source.num_qubits):
        assert [e["id"] for e in events if wire in e["q"]] == [
            i for i in order if wire in events[i]["q"]]
    blocks = []
    pending = []
    for i in order:
        event = events[i]
        if event["name"] == "rzz":
            pending.append(event["q"] + event["params"])
        elif pending:
            blocks.append(pending)
            pending = []
    if pending:
        blocks.append(pending)
    assert blocks == [c["interactions"] for c in row["certificates"]]
    initial = candidate.metadata["logical_initial"]
    placement = initial[:]
    swaps = []
    support = set(initial)
    support.update(candidate.find_bit(q).index for inst in candidate.data for q in inst.qubits)
    for cert in row["certificates"]:
        assert cert["initial"] == placement
        consumed = []
        for action in cert["actions"]:
            a, b = action["physical"]
            assert (a, b) in edges
            support.update((a, b))
            if action["kind"] != "swap":
                index = action["interaction"]
                u, v, _ = cert["interactions"][index]
                assert {placement[u], placement[v]} == {a, b}
                consumed.append(index)
            if action["kind"] in ("swap", "rzz_swap"):
                swaps.append((a, b))
                placement = [b if p == a else a if p == b else p for p in placement]
        assert sorted(consumed) == list(range(len(cert["interactions"])))
        assert cert["final"] == placement
    assert placement == candidate.metadata["logical_final"]
    support = sorted(support)
    compact = {p: i for i, p in enumerate(support)}
    reference = QuantumCircuit(len(support))
    reference.compose(source, qubits=[compact[p] for p in initial], inplace=True)
    for a, b in swaps:
        reference.swap(compact[a], compact[b])
    routed = QuantumCircuit(len(support))
    routed.global_phase = candidate.global_phase
    for inst in candidate.data:
        routed.append(inst.operation, [compact[candidate.find_bit(q).index] for q in inst.qubits])
    bad = routed.copy()
    del bad.data[next(i for i, inst in enumerate(bad.data) if inst.operation.name == "cx")]
    return reference, routed, bad, {"physical_support": support, "physical_swaps": len(swaps),
                                    "spectator_inputs": len(support) - source.num_qubits}


def worker(args):
    from mqt import qcec
    with args.worker.open("rb") as stream:
        reference, candidate = qpy.load(stream)
    options = {"run_construction_checker": True, "run_alternating_checker": False,
               "run_simulation_checker": False, "run_zx_checker": False,
               "parallel": False, "nthreads": 1, "timeout": args.timeout,
               "numerical_tolerance": 1e-12, "trace_threshold": 1e-8,
               "reconstruct_swaps": False, "elide_permutations": False,
               "backpropagate_output_permutation": False}
    result = qcec.verify(reference, candidate, **options)
    payload = result.json() if hasattr(result, "json") else {}
    if isinstance(payload, str):
        payload = json.loads(payload)
    args.worker_out.write_text(json.dumps({"equivalence": str(result.equivalence),
                                           "result": payload,
                                           "options": options}, indent=2) + "\n")


def verify_pair(pair, timeout, work):
    packet = work / "pair.qpy"
    output = work / "result.json"
    if output.exists():
        output.unlink()
    with packet.open("wb") as stream:
        qpy.dump(pair, stream)
    started = time.monotonic()
    try:
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                  "--worker", str(packet), "--worker-out", str(output),
                                  "--timeout", str(timeout)], capture_output=True, text=True,
                                 timeout=timeout + 8)
        if process.returncode:
            return {"status": "failure", "returncode": process.returncode,
                    "stdout": process.stdout, "stderr": process.stderr,
                    "seconds": time.monotonic() - started}
        result = json.loads(output.read_text())
        result.update(status="completed", seconds=time.monotonic() - started)
        return result
    except subprocess.TimeoutExpired:
        return {"status": "hard_timeout", "seconds": time.monotonic() - started}


def run(args):
    args.inputs = [p.resolve() for p in args.inputs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {"checker": "MQT QCEC construction decision diagrams, full unitary, no simulation",
              "qcec_version": importlib.metadata.version("mqt.qcec"),
              "checker_source_sha256": sha(Path(__file__)),
              "documentation": "https://mqt.readthedocs.io/projects/qcec/en/latest/equivalence_checking.html",
              "scope": "original source embedded on initial physical wires, followed by all certified physical SWAPs; arbitrary spectator inputs included",
              "hard_timeout_per_check_seconds": args.timeout + 8,
              "inputs": {str(p.relative_to(ROOT)): sha(p) for p in args.inputs}, "rows": []}
    assert report["qcec_version"] == "3.11.0"
    accepted = {"equivalent", "equivalent_up_to_global_phase"}
    for path in args.inputs:
        data = json.loads(path.read_text())
        rows = [r for r in data["rows"] if r["status"] == "ok"]
        for row in rows[:args.limit] if args.limit else rows:
            checked = {"map": data["map"], "case": row["case"],
                       "input_fingerprint": row["input_fingerprint"],
                       "artifact": row["artifact"], "artifact_sha256": row["artifact_sha256"]}
            try:
                reference, candidate, bad, stats = circuits(row, data)
                checked.update(stats)
                with tempfile.TemporaryDirectory(dir=args.out.parent) as directory:
                    work = Path(directory)
                    checked["positive"] = verify_pair([reference, candidate], args.timeout, work)
                    checked["negative"] = verify_pair([reference, bad], args.timeout, work)
                pos = str(checked["positive"].get("equivalence", ""))
                neg = str(checked["negative"].get("equivalence", ""))
                checked["status"] = "checked" if (
                    (pos.endswith("equivalent") or pos.endswith("equivalent_up_to_global_phase")) and
                    neg.endswith("not_equivalent")) else "failure"
            except Exception as exc:
                checked.update(status="failure", error=f"{type(exc).__name__}: {exc}")
            report["rows"].append(checked)
            report["summary"] = {"attempted": len(report["rows"]),
                                 "checked": sum(r["status"] == "checked" for r in report["rows"]),
                                 "failures": sum(r["status"] != "checked" for r in report["rows"])}
            args.out.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(checked), flush=True)
    print("SUMMARY", json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=12)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--worker-out", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        run(args)
