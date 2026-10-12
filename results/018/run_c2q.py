"""Extra baseline for the frozen 100-input QAOA holdout (results/015): Qiskit's own commuting-gate flow
(FindCommutingPauliEvolutions + Commuting2qGateRouter with a line SwapStrategy, then basis translation and
optimization level 3 with the line placed on a physical path), best over 32 line orderings (identity + 31 seeded random), each transpiled on one simple physical path.
Same inputs, same scoring (comparator.count_qiskit), same layout-aware original-input check."""
from __future__ import annotations
import argparse, json, random, statistics, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for sub in ("src", "results/011", "results/013", "results/015"):
    sys.path.insert(0, str(ROOT / sub))
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate, CXGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import PassManager, CouplingMap
from qiskit.transpiler.passes import HighLevelSynthesis, InverseCancellation
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import (
    FindCommutingPauliEvolutions, Commuting2qGateRouter, SwapStrategy)
import comparator
from run_probe import original_check
from run_bridge_probe import MappedCircuit
from run_holdout import inputs


def rebuild(source: QuantumCircuit) -> QuantumCircuit:
    """Same QAOA semantics: H layer, then per QAOA layer k one PauliEvolutionGate holding every RZZ of that layer
    (all commute; layer index = number of RX already applied on the gate's qubits), then that layer's RX gates."""
    n = source.num_qubits
    rx_seen = [0] * n
    head, layers, mixers, touched = [], {}, {}, set()
    for inst in source.data:
        name = inst.operation.name; qs = [source.find_bit(q).index for q in inst.qubits]
        if name == "rzz":
            k = rx_seen[qs[0]]; assert rx_seen[qs[1]] == k; touched.update(qs)
            layers.setdefault(k, []).append((qs[0], qs[1], float(inst.operation.params[0])))
        elif name == "rx":
            mixers.setdefault(rx_seen[qs[0]], []).append((inst.operation, qs)); rx_seen[qs[0]] += 1
        elif name == "h" and rx_seen[qs[0]] == 0 and qs[0] not in touched:
            head.append((inst.operation, qs))
        else:
            raise ValueError(f"unexpected gate {name}")
    out = QuantumCircuit(n); out.global_phase = source.global_phase
    for op, qs in head:
        out.append(op, qs)
    for k in range(max(list(layers) + list(mixers)) + 1):
        if layers.get(k):
            terms = []
            for a, b, theta in layers[k]:
                label = ["I"] * n; label[a] = "Z"; label[b] = "Z"
                terms.append(("".join(reversed(label)), theta / 2))
            out.append(PauliEvolutionGate(SparsePauliOp.from_list(terms), time=1.0), range(n))
        for op, qs in mixers.get(k, []):
            out.append(op, qs)
    return out


def simple_paths(cm: CouplingMap, length: int, k: int, seed: int = 0):
    adj = {q: set() for q in range(cm.size())}
    for a, b in cm.get_edges():
        adj[a].add(b); adj[b].add(a)
    rng = random.Random(seed); found = []; seen = set()
    starts = list(adj); rng.shuffle(starts)
    def dfs(path):
        if len(found) >= k: return
        if len(path) == length:
            key = tuple(path) if path[0] < path[-1] else tuple(reversed(path))
            if key not in seen: seen.add(key); found.append(list(path))
            return
        nxt = list(adj[path[-1]] - set(path)); rng.shuffle(nxt)
        for q in nxt:
            dfs(path + [q])
            if len(found) >= k: return
    for s in starts:
        if len(found) >= k: break
        before = len(found); dfs([s])
    return found


def line_route(logical, order):
    """Commuting2qGateRouter on a line, logical qubit i starting at line position order[i]."""
    n = logical.num_qubits
    placed = QuantumCircuit(n); placed.global_phase = logical.global_phase
    placed.compose(logical, qubits=list(order), inplace=True)
    routed = PassManager([FindCommutingPauliEvolutions(),
                          Commuting2qGateRouter(SwapStrategy.from_line(list(range(n))))]).run(placed)
    at = [None] * n
    for i, j in enumerate(order):
        at[j] = i
    swaps = 0
    for inst in routed.data:
        if inst.operation.name == "swap":
            a, b = (routed.find_bit(q).index for q in inst.qubits)
            at[a], at[b] = at[b], at[a]; swaps += 1
    final_pos = [None] * n
    for j, i in enumerate(at):
        final_pos[i] = j
    return routed, list(order), final_pos, swaps


def c2q_route(source, cm, path, order=None, routed_info=None):
    n = source.num_qubits
    if routed_info is None:
        routed_info = line_route(rebuild(source), order if order is not None else list(range(n)))
    routed, start_pos, final_pos, _ = routed_info
    synth = transpile(routed, basis_gates=comparator.BASIS, optimization_level=0)  # swaps -> CX before L3
    synth = PassManager([InverseCancellation(gates_to_cancel=[CXGate()])]).run(synth)
    phys = transpile(synth, coupling_map=cm, basis_gates=comparator.BASIS, optimization_level=3,
                     initial_layout=path, seed_transpiler=0)
    assert "swap" not in phys.count_ops()
    tinit = list(phys.layout.initial_index_layout(filter_ancillas=False))[:n]
    tfin = list(phys.layout.final_index_layout(filter_ancillas=False))[:n]
    phys.metadata = {"logical_initial": [tinit[start_pos[i]] for i in range(n)],
                     "logical_final": [tfin[final_pos[i]] for i in range(n)]}
    return phys


def best_c2q(source, cm, n_orders, n_keep, n_paths, seed):
    n = source.num_qubits
    logical = rebuild(source)
    rng = random.Random(seed)
    orders = [list(range(n))] + [rng.sample(range(n), n) for _ in range(n_orders - 1)]
    routed = sorted((line_route(logical, o) for o in orders), key=lambda r: r[3])[:n_keep]
    best = None
    for info in routed:
        for path in simple_paths(cm, n, n_paths):
            phys = c2q_route(source, cm, path, routed_info=info)
            count = comparator.count_qiskit(phys)
            if best is None or count < best[0]:
                best = (count, path, info[1], info[3], phys)
    return best


def run(args):
    cm = comparator.coupling_map(args.map)
    edges = set(cm.get_edges()) | {(b, a) for a, b in cm.get_edges()}
    hold = {r["case"]: r for r in json.loads((ROOT / f"results/015/{args.map}_holdout.json").read_text())["rows"]}
    rows = []
    for case, width, seed, fingerprint, source in inputs([5, 6, 7, 8, 9], 20, 2000):
        h = hold[case]; assert h["input_fingerprint"] == fingerprint
        row = {"case": case, "width": width, "holdout_status": h["status"], "incumbent": h.get("incumbent"),
               "candidate_cx": h.get("candidate_cx")}
        t0 = time.monotonic()
        try:
            count, path, order, swaps, phys = best_c2q(source, cm, args.orders, args.keep, args.paths, seed)
            for inst in phys.data:
                if inst.operation.num_qubits == 2:
                    a, b = (phys.find_bit(q).index for q in inst.qubits); assert (a, b) in edges
            check = original_check(source, MappedCircuit(phys))
            assert check["status"] == "checked"
            row.update(status="ok", c2q_cx=count, c2q_path=path, c2q_line_order=order, c2q_line_swaps=swaps,
                       c2q_check=check)
        except Exception as exc:
            row.update(status="failure", error=f"{type(exc).__name__}: {exc}")
        row["seconds"] = time.monotonic() - t0
        rows.append(row); print(json.dumps({k: v for k, v in row.items() if k != "c2q_check"}), flush=True)
    ok = [r for r in rows if r["status"] == "ok"]
    both = [r for r in ok if r["holdout_status"] == "ok"]
    summ = {"map": args.map, "paths_per_input": args.paths, "line_orders_screened": args.orders,
            "orders_transpiled": args.keep, "c2q_valid": len(ok), "paired_with_candidate": len(both)}
    if both:
        c2q = [r["c2q_cx"] for r in both]; cand = [r["candidate_cx"] for r in both]; inc = [r["incumbent"] for r in both]
        summ.update(c2q_median=statistics.median(c2q), candidate_median=statistics.median(cand),
                    incumbent_l3_tket_median=statistics.median(inc),
                    candidate_gain_vs_c2q=1 - statistics.median(cand) / statistics.median(c2q),
                    candidate_wins_vs_c2q=sum(c < q for c, q in zip(cand, c2q)),
                    ties=sum(c == q for c, q in zip(cand, c2q)),
                    candidate_losses_vs_c2q=sum(c > q for c, q in zip(cand, c2q)),
                    c2q_beats_l3_tket=sum(q < i for q, i in zip(c2q, inc)))
    out = {"versions": comparator.package_versions(), "summary": summ, "rows": rows}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print("SUMMARY", json.dumps(summ), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--map", required=True, choices=["grid3", "heavy5"])
    p.add_argument("--paths", type=int, default=1)
    p.add_argument("--orders", type=int, default=32)
    p.add_argument("--keep", type=int, default=32)
    p.add_argument("--out", required=True)
    run(p.parse_args())
