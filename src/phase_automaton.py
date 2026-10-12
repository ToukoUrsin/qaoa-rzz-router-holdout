"""Exact hardware-edge CX synthesis for small linear/phase blocks.

The finite state is (GF(2) row basis, set of phase parities visited).
RZ gates cost zero CX and can be placed when their parity is first visited.
The terminal row basis equals the original block's basis: no wire relabeling.
Optimality is only within this CX/RZ phase-polynomial representation, not
over arbitrary gate sets or modulo phase-polynomial identities.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

from qiskit import QuantumCircuit


@dataclass
class SearchResult:
    gates: list[tuple[int, int]] | None
    expanded: int


def phase_form(n: int, ops: list[dict]) -> tuple[tuple[int, ...], dict[int, float]]:
    rows = [1 << i for i in range(n)]
    angles: dict[int, float] = {}
    for op in ops:
        if op["name"] == "cx":
            c, t = op["q"]
            rows[t] ^= rows[c]
        elif op["name"] == "rz":
            mask = rows[op["q"][0]]
            angles[mask] = angles.get(mask, 0.0) + op["angle"]
        else:
            raise ValueError(op["name"])
    return tuple(rows), {m: a for m, a in angles.items() if a != 0.0}


def shortest_walk(n: int, edges: tuple[tuple[int, int], ...],
                  target: tuple[int, ...], masks: tuple[int, ...],
                  max_depth: int, max_states: int = 150_000) -> SearchResult:
    """BFS; None means no strict improvement within the stated caps."""
    flags = {m: 1 << i for i, m in enumerate(masks)}
    full = (1 << len(masks)) - 1
    initial = tuple(1 << i for i in range(n))
    seen = sum(flags.get(m, 0) for m in initial)
    start = (initial, seen)
    queue = deque([(start, 0)])
    parents = {start: None}
    goal = None
    while queue:
        state, depth = queue.popleft()
        rows, visited = state
        if rows == target and visited == full:
            goal = state
            break
        if depth == max_depth:
            continue
        for c, t in edges:
            changed = list(rows)
            changed[t] ^= changed[c]
            nxt = (tuple(changed), visited | flags.get(changed[t], 0))
            if nxt in parents:
                continue
            if len(parents) >= max_states:
                return SearchResult(None, len(parents))
            parents[nxt] = (state, (c, t))
            queue.append((nxt, depth + 1))
    if goal is None:
        return SearchResult(None, len(parents))
    gates = []
    while parents[goal] is not None:
        prev, gate = parents[goal]
        gates.append(gate)
        goal = prev
    return SearchResult(list(reversed(gates)), len(parents))


def place_phases(n: int, gates: list[tuple[int, int]], angles: dict[int, float]) -> list[dict]:
    rows = [1 << i for i in range(n)]
    todo = dict(angles)
    out = []

    def emit() -> None:
        for wire, mask in enumerate(rows):
            if mask in todo:
                out.append({"name": "rz", "q": [wire], "angle": todo.pop(mask)})

    emit()
    for c, t in gates:
        out.append({"name": "cx", "q": [c, t]})
        rows[t] ^= rows[c]
        emit()
    assert not todo
    return out


class PhaseAutomaton:
    def __init__(self, physical_edges: Iterable[tuple[int, int]], width: int = 3,
                 max_states: int = 150_000):
        self.physical_edges = set(physical_edges)
        self.width = width
        self.max_states = max_states
        self.cache: dict[tuple, SearchResult] = {}

    def optimize(self, circuit: QuantumCircuit) -> tuple[QuantumCircuit, list[dict], dict]:
        out = circuit.copy_empty_like()
        out._layout = circuit.layout
        certificates = []
        stats = {"blocks": 0, "searched": 0, "cache_hits": 0, "expanded": 0}
        block = []
        support = set()

        def flush() -> None:
            nonlocal block, support
            if not block:
                return
            stats["blocks"] += 1
            wires = sorted(support)
            pos = {w: i for i, w in enumerate(wires)}
            ops = []
            for inst in block:
                op = {"name": inst.operation.name,
                      "q": [pos[circuit.find_bit(q).index] for q in inst.qubits]}
                if op["name"] == "rz":
                    op["angle"] = float(inst.operation.params[0])
                ops.append(op)
            before = sum(op["name"] == "cx" for op in ops)
            replacement = None
            if before >= 3:
                n = len(wires)
                edges = tuple(sorted((pos[a], pos[b]) for a, b in self.physical_edges
                                     if a in pos and b in pos))
                target, angles = phase_form(n, ops)
                masks = tuple(sorted(angles))
                key = (n, edges, target, masks, before - 1)
                if key in self.cache:
                    result = self.cache[key]
                    stats["cache_hits"] += 1
                else:
                    result = shortest_walk(n, edges, target, masks, before - 1, self.max_states)
                    self.cache[key] = result
                    stats["searched"] += 1
                    stats["expanded"] += result.expanded
                if result.gates is not None:
                    replacement = place_phases(n, result.gates, angles)
                    assert phase_form(n, replacement) == (target, angles)
                    certificates.append({"wires": wires, "edges": [list(e) for e in edges],
                                         "original": ops, "replacement": replacement,
                                         "before_cx": before, "after_cx": len(result.gates),
                                         "search_states": result.expanded,
                                         "final_rows": list(target),
                                         "phase_angles": {str(m): a for m, a in angles.items()}})
            if replacement is None:
                for inst in block:
                    out.append(inst.operation, inst.qubits, inst.clbits)
            else:
                for op in replacement:
                    q = [wires[i] for i in op["q"]]
                    if op["name"] == "cx":
                        out.cx(*q)
                    else:
                        out.rz(op["angle"], q[0])
            block, support = [], set()

        for inst in circuit.data:
            wires = {circuit.find_bit(q).index for q in inst.qubits}
            if inst.operation.name not in {"cx", "rz"} or len(support | wires) > self.width:
                flush()
            if inst.operation.name in {"cx", "rz"}:
                block.append(inst)
                support |= wires
            else:
                out.append(inst.operation, inst.qubits, inst.clbits)
        flush()
        return out, certificates, stats
