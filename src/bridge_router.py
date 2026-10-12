"""Bounded routing with dirty-wire CX bridges and explicit permutations.

Input is an already decomposed logical circuit. Each nonlocal CX can use a
SWAP route or a layout-preserving remote-CX ladder. Beam search is heuristic;
no optimal routing claim is made. All CX operations are physical edges.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CommutativeCancellation


@dataclass(frozen=True)
class Node:
    previous: object
    op: object
    q: tuple[int, ...]


@dataclass
class State:
    placement: tuple[int, ...]
    tail: Node | None
    cost: int
    certificates: tuple[dict, ...]


def bridge_ladder(path):
    d = len(path) - 1
    assert d >= 2
    pairs = [(path[i], path[i + 1]) for i in range(d)]
    return pairs + list(reversed(pairs[:-1])) + pairs[1:] + list(reversed(pairs[1:-1]))


class BridgeRouter:
    def __init__(self, physical_edges, physical_width, beam=8, bridges=True):
        self.edges = set(physical_edges)
        self.width = physical_width
        self.beam = beam
        self.bridges = bridges
        self.adj = {i: [] for i in range(physical_width)}
        for a, b in sorted(self.edges):
            self.adj[a].append(b)
        self.distances = []
        for a in range(physical_width):
            distances = [physical_width + 1] * physical_width
            distances[a] = 0
            queue = deque([a])
            while queue:
                u = queue.popleft()
                for v in self.adj[u]:
                    if distances[v] > distances[u] + 1:
                        distances[v] = distances[u] + 1
                        queue.append(v)
            self.distances.append(distances)
        self.paths = {}
        c = QuantumCircuit(2)
        c.cx(0, 1)
        self.cx = c.data[0].operation

    def shortest_paths(self, a, b):
        if (a, b) in self.paths:
            return self.paths[a, b]
        paths = []

        def visit(path):
            if len(paths) == 2:
                return
            u = path[-1]
            if u == b:
                paths.append(path)
                return
            for v in self.adj[u]:
                if self.distances[v][b] == self.distances[u][b] - 1:
                    visit(path + [v])

        visit([a])
        self.paths[a, b] = paths
        return paths

    def append(self, state, op, q):
        # Exact adjacent inverse-CX cancellation; count the emitted circuit.
        if op.name == "cx" and state.tail and state.tail.op.name == "cx" and state.tail.q == q:
            state.tail = state.tail.previous
            state.cost -= 1
        else:
            state.tail = Node(state.tail, op, q)
            state.cost += int(op.name == "cx")

    @staticmethod
    def swapped(placement, a, b):
        return tuple(b if p == a else a if p == b else p for p in placement)

    def options(self, state, control, target, instruction_index):
        a, b = state.placement[control], state.placement[target]
        if (a, b) in self.edges:
            result = State(state.placement, state.tail, state.cost, state.certificates)
            self.append(result, self.cx, (a, b))
            return [result]
        results = []
        for path in self.shortest_paths(a, b):
            d = len(path) - 1
            for split in sorted({0, d // 2, d - 1}):
                swaps = [(path[i], path[i + 1]) for i in range(split)]
                swaps += [(path[i], path[i - 1]) for i in range(d, split + 1, -1)]
                placement = state.placement
                ops = []
                for u, v in swaps:
                    placement = self.swapped(placement, u, v)
                    ops += [(u, v), (v, u), (u, v)]
                ops.append((placement[control], placement[target]))
                cert = {"instruction": instruction_index, "kind": "swap_route", "path": path,
                        "control": control, "target": target, "before": list(state.placement),
                        "after": list(placement), "swaps": [list(s) for s in swaps],
                        "physical_cx": [list(p) for p in ops]}
                result = State(placement, state.tail, state.cost, state.certificates + (cert,))
                for pair in ops:
                    self.append(result, self.cx, pair)
                results.append(result)
            if self.bridges:
                pairs = bridge_ladder(path)
                for ops in (pairs, list(reversed(pairs))):
                    cert = {"instruction": instruction_index, "kind": "bridge", "path": path,
                            "control": control, "target": target, "before": list(state.placement),
                            "after": list(state.placement), "swaps": [],
                            "physical_cx": [list(p) for p in ops]}
                    result = State(state.placement, state.tail, state.cost, state.certificates + (cert,))
                    for pair in ops:
                        self.append(result, self.cx, pair)
                    results.append(result)
        return results

    def route(self, circuit, initial):
        assert len(initial) == circuit.num_qubits and len(set(initial)) == len(initial)
        instructions = [(i.operation, tuple(circuit.find_bit(q).index for q in i.qubits)) for i in circuit.data]
        upcoming = []
        lookaheads = [None] * len(instructions)
        for index in range(len(instructions) - 1, -1, -1):
            lookaheads[index] = tuple(upcoming)
            op, logical = instructions[index]
            if op.name == "cx":
                upcoming = [logical] + upcoming[:5]
        states = [State(tuple(initial), None, 0, ())]
        max_frontier = 1
        for index, (op, logical) in enumerate(instructions):
            if op.name == "cx":
                expanded = [x for state in states for x in self.options(state, *logical, index)]
                next_cx = lookaheads[index]

                def score(state):
                    lookahead = sum(max(0, self.distances[state.placement[a]][state.placement[b]] - 1)
                                    for a, b in next_cx)
                    return state.cost + 1.5 * lookahead

                best = {}
                for state in expanded:
                    key = state.placement
                    if key not in best or state.cost < best[key].cost:
                        best[key] = state
                states = sorted(best.values(), key=lambda s: (score(s), s.cost, s.placement))[:self.beam]
                max_frontier = max(max_frontier, len(expanded))
            elif len(logical) == 1:
                for state in states:
                    self.append(state, op, (state.placement[logical[0]],))
            else:
                raise ValueError(f"unsupported logical operation {op.name}")
        candidates = []
        for state in states:
            output = QuantumCircuit(self.width)
            output.global_phase = circuit.global_phase
            nodes = []
            node = state.tail
            while node:
                nodes.append(node)
                node = node.previous
            for node in reversed(nodes):
                output.append(node.op, node.q)
            output = PassManager([CommutativeCancellation()]).run(output)
            assert all(len(i.qubits) == 1 or tuple(output.find_bit(q).index for q in i.qubits) in self.edges
                       for i in output.data)
            count = int(output.count_ops().get("cx", 0))
            output.metadata = {"logical_initial": list(initial), "logical_final": list(state.placement)}
            candidates.append((count, output, state.certificates))
        count, output, certificates = min(candidates, key=lambda x: x[0])
        return output, list(certificates), {"max_frontier": max_frontier, "beam": self.beam,
                                          "bridges_enabled": self.bridges,
                                          "selected_bridges": sum(c["kind"] == "bridge" for c in certificates),
                                          "cx": count}
