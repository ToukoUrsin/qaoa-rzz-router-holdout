"""Joint ordering of commuting RZZ interactions and physical wire moves.

Macro costs are 2 CX for RZZ, 3 CX for SWAP, and 3 CX for RZZ followed
by SWAP (fixed-wire KAK synthesis). Search is bounded beam search; it does
not assert optimality. Layer certificates bind the original interaction set
and every physical permutation, rather than treating routing as a free relabel.
"""
from __future__ import annotations
from dataclasses import dataclass
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import Operator
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CommutativeCancellation
from bridge_router import BridgeRouter


@dataclass
class State:
    placement: tuple[int, ...]
    remaining: int
    cost: int
    actions: tuple[dict, ...]


class IsingRouter:
    def __init__(self, edges, width, beam=64, state_cap=30_000):
        self.edges = set(edges)
        self.width = width
        self.beam = beam
        self.state_cap = state_cap
        self.graph = BridgeRouter(edges, width)
        self.fused = {}
        self.oneq = {}
        self.decomposer = TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX", pulse_optimize=False)

    def fusion(self, angle):
        if angle not in self.fused:
            ideal = QuantumCircuit(2)
            ideal.rzz(angle, 0, 1)
            ideal.swap(0, 1)
            native = self.decomposer(Operator(ideal).data)
            assert native.count_ops().get("cx", 0) == 3
            self.fused[angle] = native
        return self.fused[angle]

    def layer(self, interactions, initial, continued):
        count = len(interactions)
        initial = tuple(initial)
        region = set(initial)
        for a in initial:
            for b in initial:
                region.update(self.graph.shortest_paths(a, b)[0])
        assert len(region) <= 12, f"region {len(region)} exceeds certificate cap"
        bonds = sorted((a, b) for a, b in self.edges if a < b and a in region and b in region)
        all_remaining = (1 << count) - 1
        start = State(initial, all_remaining, 0, ())

        def swap(p, a, b):
            return self.graph.swapped(p, a, b)

        def act(state, kind, pair, interaction=None):
            remaining = state.remaining if interaction is None else state.remaining & ~(1 << interaction)
            placement = state.placement if kind == "rzz" else swap(state.placement, *pair)
            action = {"kind": kind, "physical": list(pair)}
            if interaction is not None:
                action["interaction"] = interaction
            return State(placement, remaining, state.cost + (2 if kind == "rzz" else 3),
                         state.actions + (action,))

        # A feasible upper bound uses shortest paths, so search caps never
        # manufacture a timeout score or an infeasible zero-count candidate.
        fallback = start
        for interaction, (a, b, angle) in enumerate(interactions):
            path = self.graph.shortest_paths(fallback.placement[a], fallback.placement[b])[0]
            for u, v in zip(path[:-2], path[1:-1]):
                fallback = act(fallback, "swap", (u, v))
            fallback = act(fallback, "rzz", (fallback.placement[a], fallback.placement[b]), interaction)
        goals = [fallback]
        frontier = [start]
        seen = {(initial, all_remaining): 0}
        rounds = 0
        capped = False

        def distance_sum(p, remaining):
            return sum(max(0, self.graph.distances[p[a]][p[b]] - 1)
                       for i, (a, b, _) in enumerate(interactions) if remaining & (1 << i))

        def score(state):
            return state.cost + 2 * state.remaining.bit_count() + 1.5 * distance_sum(state.placement, state.remaining)

        for rounds in range(1, 2 * count + 2 * len(initial) + 1):
            expanded = []
            for state in frontier:
                if state.cost + 2 * state.remaining.bit_count() > min(g.cost for g in goals):
                    continue
                eligible = {}
                for i, (a, b, _) in enumerate(interactions):
                    if state.remaining & (1 << i):
                        pair = (state.placement[a], state.placement[b])
                        if pair in self.edges:
                            eligible[tuple(sorted(pair))] = i
                            expanded.append(act(state, "rzz", pair, i))
                            expanded.append(act(state, "rzz_swap", pair, i))
                occupied = set(state.placement)
                for pair in bonds:
                    if (pair[0] in occupied or pair[1] in occupied) and pair not in eligible:
                        expanded.append(act(state, "swap", pair))
            next_frontier = {}
            for state in expanded:
                if not state.remaining:
                    goals.append(state)
                    continue
                key = (state.placement, state.remaining)
                if key in seen and seen[key] <= state.cost:
                    continue
                if len(seen) >= self.state_cap:
                    capped = True
                    continue
                seen[key] = state.cost
                next_frontier[key] = state
            frontier = sorted(next_frontier.values(), key=lambda s: (score(s), s.cost, s.placement))[:self.beam]
            if not frontier:
                break
        goal = min(goals, key=lambda s: (s.cost + (0.3 * distance_sum(s.placement, all_remaining) if continued else 0), s.cost))
        certificate = {"region": sorted(region), "initial": list(initial), "final": list(goal.placement),
                       "interactions": [list(x) for x in interactions], "actions": list(goal.actions),
                       "planned_cx": goal.cost, "search_states": len(seen), "state_cap_hit": capped,
                       "rounds": rounds, "beam": self.beam}
        return goal.placement, certificate

    def route(self, source, initial):
        out = QuantumCircuit(self.width)
        out.global_phase = source.global_phase
        placement = tuple(initial)
        certificates = []
        pending = []

        def flush(continued):
            nonlocal placement, pending
            if not pending:
                return
            placement, certificate = self.layer(pending, placement, continued)
            certificates.append(certificate)
            for action in certificate["actions"]:
                a, b = action["physical"]
                if action["kind"] == "swap":
                    out.cx(a, b)
                    out.cx(b, a)
                    out.cx(a, b)
                else:
                    angle = pending[action["interaction"]][2]
                    if action["kind"] == "rzz_swap":
                        out.compose(self.fusion(angle), qubits=[a, b], inplace=True)
                    else:
                        out.cx(a, b)
                        out.rz(angle, b)
                        out.cx(a, b)
            pending = []

        data = list(source.data)
        for index, instruction in enumerate(data):
            logical = [source.find_bit(q).index for q in instruction.qubits]
            op = instruction.operation
            if op.name == "rzz":
                pending.append((logical[0], logical[1], float(op.params[0])))
            else:
                flush(any(i.operation.name == "rzz" for i in data[index + 1:]))
                assert len(logical) == 1, f"unsupported operation {op.name}"
                key = (op.name, tuple(float(x) for x in op.params))
                if key not in self.oneq:
                    q = QuantumCircuit(1)
                    q.append(op, [0])
                    self.oneq[key] = transpile(q, basis_gates=["rz", "sx", "x", "cx"],
                                               optimization_level=0, qubits_initially_zero=False)
                out.compose(self.oneq[key], qubits=[placement[logical[0]]], inplace=True)
        flush(False)
        out = PassManager([CommutativeCancellation()]).run(out)
        out.metadata = {"logical_initial": list(initial), "logical_final": list(placement)}
        return out, certificates, self.fused


def normalize_qaoa(source):
    """Recover cost layers from an equivalent interleaved DAG listing.

    This guarded normalizer accepts only H initialization, RZZ cost gates,
    and RX mixers with equal mixer epochs at each interaction endpoint.
    A per-wire event-order certificate proves the regrouping is a legal
    topological sort, before commuting interactions are optimized.
    """
    epochs = [0] * source.num_qubits
    preparation = []
    phases = {}
    mixers = {}
    events = []
    for index, instruction in enumerate(source.data):
        q = [source.find_bit(b).index for b in instruction.qubits]
        name = instruction.operation.name
        events.append({"id": index, "name": name, "q": q,
                       "params": [float(x) for x in instruction.operation.params]})
        if name == "h":
            assert epochs[q[0]] == 0
            preparation.append(index)
        elif name == "rx":
            mixers.setdefault(epochs[q[0]], []).append(index)
            epochs[q[0]] += 1
        elif name == "rzz":
            assert epochs[q[0]] == epochs[q[1]], "not a QAOA layer structure"
            phases.setdefault(epochs[q[0]], []).append(index)
        else:
            raise ValueError(f"QAOA normalizer does not accept {name}")
    order = preparation[:]
    for epoch in sorted(set(phases) | set(mixers)):
        order.extend(phases.get(epoch, []))
        order.extend(mixers.get(epoch, []))
    assert sorted(order) == list(range(len(events)))
    for wire in range(source.num_qubits):
        before = [e["id"] for e in events if wire in e["q"]]
        after = [i for i in order if wire in events[i]["q"]]
        assert before == after, f"wire {wire} dependency order changed"
    normalized = source.copy_empty_like()
    for index in order:
        instruction = source.data[index]
        normalized.append(instruction.operation, instruction.qubits, instruction.clbits)
    certificate = {"events": events, "normalized_order": order,
                   "wires": source.num_qubits, "cost_layers": len(phases)}
    return normalized, certificate
