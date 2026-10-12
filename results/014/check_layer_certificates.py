"""Exhaustive diagonal-phase and permutation certificates, without search imports."""
from __future__ import annotations
import cmath


def check_regrouping(cert):
    events = cert["events"]
    order = cert["normalized_order"]
    assert sorted(order) == list(range(len(events)))
    for wire in range(cert["wires"]):
        before = [e["id"] for e in events if wire in e["q"]]
        after = [i for i in order if wire in events[i]["q"]]
        assert before == after
    # Negative control crosses a shared-wire RX/RZZ boundary.
    negative = order[:]
    pair = next((i, j) for i in range(len(order)) for j in range(i + 1, len(order))
                if {events[order[i]]["name"], events[order[j]]["name"]} == {"rx", "rzz"}
                and set(events[order[i]]["q"]) & set(events[order[j]]["q"]))
    negative[pair[0]], negative[pair[1]] = negative[pair[1]], negative[pair[0]]
    detected = any([e["id"] for e in events if w in e["q"]] !=
                   [i for i in negative if w in events[i]["q"]] for w in range(cert["wires"]))
    assert detected
    return {"events": len(events), "wires": cert["wires"], "cost_layers": cert["cost_layers"],
            "negative_control_detected": detected}


def check_layer(cert, edges):
    region = cert["region"]
    positions = {p: i for i, p in enumerate(region)}
    assert len(region) <= 12
    initial = cert["initial"]
    interactions = cert["interactions"]
    placement = list(initial)
    consumed = []
    for action in cert["actions"]:
        a, b = action["physical"]
        assert (a, b) in edges
        if action["kind"] != "swap":
            i = action["interaction"]
            u, v, _ = interactions[i]
            assert {placement[u], placement[v]} == {a, b}
            consumed.append(i)
        if action["kind"] != "rzz":
            placement = [b if p == a else a if p == b else p for p in placement]
    assert placement == cert["final"]
    assert sorted(consumed) == list(range(len(interactions)))
    max_error = 0
    negatives = []
    for initial_bits in range(1 << len(region)):
        expected_phase = 1 + 0j
        for u, v, theta in interactions:
            parity = ((initial_bits >> positions[initial[u]]) ^ (initial_bits >> positions[initial[v]])) & 1
            expected_phase *= cmath.exp(-0.5j * theta * (1 - 2 * parity))
        bits = initial_bits
        phase = 1 + 0j
        expected_bits = initial_bits
        negative_phase = 1 + 0j
        corrupted = False
        for action in cert["actions"]:
            a, b = map(positions.get, action["physical"])
            if action["kind"] != "swap":
                theta = interactions[action["interaction"]][2]
                parity = ((bits >> a) ^ (bits >> b)) & 1
                phase *= cmath.exp(-0.5j * theta * (1 - 2 * parity))
                delta = 0.031 if not corrupted else 0
                negative_phase *= cmath.exp(-0.5j * (theta + delta) * (1 - 2 * parity))
                corrupted = True
            if action["kind"] != "rzz":
                if ((bits >> a) ^ (bits >> b)) & 1:
                    bits ^= (1 << a) | (1 << b)
                if ((expected_bits >> a) ^ (expected_bits >> b)) & 1:
                    expected_bits ^= (1 << a) | (1 << b)
        assert bits == expected_bits
        max_error = max(max_error, abs(phase - expected_phase))
        negatives.append(negative_phase / expected_phase)
    assert max_error < 1e-8
    negative_error = max(abs(x - negatives[0]) for x in negatives)
    assert negative_error > 1e-5
    return {"basis_inputs": 1 << len(region), "max_error": max_error, "negative_error": negative_error,
            "fused_rzz_swaps": sum(a["kind"] == "rzz_swap" for a in cert["actions"])}
