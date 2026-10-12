"""Independent exhaustive bit-permutation checks of remote-CX actions."""
from __future__ import annotations
import json
from pathlib import Path
import sys


def execute(bits, pairs, positions):
    for c, t in pairs:
        bits ^= ((bits >> positions[c]) & 1) << positions[t]
    return bits


def check(cert, hardware_edges):
    path = cert["path"]
    positions = {p: i for i, p in enumerate(path)}
    assert len(path) == len(positions) <= 12
    pairs = [tuple(p) for p in cert["physical_cx"]]
    assert all(p in hardware_edges for p in pairs)
    control = cert["before"][cert["control"]]
    target = cert["before"][cert["target"]]
    moved = list(cert["before"])
    for a, b in cert["swaps"]:
        moved = [b if p == a else a if p == b else p for p in moved]
    assert moved == cert["after"]
    negatives = 0
    for x in range(1 << len(path)):
        expected = execute(x, [(control, target)], positions)
        for a, b in cert["swaps"]:
            ia, ib = positions[a], positions[b]
            if ((expected >> ia) ^ (expected >> ib)) & 1:
                expected ^= (1 << ia) | (1 << ib)
        actual = execute(x, pairs, positions)
        assert actual == expected
        negatives += execute(x, pairs[:-1], positions) != expected
    assert negatives > 0
    return {"basis_inputs": 1 << len(path), "negative_mismatches": negatives, "kind": cert["kind"]}


if __name__ == "__main__":
    data = json.loads(Path(sys.argv[1]).read_text())
    edges = {tuple(e) for e in data["edges"]}
    checks = [check(c, edges) for row in data["rows"] for c in row.get("certificates", [])]
    print(json.dumps({"actions": len(checks), "basis_inputs": sum(c["basis_inputs"] for c in checks),
                      "bridge_actions": sum(c["kind"] == "bridge" for c in checks),
                      "negative_controls_detected": len(checks)}, indent=2))
