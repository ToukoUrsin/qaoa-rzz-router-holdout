"""Independent exhaustive computational-basis check of each replacement.

No imports from the synthesis implementation. Checks exact output bit strings
and complex amplitudes on all 2^n inputs, with one shared global phase.
"""
from __future__ import annotations
import cmath
import json
from pathlib import Path
import sys


def table(n: int, ops: list[dict]) -> list[tuple[int, complex]]:
    output = []
    for initial in range(1 << n):
        bits = initial
        phase = 1 + 0j
        for op in ops:
            if op["name"] == "cx":
                c, t = op["q"]
                bits ^= ((bits >> c) & 1) << t
            elif op["name"] == "rz":
                z = 1 - 2 * ((bits >> op["q"][0]) & 1)
                phase *= cmath.exp(-0.5j * op["angle"] * z)
            else:
                raise ValueError(op)
        output.append((bits, phase))
    return output


def error(a, b) -> float:
    if any(x[0] != y[0] for x, y in zip(a, b)):
        return 2.0
    phase = a[0][1] / b[0][1]
    return max(abs(x[1] - phase * y[1]) for x, y in zip(a, b))


def check(cert: dict) -> dict:
    n = len(cert["wires"])
    edges = {tuple(e) for e in cert["edges"]}
    for op in cert["replacement"]:
        if op["name"] == "cx" and tuple(op["q"]) not in edges:
            raise AssertionError("nonphysical CX")
    a = table(n, cert["original"])
    b = table(n, cert["replacement"])
    positive = error(a, b)
    corrupt = cert["replacement"] + [{"name": "rz", "q": [0], "angle": 0.031}]
    negative = error(a, table(n, corrupt))
    before = sum(o["name"] == "cx" for o in cert["original"])
    after = sum(o["name"] == "cx" for o in cert["replacement"])
    assert before == cert["before_cx"] and after == cert["after_cx"] and after < before
    assert positive < 1e-8 and negative > 1e-5
    return {"positive_error": positive, "negative_error": negative, "inputs": 1 << n}


def audit(path: Path) -> dict:
    data = json.loads(path.read_text())
    checks = [check(c) for row in data["rows"] for c in row.get("certificates", [])]
    return {"file": str(path), "replacements": len(checks),
            "positive_max_error": max((r["positive_error"] for r in checks), default=None),
            "negative_min_error": min((r["negative_error"] for r in checks), default=None),
            "exhaustive_basis_inputs": sum(r["inputs"] for r in checks)}


if __name__ == "__main__":
    result = [audit(Path(p)) for p in sys.argv[1:]]
    print(json.dumps(result, indent=2))
