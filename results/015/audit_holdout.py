"""Standard-library objective, input, hash and checker audit; no solver imports."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    inputs = {}
    findings = {}
    manifests = []
    versions = {"qiskit": "2.5.2", "pytket": "2.18.5", "pytket-qiskit": "0.78.0", "mqt.bench": "2.3.0"}
    for name in ("grid3", "heavy5"):
        path = ROOT / "results/015" / f"{name}_holdout.json"
        inputs[str(path.relative_to(ROOT))] = sha(path)
        report = json.loads(path.read_text())
        assert report["map"] == name
        assert all(report["versions"][key] == value for key, value in versions.items())
        assert report["parameters_frozen_before_holdout"] == {"beam": 64, "state_cap": 30000, "region_cap": 12}
        for source, expected in report["source_code_sha256"].items():
            assert sha(ROOT / source) == expected, source
        manifest = report["source"]
        assert manifest["widths"] == [5, 6, 7, 8, 9]
        assert manifest["per_width"] == 20 and manifest["start_seed"] == 2000
        assert len(manifest["rows"]) == len(report["rows"]) == 100
        assert len({row["case"] for row in report["rows"]}) == 100
        assert len({row["input_fingerprint"] for row in report["rows"]}) == 100
        for width in manifest["widths"]:
            assert sum(row["width"] == width for row in report["rows"]) == 20
        assert [{k: row[k] for k in ("case", "width", "seed", "input_fingerprint")}
                for row in report["rows"]] == manifest["rows"]
        manifests.append(manifest)
        valid = [row for row in report["rows"] if row["status"] == "ok"]
        incumbent, raw, selected = [], [], []
        negative_controls = layers = basis_inputs = fusions = 0
        max_replay_error = 0
        for row in valid:
            assert len(row["qiskit_counts"]) == 8
            assert row["qiskit_best"] == min(row["qiskit_counts"])
            assert row["qiskit_counts"][row["best_seed"]] == row["qiskit_best"]
            assert row["incumbent"] == min(row["qiskit_best"], row["tket_cx"])
            assert row["selected_cx"] == min(row["incumbent"], row["candidate_cx"])
            expected_route = ("candidate" if row["candidate_cx"] < row["incumbent"] else
                              ("qiskit" if row["qiskit_best"] <= row["tket_cx"] else "tket"))
            assert row["selected_route"] == expected_route
            assert sha(ROOT / row["artifact"]) == row["artifact_sha256"]
            incumbent.append(row["incumbent"])
            raw.append(row["candidate_cx"])
            selected.append(row["selected_cx"])
            assert row["grouping_check"]["negative_control_detected"]
            negative_controls += 1
            assert len(row["certificates"]) == len(row["certificate_checks"])
            for cert, check in zip(row["certificates"], row["certificate_checks"]):
                assert check["basis_inputs"] == 1 << len(cert["region"])
                assert check["max_error"] < 1e-8 and check["negative_error"] > 1e-5
                assert cert["beam"] == 64 and cert["search_states"] <= 30000
                layers += 1
                basis_inputs += check["basis_inputs"]
                negative_controls += 1
            for check in row["fusion_checks"]:
                assert check["full_operator_error"] < 1e-8 and check["negative_error"] > 1e-5 and check["cx"] == 3
                fusions += 1
                negative_controls += 1
            for key in ("qiskit_check", "tket_check", "candidate_check"):
                check = row[key]
                assert check["status"] == "checked" and check["shared_global_phase"]
                assert check["max_norm_error"] < 1e-7 and check["random_probe_negative_error"] > 1e-5
                max_replay_error = max(max_replay_error, check["max_norm_error"])
                negative_controls += 1
        assert valid, "no valid rows"
        median = statistics.median(incumbent)
        score = {"attempted": 100, "valid": len(valid), "failures": 100 - len(valid),
                 "incumbent_median": median, "raw_candidate_median": statistics.median(raw),
                 "selected_median": statistics.median(selected),
                 "raw_ratio_gain": 1 - statistics.median(raw) / median,
                 "selected_ratio_gain": 1 - statistics.median(selected) / median,
                 "wins": sum(c < i for c, i in zip(raw, incumbent)),
                 "ties": sum(c == i for c, i in zip(raw, incumbent)),
                 "raw_losses": sum(c > i for c, i in zip(raw, incumbent)),
                 "selected_losses": sum(c > i for c, i in zip(selected, incumbent)),
                 "original_input_checks": len(valid) * 3}
        assert score == report["summary"], (score, report["summary"])
        findings[name] = {"score": score, "layer_certificates": layers,
                          "exhaustive_layer_basis_inputs": basis_inputs,
                          "full_operator_fusion_checks": fusions,
                          "recorded_negative_controls": negative_controls,
                          "max_original_replay_error": max_replay_error,
                          "failures": [{k: row.get(k) for k in ("case", "status", "error")}
                                       for row in report["rows"] if row["status"] != "ok"]}
    assert manifests[0] == manifests[1], "map inputs differ"
    smoke = json.loads((ROOT / "results/015/smoke/grid3.json").read_text())
    assert not ({r["input_fingerprint"] for r in smoke["rows"]} &
                {r["input_fingerprint"] for r in manifests[0]["rows"]}), "smoke overlap"
    qcec_summary = None
    if args.qcec:
        data = json.loads(args.qcec.read_text())
        assert data["qcec_version"] == "3.11.0"
        assert data["checker_source_sha256"] == sha(ROOT / "results/015/verify_qcec.py")
        assert all(inputs[path] == fingerprint for path, fingerprint in data["inputs"].items())
        expected = {(name, row["case"]): row for name in ("grid3", "heavy5")
                    for row in json.loads((ROOT / "results/015" / f"{name}_holdout.json").read_text())["rows"]
                    if row["status"] == "ok"}
        assert len(data["rows"]) == len(expected)
        assert {(row["map"], row["case"]) for row in data["rows"]} == set(expected)
        checked = 0
        for row in data["rows"]:
            original = expected[row["map"], row["case"]]
            assert row["artifact_sha256"] == original["artifact_sha256"]
            if row["status"] == "checked":
                assert str(row["positive"]["equivalence"]).endswith("equivalent") or str(row["positive"]["equivalence"]).endswith("equivalent_up_to_global_phase")
                assert str(row["negative"]["equivalence"]).endswith("not_equivalent")
                checked += 1
        qcec_summary = {"attempted": len(expected), "checked": checked, "failures": len(expected) - checked}
        assert qcec_summary == data["summary"]
        inputs[str(args.qcec.resolve().relative_to(ROOT))] = sha(args.qcec)
    result = {"scope": "100 distinct MQT QAOA graphs; family result only, mixed charter remains unmet",
              "evidence_level": "independently audited objectives/hashes/checker records; QPY counts and certificates bound by separate QCEC validator",
              "inputs": inputs, "versions": versions, "maps": findings, "qcec": qcec_summary,
              "submitted": False, "accepted": False}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qcec", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
