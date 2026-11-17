"""Audit a saved checkout without downloading data, generating SQL, or training.

Only train/validation manifests and existing result receipts are read.
Final-test questions, labels, and databases are deliberately never opened.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tests", required=True)
    parser.add_argument("--observation-date", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    config = {
        "observation_date": args.observation_date,
        "seed": 42,
        "seed_note": "Inherited experiment seed; no random sampling in this audit.",
        "tests": args.tests,
        "output": str(output),
        "operations": ["verify_train_val_manifests", "read_existing_receipts"],
        "model_generation": False,
        "training": False,
        "final_test_content_access": False,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    packages = {}
    for name in ["sqlglot", "pytest"]:
        packages[name] = importlib.metadata.version(name)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "source_commit": git("rev-parse", "HEAD"),
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")

    manifests, rows = {}, {}
    for split, expected in read_json("data/manifests/checksums.json").items():
        path = Path(expected["path"])
        # Fail closed if a future checksums file adds a held-out split.
        if split not in {"train", "val", "train_eligible"}:
            raise ValueError(f"Audit cannot open split: {split}")
        actual = digest(path)
        assert actual == expected["sha256"], f"Changed manifest: {split}"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(records) == expected["questions"]
        assert len({r["id"] for r in records}) == len(records)
        rows[split] = records
        manifests[split] = {
            "path": str(path),
            "sha256": actual,
            "questions": len(records),
            "verified": True,
        }

    dbs = {s: {r["dataset"] + ":" + r["db_id"] for r in rows[s]}
           for s in ["train", "val"]}
    question_hashes = {s: {r["question_sha256"] for r in rows[s]}
                       for s in ["train", "val"]}
    assert not dbs["train"] & dbs["val"], "Database leakage"
    assert not question_hashes["train"] & question_hashes["val"], "Duplicate questions"
    assert {r["id"] for r in rows["train_eligible"]} <= {r["id"] for r in rows["train"]}

    suites = ET.parse(args.tests).getroot()
    counts = {k: sum(int(s.attrib.get(k, 0)) for s in suites.iter("testsuite"))
              for k in ["tests", "failures", "errors", "skipped"]}
    assert counts["tests"] > 0 and counts["failures"] == counts["errors"] == 0
    gold = read_json("results/gold_audit/summary.json")
    data_audit = read_json("results/data_audit.json")
    local_runs = {}
    for root in [Path("results/free_runs"), Path("results/free_runs_batch8_v2")]:
        if root.exists():
            local_runs[str(root)] = sorted(str(p) for p in root.glob("*/status.json"))
    receipt = {
        "observation_date": args.observation_date,
        "source_commit": environment["source_commit"],
        "manifests": manifests,
        "database_overlap_count": len(dbs["train"] & dbs["val"]),
        "train_val_question_hash_overlap_count": len(question_hashes["train"] & question_hashes["val"]),
        "database_counts": {s: len(v) for s, v in dbs.items()},
        "previous_audit_duplicates_removed": data_audit["duplicate_count"],
        "tests": {**counts, "path": args.tests, "sha256": digest(args.tests)},
        "historical_gold_audit": {
            "path": "results/gold_audit/summary.json",
            "n": gold["n"],
            "status_counts": gold["status_counts"],
            "execution_rate": gold["all_queries_execution_rate"],
            "rerun_in_this_audit": False,
        },
        "local_free_run_status_files": local_runs,
        "last_saved_colab_progress": {
            "observed_utc": "2026-11-07T10:25:57.164250+00:00",
            "source": "docs/RESUME_PHASE03.md",
            "base_zero_predictions": 720,
            "validation_questions": 2304,
            "current_runtime_verified": False,
            "current_private_backup_verified": False,
        },
        "selected_model_record": read_json("results/model_selection.json"),
        "pending": [
            "Current model choice required by the new request before GPU work.",
            "Verify newest private backup and current free GPU availability.",
            "Complete base zero-shot, few-shot, hosted baseline, SFT and SFT validation.",
            "Final manifest checksums and cross-final duplicate audit remain Phase 9 work.",
        ],
        "phase3_complete": False,
        "new_accuracy_result": False,
        "final_test_content_access": False,
    }
    (output / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({
        "tests": counts, "manifests_verified": list(manifests),
        "database_overlap": 0, "question_overlap": 0,
        "phase3_complete": False, "final_test_content_access": False,
    }, indent=2))


if __name__ == "__main__":
    main()
