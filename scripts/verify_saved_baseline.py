"""Recompute the imported validation metrics without calling a model or SQLite.

Run from the repository root. The import manifest pins every exported file.
The optional validation file must match the exact bytes used by the saved run.
This check never downloads or reads the final test set.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from sqlrl.common import sha256, write_json
from sqlrl.eval.statistics import summarize


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def verify(validation=None):
    manifest = read_json("results/colab_import_20261118/import_manifest.json")
    for entry in manifest["files"].values():
        assert sha256(entry["path"]) == entry["sha256"], entry["path"]
    folder = Path("results/free_runs_batch8_v2")
    config = read_json(folder / "base_zero_evaluation/config.json")
    assert sha256(config["predictions"]) == config["predictions_sha256"]
    if validation:
        assert sha256(validation) == config["data_sha256"]

    predictions = read_rows(config["predictions"])
    rows = read_rows(folder / "base_zero_evaluation/per_question.jsonl")
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids)) == 2304
    assert ids == [row["id"] for row in predictions]
    saved = read_json(folder / "base_zero_evaluation/summary.json")
    recomputed = summarize(rows, seed=saved["bootstrap_seed"], samples=saved["bootstrap_samples"])
    assert recomputed == saved, "Saved metrics or bootstrap intervals do not reproduce"
    by_dataset = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        subset = [row for row in rows if row["dataset"] == dataset]
        by_dataset[dataset] = {
            "questions": len(subset), "databases": len({row["db_id"] for row in subset}),
            "correct": sum(row["correct"] for row in subset),
            "valid": sum(row["valid"] for row in subset),
            "statuses": dict(Counter(row["status"] for row in subset)),
        }
    return {
        "passed": True, "export_files_verified": len(manifest["files"]),
        "validation_bytes_checked": bool(validation), "validation_sha256": config["data_sha256"],
        "questions": len(rows), "correct": sum(row["correct"] for row in rows),
        "formatted": sum(row["formatted"] for row in rows),
        "summary_and_bootstrap_reproduced": True, "summary": recomputed,
        "by_dataset": by_dataset, "final_test_accessed": False,
        "model_called": False, "sql_executed": False,
        "cost_per_accuracy_point_available": False,
        "latency_caveat": "Generation latency is shared batch wall time, not isolated request latency.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", help="Optional processed VAL file; never pass final-test data")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify(args.validation)
    write_json(args.output, result)
    print(json.dumps({"passed": result["passed"], "questions": result["questions"],
                      "correct": result["correct"], "metrics_reproduced": True}))
