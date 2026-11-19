"""Replay a complete saved VAL run's metrics without models, SQL or final data."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from sqlrl.common import sha256, write_json
from sqlrl.eval.statistics import summarize


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def verify(run, evaluation, *, verify_val_bytes=False):
    run, evaluation = Path(run), Path(evaluation)
    config = read_json(run / "config.json")
    eval_config = read_json(evaluation / "config.json")
    status = read_json(run / "status.json")
    if config["data"] != "data/processed/val.jsonl" or eval_config["data"] != config["data"]:
        raise ValueError("Only the fixed project VAL cohort is supported")
    expected_hash = config["data_sha256"]
    if eval_config["data_sha256"] != expected_hash:
        raise ValueError("Generation and evaluation used different validation bytes")
    raw_path, gzip_path = run / "predictions.jsonl", run / "predictions.jsonl.gz"
    if raw_path.exists() == gzip_path.exists():
        raise ValueError("Keep exactly one plain or gzip prediction artifact")
    prediction_path = raw_path if raw_path.exists() else gzip_path
    raw_predictions = prediction_path.read_bytes()
    if prediction_path == gzip_path:
        raw_predictions = gzip.decompress(raw_predictions)
    if hashlib.sha256(raw_predictions).hexdigest() != eval_config["predictions_sha256"]:
        raise ValueError("Prediction bytes differ from the evaluation receipt")

    expected = read_json("data/manifests/checksums.json")["val"]
    if sha256(expected["path"]) != expected["sha256"]:
        raise ValueError("Validation manifest changed")
    manifest = read_rows(expected["path"])
    predictions = [json.loads(line) for line in raw_predictions.splitlines() if line.strip()]
    rows = read_rows(evaluation / "per_question.jsonl")
    ids = [r["id"] for r in manifest]
    if status["status"] != "completed" or len(ids) != expected["questions"]:
        raise ValueError("A completed full-cohort VAL run is required")
    if [p["id"] for p in predictions] != ids or [r["id"] for r in rows] != ids:
        raise ValueError("Predictions and scores must match the complete ordered VAL cohort")
    if [(r["dataset"], r["db_id"]) for r in rows] != [(r["dataset"], r["db_id"]) for r in manifest]:
        raise ValueError("Saved score database identities differ from VAL")
    if verify_val_bytes and sha256("data/processed/val.jsonl") != expected_hash:
        raise ValueError("Processed validation bytes changed")

    saved = read_json(evaluation / "summary.json")
    recomputed = summarize(rows, seed=saved["bootstrap_seed"], samples=saved["bootstrap_samples"])
    if recomputed != saved:
        raise ValueError("Saved metrics or bootstrap intervals do not reproduce")
    files = [run / "config.json", run / "status.json", prediction_path,
             evaluation / "config.json", evaluation / "per_question.jsonl", evaluation / "summary.json"]
    return {
        "passed": True, "questions": len(ids), "correct": sum(r["correct"] for r in rows),
        "model": config["model"], "revision": config.get("revision"), "seed": config.get("seed"),
        "data_sha256": expected_hash, "validation_bytes_checked": verify_val_bytes,
        "prediction_compression": "gzip" if prediction_path == gzip_path else "none",
        "source_hashes": {str(path): sha256(path) for path in files},
        "summary_and_bootstrap_reproduced": True, "summary": recomputed,
        "model_called": False, "sql_executed": False, "final_test_accessed": False,
        "scope": "Replays saved execution records and their arithmetic; does not re-execute SQL.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--verify-val-bytes", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("Preserve the previous replay receipt")
    report = verify(args.run, args.evaluation, verify_val_bytes=args.verify_val_bytes)
    write_json(args.output, report)
    print(json.dumps({key: report[key] for key in ["passed", "questions", "correct", "final_test_accessed"]}))


if __name__ == "__main__":
    main()
