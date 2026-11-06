"""Evaluate saved predictions. No model or API dependency in the verifier."""
import argparse
import json
import time
from pathlib import Path

from sqlrl.common import sha256, start_run, write_json, write_jsonl
from sqlrl.data.guard import load_records
from sqlrl.eval import extract_sql, summarize
from sqlrl.executor import Limits, execute, matches, ordered_gold


def evaluate(rows, predictions, limits=Limits()):
    by_id = {p["id"]: p for p in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("Duplicate prediction IDs")
    if set(by_id) != {r["id"] for r in rows}:
        raise ValueError("Predictions must cover exactly the evaluated split, including failures")
    results = []
    for row in rows:
        pred = by_id[row["id"]]
        begin = time.monotonic()
        sql, formatted = extract_sql(pred.get("answer", ""))
        gold = execute(row["db_path"], row["sql"], limits)
        candidate = execute(row["db_path"], sql, limits) if sql is not None else None
        status = candidate.status if candidate else pred.get("error", "format_error")
        correct, error = False, None
        if gold.status != "ok":
            status, error = "gold_error", gold.status
        elif candidate and candidate.status == "ok":
            try:
                correct = matches(gold.rows, candidate.rows, ordered=ordered_gold(row["sql"]), gold_columns=gold.columns, pred_columns=candidate.columns)
            except ValueError as exception:
                status, error = "comparison_limit", str(exception)
        results.append({"id":row["id"], "dataset":row["dataset"], "db_id":row["db_id"],
            "difficulty":row.get("difficulty", "unknown"), "status":status, "correct":correct,
            "valid":candidate is not None and candidate.status == "ok", "formatted":formatted,
            "input_tokens":pred.get("input_tokens"), "output_tokens":pred.get("output_tokens"),
            "cost_usd":pred.get("cost_usd"), "generation_latency_s":pred.get("latency_s"),
            "verification_latency_s":time.monotonic()-begin,
            "latency_s":pred.get("latency_s",0) + time.monotonic()-begin,
            "generated_sql":sql, "error":error})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/val.jsonl")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = load_records(args.data, expected_split="val")
    predictions = [json.loads(line) for line in Path(args.predictions).read_text().splitlines() if line.strip()]
    config = dict(vars(args), data_sha256=sha256(args.data), predictions_sha256=sha256(args.predictions))
    directory = start_run(args.output, config)
    results = evaluate(rows, predictions)
    write_jsonl(directory / "per_question.jsonl", results)
    write_json(directory / "summary.json", summarize(results, seed=args.seed))


if __name__ == "__main__":
    main()
