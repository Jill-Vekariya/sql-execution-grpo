"""Diagnose train/val gold failures without editing labels or the main limits."""
import json
from collections import Counter
from pathlib import Path

from sqlrl.common import write_json
from sqlrl.data.guard import load_records
from sqlrl.executor import Limits, execute


def main():
    audit = [json.loads(line) for line in Path("results/gold_audit/queries.jsonl").read_text().splitlines()]
    by_id = {r["id"]:r for name in ["train","val"] for r in load_records("data/processed/" + name + ".jsonl")}
    failures = [r for r in audit if r["status"] not in {"ok","missing_database"}]
    records, diagnosed = [], set()
    limits = Limits(timeout_s=15, max_rows=100000, memory_mb=512, max_result_bytes=16000000)
    for failure in failures:
        row = by_id[failure["id"]]
        record = dict(failure, question=row["question"], gold_sql=row["sql"])
        key = (row["db_id"], failure["status"])
        # One representative per resource-bound group; all individual IDs and
        # SQL errors are retained. This is diagnostics, not a second metric run.
        if failure["status"] == "sql_error" or key not in diagnosed:
            result = execute(row["db_path"], row["sql"], limits)
            record["relaxed_diagnostic"] = {"status":result.status, "row_count":len(result.rows), "elapsed_s":result.elapsed_s,"error":result.error}
            diagnosed.add(key)
        if failure["status"] in {"timeout","row_limit","memory_limit","result_limit"}:
            record["interpretation"] = "Resource bound; not evidence of a bad annotation"
        else:
            record["interpretation"] = "Nonexecuting against retrieved SQLite snapshot; original label retained, unresolved for training eligibility"
        records.append(record)
    write_json("results/gold_investigation.json", {
        "kind":"train_val_gold_diagnostics_not_model_evaluation", "primary_limits_changed":False,
        "diagnostic_limits":{"timeout_s":15,"max_rows":100000,"memory_mb":512,"max_result_bytes":16000000},
        "status_counts":dict(Counter(r["status"] for r in failures)), "cases":records,
        "missing_databases_not_retried":sum(r["status"]=="missing_database" for r in audit),
        "labels_modified":False, "final_test_accessed":False})


if __name__ == "__main__":
    main()
