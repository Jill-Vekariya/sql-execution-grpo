"""Execute every available train/val gold query; keep all failures in the log."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from sqlrl.common import write_json, write_jsonl
from sqlrl.data.guard import load_records
from sqlrl.executor import execute


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", default=["data/processed/train.jsonl", "data/processed/val.jsonl"])
    parser.add_argument("--output", default="results/gold_audit")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    records = [row for path in args.data for row in load_records(path)]
    results = []
    def check(row):
        result = execute(row["db_path"], row["sql"])
        # Save status and cardinality; avoid duplicating entire answer tables.
        return {"id":row["id"], "dataset":row["dataset"], "db_id":row["db_id"], "split":row["split"],
                        "status":result.status, "row_count":len(result.rows), "columns":result.columns,
                        "elapsed_s":result.elapsed_s, "error":result.error}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(check, records)):
            results.append(result)
            if (i+1) % 500 == 0:
                print(i+1, "gold queries checked", flush=True)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    write_jsonl(directory / "queries.jsonl", results)
    counts = dict(Counter(r["status"] for r in results))
    available = len(results) - counts.get("missing_database", 0)
    write_json(directory / "summary.json", {"config":vars(args), "n":len(results), "status_counts":counts,
        "available_queries":available, "available_execution_rate":counts.get("ok",0)/available if available else None,
        "all_queries_execution_rate":counts.get("ok",0)/len(results) if results else None,
        "by_dataset":{d:dict(Counter(r["status"] for r in results if r["dataset"]==d)) for d in sorted({r["dataset"] for r in results})},
        "final_test_accessed":False})


if __name__ == "__main__":
    main()
