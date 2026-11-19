"""Classify saved TRAIN/VAL gold failures; never execute SQL or open final data."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from sqlrl.common import sha256, write_json


def classify(case):
    """Report observed failure mechanisms without asserting that labels are wrong."""
    if case["status"] != "sql_error":
        return "resource_" + case["status"]
    error = case.get("error") or ""
    if error.startswith("no such table:"):
        return "missing_table"
    if error.startswith("no such column:"):
        return "missing_column"
    if "ORDER BY clause should come after" in error:
        return "compound_select_ordering"
    return "other_sql_error"


def summarize(audit, queries, diagnostics):
    """Require a one-to-one account of every original failing query."""
    query_ids = [row["id"] for row in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Duplicate primary query IDs")
    if len(queries) != audit["n"] or dict(Counter(r["status"] for r in queries)) != audit["status_counts"]:
        raise ValueError("Primary query records differ from the audit summary")
    failed = {row["id"]: row for row in queries if row["status"] != "ok"}
    cases = diagnostics["cases"]
    case_ids = [row["id"] for row in cases]
    if len(case_ids) != len(set(case_ids)) or set(case_ids) != set(failed):
        raise ValueError("Diagnostics must account for every failure exactly once")
    for case in cases:
        if case["status"] != failed[case["id"]]["status"] or case.get("error") != failed[case["id"]].get("error"):
            raise ValueError("Diagnostic status/error differs from primary evidence")

    taxonomy = defaultdict(list)
    databases = defaultdict(Counter)
    sql_errors = Counter()
    relaxed = Counter()
    by_split = defaultdict(Counter)
    for case in cases:
        taxonomy[classify(case)].append(case["id"])
        databases[case["dataset"] + ":" + case["db_id"]][case["status"]] += 1
        by_split[case["split"]][case["status"]] += 1
        if case["status"] == "sql_error":
            sql_errors[case.get("error") or "unspecified"] += 1
        if "relaxed_diagnostic" in case:
            relaxed[case["relaxed_diagnostic"]["status"]] += 1

    return {
        "kind": "classification_of_saved_gold_execution_failures",
        "queries": len(queries),
        "gold_executable": audit["status_counts"]["ok"],
        "gold_execution_rate": audit["all_queries_execution_rate"],
        "failures": len(failed),
        "all_failures_accounted_for": True,
        "taxonomy": {name: {"count": len(ids), "ids": sorted(ids)} for name, ids in sorted(taxonomy.items())},
        "by_split": {name: dict(counts) for name, counts in sorted(by_split.items())},
        "by_database": {name: dict(counts) for name, counts in sorted(databases.items())},
        "sql_errors": [{"error": error, "count": count} for error, count in sql_errors.most_common()],
        "previous_relaxed_probe_statuses": dict(relaxed),
        "interpretation": [
            "Missing table/column errors establish incompatibility between saved SQL and the frozen SQLite snapshot; they do not identify which source is at fault.",
            "Resource-limit failures do not establish annotation errors.",
            "No label, database, training eligibility, comparator or resource limit was changed.",
            "The near-100-percent gold-execution target remains unmet.",
            "Gold-error VAL questions remain in the evaluation denominator.",
        ],
        "sql_executed": False,
        "final_test_accessed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()
    output, markdown = Path(args.output), Path(args.markdown)
    if output.exists() or markdown.exists():
        raise FileExistsError("Preserve existing reports; choose new output paths")
    paths = {
        "audit": Path("results/gold_audit/summary.json"),
        "queries": Path("results/gold_audit/queries.jsonl"),
        "diagnostics": Path("results/gold_investigation.json"),
    }
    audit = json.loads(paths["audit"].read_text())
    queries = [json.loads(line) for line in paths["queries"].read_text().splitlines() if line.strip()]
    diagnostics = json.loads(paths["diagnostics"].read_text())
    report = summarize(audit, queries, diagnostics)
    report["sources"] = {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()}
    write_json(output, report)
    lines = [
        "# Saved gold-query failure analysis", "",
        f"All {report['failures']:,} failures from the saved {report['queries']:,}-query TRAIN/VAL audit are accounted for exactly once.",
        "This report classifies saved evidence. It runs no queries and opens no final-test content.", "",
        "| Observed mechanism | Queries |", "|---|---:|",
    ]
    lines += [f"| {name.replace('_', ' ')} | {row['count']} |" for name, row in report["taxonomy"].items()]
    lines += ["", "## Interpretation", ""] + [f"- {item}" for item in report["interpretation"]]
    lines += ["", "## Most frequent SQL errors", "", "| Saved SQLite error | Count |", "|---|---:|"]
    lines += [f"| {row['error']} | {row['count']} |" for row in report["sql_errors"][:10]]
    lines += ["", f"Full case IDs, split/database breakdowns, prior relaxed probes and source hashes: [{output.name}](../{output.as_posix()}).", ""]
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))
    print(json.dumps({"failures_accounted_for": report["failures"], "taxonomy_counts": {k: v["count"] for k, v in report["taxonomy"].items()}, "final_test_accessed": False}))


if __name__ == "__main__":
    main()
