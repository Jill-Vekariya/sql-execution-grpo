"""The diagnostic report must not hide, duplicate or relabel failed gold cases."""
import copy
import json
from pathlib import Path
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[1]
summarize = runpy.run_path(str(ROOT / "scripts/summarize_gold_failures.py"))["summarize"]


@pytest.fixture
def saved_evidence():
    audit = json.loads((ROOT / "results/gold_audit/summary.json").read_text())
    queries = [json.loads(line) for line in (ROOT / "results/gold_audit/queries.jsonl").read_text().splitlines()]
    diagnostics = json.loads((ROOT / "results/gold_investigation.json").read_text())
    return audit, queries, diagnostics


def test_original_evidence_has_exhaustive_failure_accounting(saved_evidence):
    report = summarize(*saved_evidence)
    assert report["failures"] == 720
    assert sum(row["count"] for row in report["taxonomy"].values()) == 720
    assert report["all_failures_accounted_for"] is True
    assert report["sql_executed"] is False
    assert report["final_test_accessed"] is False


def test_missing_failure_is_rejected(saved_evidence):
    audit, queries, diagnostics = saved_evidence
    changed = copy.deepcopy(diagnostics)
    changed["cases"].pop()
    with pytest.raises(ValueError, match="every failure exactly once"):
        summarize(audit, queries, changed)


def test_relabelled_failure_is_rejected(saved_evidence):
    audit, queries, diagnostics = saved_evidence
    changed = copy.deepcopy(diagnostics)
    changed["cases"][0]["status"] = "ok"
    with pytest.raises(ValueError, match="differs from primary"):
        summarize(audit, queries, changed)
