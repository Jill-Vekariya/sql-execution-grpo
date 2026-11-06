"""Render a compact README from saved evidence; no hand-entered run scores."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from sqlrl.common import write_json


def read(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def main():
    data = read("results/data_audit.json")
    gold = read("results/gold_audit/summary.json")
    environment = read("results/environment.json")
    model = read("results/model_selection.json")
    cpu_smoke = read("results/cpu_sft_smoke/status.json")
    eligibility = read("results/training_eligibility.json")
    suites = ET.parse("results/tests-final.xml").getroot().findall(".//testsuite")
    tests = {key:sum(int(s.get(key, 0)) for s in suites) for key in ["tests","failures","errors","skipped"]}
    runs = {}
    for label, name in [("Base zero-shot","base_zero_seed42"),("Base few-shot","base_few_seed42"),("Frontier","frontier_seed42"),("SFT","sft_seed42")]:
        runs[label] = read("results/runs/" + name + "/status.json") or {"status":"not_run"}
    state = {"scope":[0,1,2,3], "project":"Execution-rewarded text-to-SQL with GRPO",
        "model":model, "tests":tests, "runs":runs, "data_audit":"results/data_audit.json",
        "gold_audit":"results/gold_audit/summary.json", "final_test_accessed":False,
        "rl_trained":False, "adapter_released":False, "research_accuracy_available":False,"cpu_api_smoke":cpu_smoke,"training_eligibility":eligibility,
        "limitations":["No CUDA GPU available", "No frontier model/API budget configured", "Final checksum/cross-final duplicate audit pending under no-access rule", "Benchmark contamination possible", "SQLite only", "No benchmark training seeds run yet", "Official dev will serve as final test"]}
    write_json("results/project_status.json", state)
    lines = ["# Execution-rewarded text-to-SQL with GRPO", "",
        "Study whether execution rewards improve a small Qwen model beyond base prompting and LoRA SFT, with auditable costs and reward exploits.", "",
        "**Work in progress: Phases 0–3 implementation is present; model benchmark runs and full SFT are blocked. No RL performance or released adapter is claimed.**", "",
        "Evidence: [project status](results/project_status.json), [tests](results/tests-final.xml), [data audit](results/data_audit.json), [gold-query audit](results/gold_audit/summary.json).", "",
        "| System | Validation accuracy / 95% CI | Training / API cost | Run status |", "|---|---|---|---|"]
    for label, run in runs.items():
        lines.append(f"| {label} | Not measured | Not measured | {run['status']} |")
    lines += ["| GRPO | Not measured | Not measured | Future Phase 5 |", "",
        "Missing costs are unknown, not zero. Failure/preflight reasons and configs are saved under [results/runs](results/runs).", "",
        f"Verifier suite: **{tests['tests']-tests['failures']-tests['errors']-tests['skipped']} passed**, {tests['failures']} failures, {tests['errors']} errors ([saved XML](results/tests-final.xml))."]
    if data:
        lines += [f"Training annotation audit: **{sum(s.get('questions',0) for s in data['sources'].values()):,} source questions**; **{data['duplicate_count']} duplicates removed** across project train/val ([audit](results/data_audit.json))."]
    if gold:
        lines += [f"Gold execution on available databases: **{gold['status_counts'].get('ok',0):,}/{gold['available_queries']:,} ({100*gold['available_execution_rate']:.2f}%)**. **{gold['status_counts'].get('missing_database',0):,}** declared examples lack downloaded databases; all statuses remain in the [audit](results/gold_audit/summary.json)."]
    if cpu_smoke:
        lines += [f"CPU SFT API integration: **{cpu_smoke['status']}** on a random tiny Qwen3 fixture. This is not a benchmark model or accuracy result ([evidence](results/cpu_sft_smoke/status.json))."]
    if eligibility:
        lines += [f"Shared SFT/RL/few-shot training pool: **{eligibility['eligible_questions']:,}** eligible questions; exclusions and the immutable manifest are recorded in the [eligibility report](results/training_eligibility.json)."]
    lines += ["", "```mermaid", "flowchart TD", '  D["Database-disjoint train / val"] --> P["Fixed prompt and model"]', '  P --> X["Bounded read-only SQLite execution"]', '  X --> E["Accuracy, uncertainty, cost and failures"]', "```", "",
        "CPU reproduction in three commands (Python 3.12, uv 0.13.7):", "", "```bash", "uv sync --frozen --extra test --extra data", "uv run --no-sync pytest", "uv run --no-sync python scripts/synthetic_checks.py --output results/reproduction_synthetic", "```", "",
        "The synthetic check is a verifier demonstration, not model accuracy. [Full data/baseline/SFT reproduction](docs/REPRODUCE.md).", "",
        "What failed: [FAILURES.md](docs/FAILURES.md) preserves the initial memory-cap test failure and infrastructure blockers. [Cheating analysis](docs/WHAT_THE_MODEL_LEARNED_TO_CHEAT.md) distinguishes anticipated exploits from learned behavior. [Decisions](docs/DECISIONS.md) explains the stricter comparator and pending final-set audit.", "",
        "Limitations: unmeasured model performance; no training seeds completed; possible benchmark contamination; SQLite only; official-dev-as-final-test caveat; BIRD database coverage and final-manifest checks remain explicit in the audit. No final test was run ([status](results/project_status.json)).", "",
        "How I would extend this:", "",
        "- Run the frozen base and SFT configurations on a CUDA GPU.", "- Add a budgeted, versioned frontier baseline.", "- Freeze and ablate the GRPO reward after counterfactual-query tests.", "- Train GRPO with multiple seeds and select by validation execution accuracy.", "- Run the final test once after freezing the complete experiment.", "",
        "Code: MIT. Dataset terms and primary sources: [data audit](docs/DATA_AUDIT.md).", ""]
    Path("README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
