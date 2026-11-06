"""Verify file integrity and state phase completion without overclaiming."""
import json
from pathlib import Path

from sqlrl.common import sha256, write_json


def main():
    audit = json.loads(Path("results/data_audit.json").read_text())
    missing=sum(s.get("missing_database_count",0) for s in audit["sources"].values())
    eligibility=json.loads(Path("results/training_eligibility.json").read_text()) if Path("results/training_eligibility.json").exists() else None
    checks = []
    for dataset, data in audit["sources"].items():
        for db_id, expected in data.get("database_sha256",{}).items():
            path = Path("data/databases") / dataset / db_id / (db_id + ".sqlite")
            checks.append({"dataset":dataset,"db_id":db_id,"matches_initial_sha256":sha256(path)==expected})
    for split, manifest in audit["splits"].items():
        if sha256(manifest["path"]) != manifest["sha256"]:
            raise AssertionError("Split manifest checksum changed: " + split)
    if eligibility and sha256(eligibility["manifest"]["path"])!=eligibility["manifest"]["sha256"]:
        raise AssertionError("Eligible training manifest checksum changed")
    if not all(check["matches_initial_sha256"] for check in checks):
        raise AssertionError("Database contents changed during read-only verification")
    write_json("results/database_integrity.json", {"checked":len(checks),"all_unchanged":True,"checks":checks})
    write_json("results/phase_completion.json", {
        "scope":[0,1,2,3], "user_authorized_no_intermediate_stops":True,
        "phase_0":{"status":"train_val_audit_complete" if not missing else "partial","completed":["repository layout","official license/source audit","full TRAIN annotation audit","database-disjoint train/val manifests with checksums","normalized duplicate removal"],
                   "pending":(["training database download"] if missing else [])+["final checksum and cross-final duplicate check reserved under no-access rule"]},
        "phase_1":{"status":"implemented_and_tested_full_corpus" if not missing else "implemented_and_tested_partial_corpus","evidence":["results/tests-final.xml","results/gold_audit/summary.json","results/gold_investigation.json"],"pending":(["full gold execution verification"] if missing else [])+([] if eligibility else ["shared training eligibility freeze"])},
        "phase_2":{"status":"implemented_model_runs_blocked","evidence":"results/run_attempts.json","pending":["base zero-shot VAL run","base few-shot VAL run","frontier VAL run"]},
        "phase_3":{"status":"implemented_cpu_api_smoke_passed_full_sft_blocked","evidence":"results/cpu_sft_smoke/status.json","pending":["full Qwen LoRA SFT on CUDA","SFT VAL execution evaluation","measured GPU training cost"]},
        "final_data_policy":{"dev_questions_opened":False,"dev_gold_sql_opened":False,"dev_databases_opened":False,"shared_schema_metadata_read":True,"final_evaluation_run":False},
        "no_accuracy_improvement_claim":True})


if __name__ == "__main__":
    main()
