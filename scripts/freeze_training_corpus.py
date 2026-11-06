"""Freeze shared SFT/RL/few-shot eligibility before any benchmark model run.

Keep the raw train/val split intact. Quarantine training gold that cannot be
verified within the declared executor budget, and examples beyond SFT context.
Validation is never filtered by this script.
"""
import json
from pathlib import Path

from transformers import AutoTokenizer

from sqlrl.common import sha256,write_json,write_jsonl
from sqlrl.data import question_digest
from sqlrl.data.guard import load_records
from sqlrl.eval.prompts import messages,count_chat_tokens


def main():
    config=json.loads(Path("configs/sft.json").read_text())
    raw=load_records("data/processed/train.jsonl",expected_split="train")
    audits=[json.loads(line) for line in Path("results/gold_audit/queries.jsonl").read_text().splitlines()]
    by_id={r["id"]:r for r in audits}
    if any(r["status"]=="missing_database" for r in audits):
        raise RuntimeError("Cannot freeze eligibility while declared databases are missing")
    tokenizer=AutoTokenizer.from_pretrained(config["model"],revision=config["revision"])
    selected,excluded,lengths=[],[],[]
    for row in raw:
        if by_id[row["id"]]["status"]!="ok":
            excluded.append({"id":row["id"],"reason":"gold_"+by_id[row["id"]]["status"]})
            continue
        completion={"role":"assistant","content":"```sql\n"+row["sql"]+"\n```"}
        length=count_chat_tokens(tokenizer,messages(row)+[completion])
        lengths.append(length)
        if length>config["max_length"]:
            excluded.append({"id":row["id"],"reason":"context_limit","tokens":length})
        else:
            selected.append(row)
    output=Path("data/processed/train_eligible.jsonl")
    write_jsonl(output,selected)
    manifest=Path("data/manifests/train_eligible.jsonl")
    write_jsonl(manifest,[{"id":r["id"],"dataset":r["dataset"],"db_id":r["db_id"],"question_sha256":question_digest(r["question"])} for r in selected])
    for path in Path("configs").glob("*.json"):
        value=json.loads(path.read_text())
        if "train_data" in value:
            value["train_data"]=str(output)
            write_json(path,value)
    write_json("results/training_eligibility.json",{
        "policy":"shared_gold_executable_and_fits_context", "raw_train_questions":len(raw),
        "eligible_questions":len(selected),"excluded":excluded,"max_context_tokens":config["max_length"],
        "tokenizer_model":config["model"],"tokenizer_revision":config["revision"],
        "token_lengths":{"min":min(lengths),"max":max(lengths)},
        "manifest":{"path":str(manifest),"sha256":sha256(manifest)},
        "processed_train_sha256":sha256(output),"raw_train_sha256":sha256("data/processed/train.jsonl"),
        "gold_audit_sha256":sha256("results/gold_audit/queries.jsonl"),
        "validation_modified":False,"labels_rewritten":False,"benchmark_models_run":False})


if __name__=="__main__":main()
