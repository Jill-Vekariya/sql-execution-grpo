"""Restore the published training pool without rerunning hardware-dependent gold audits."""
import hashlib
import json
from pathlib import Path

from sqlrl.common import sha256
from sqlrl.data import question_digest
from sqlrl.data.guard import load_records


def restore():
    report = json.loads(Path("results/training_eligibility.json").read_text())
    raw_path = Path("data/processed/train.jsonl")
    manifest = Path("data/manifests/train_eligible.jsonl")
    if sha256(raw_path) != report["raw_train_sha256"]:
        raise ValueError("Prepared training bytes differ from the published corpus")
    if sha256(manifest) != report["manifest"]["sha256"]:
        raise ValueError("Eligible manifest differs from the published corpus")
    rows = {r["id"]:r for r in load_records(raw_path, expected_split="train")}
    selected = []
    seen = set()
    for line in manifest.read_text().splitlines():
        entry = json.loads(line)
        row = rows[entry["id"]]
        actual = {"id":row["id"], "dataset":row["dataset"], "db_id":row["db_id"], "question_sha256":question_digest(row["question"])}
        if entry != actual or row["id"] in seen:
            raise ValueError("Eligible manifest identity mismatch or duplicate")
        seen.add(row["id"])
        selected.append(row)
    # This is the same canonical serialization used by write_jsonl.
    payload = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in selected).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != report["processed_train_sha256"]:
        raise ValueError("Restored eligible bytes differ from the published corpus")
    output = Path("data/processed/train_eligible.jsonl")
    if output.exists():
        if sha256(output) != digest:
            raise FileExistsError("Existing eligible training file differs; preserving it")
    else:
        with output.open("xb") as stream:
            stream.write(payload)
    print(json.dumps({"restored_examples":len(selected), "sha256":digest, "new_audit_performed":False, "final_test_accessed":False}))


if __name__ == "__main__":
    restore()
