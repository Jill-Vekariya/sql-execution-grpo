"""Prepare reproducible project train/val splits without opening official dev."""
import argparse
import json
import sqlite3
from pathlib import Path

from sqlrl.common import sha256, write_json, write_jsonl
from sqlrl.data import database_split, deduplicate, question_digest


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def schema_from_metadata(metadata):
    statements = []
    for index, table in enumerate(metadata["table_names_original"]):
        columns = []
        for (table_index, name), kind in zip(metadata["column_names_original"], metadata["column_types"]):
            if table_index == index:
                columns.append(quote(name) + " " + {"number":"REAL", "text":"TEXT", "time":"TEXT", "boolean":"INTEGER"}.get(kind, "TEXT"))
        statements.append("CREATE TABLE " + quote(table) + " (" + ", ".join(columns) + ");")
    for first, second in metadata.get("foreign_keys", []):
        ta, ca = metadata["column_names_original"][first]
        tb, cb = metadata["column_names_original"][second]
        statements.append("-- Foreign key: " + quote(metadata["table_names_original"][ta]) + "." + quote(ca) + " -> " + quote(metadata["table_names_original"][tb]) + "." + quote(cb))
    return "\n".join(statements)


def read_schema(path):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as con:
        # Trusted source DB metadata, never a model-produced query.
        return "\n".join(r[0] + ";" for r in con.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY name"))


def prepare(seed=42):
    records, stats, assignments = [], {}, {"train": [], "val": []}
    for dataset in ["spider", "bird"]:
        path = Path("data/raw") / dataset / "train.json"
        if not path.exists():
            stats[dataset] = {"status": "missing_training_annotations"}
            continue
        source = json.loads(path.read_text(encoding="utf-8"))
        tables_path = path.with_name("tables.json")
        tables = {t["db_id"]: t for t in json.loads(tables_path.read_text())} if tables_path.exists() else {}
        schemas, database_hashes = {}, {}
        for db_id in sorted({r["db_id"] for r in source}):
            db = Path("data/databases") / dataset / db_id / (db_id + ".sqlite")
            if db.exists():
                schemas[db_id] = read_schema(db)
                database_hashes[db_id] = sha256(db)
            else:
                schemas[db_id] = schema_from_metadata(tables[db_id]) if db_id in tables else ""
        rows = [{"id": f"{dataset}:{i:05d}", "dataset": dataset, "source_split":"train", "db_id": r["db_id"],
                 "question": r["question"], "sql": r.get("SQL", r.get("query")),
                 "evidence": r.get("evidence", ""), "difficulty": r.get("difficulty", "unknown"),
                 "schema": schemas[r["db_id"]],
                 "db_path": str(Path("data/databases") / dataset / r["db_id"] / (r["db_id"] + ".sqlite"))}
                for i, r in enumerate(source)]
        split = database_split(rows, seed=seed)
        for name in assignments:
            assignments[name].extend(split[name])
        records.extend(rows)
        stats[dataset] = {"status":"audited", "source_sha256":sha256(path), "questions": len(rows),
                          "databases":len(schemas), "database_sha256":database_hashes,
                          "missing_database_count":len(schemas)-len(database_hashes),
                          "questions_missing_schema":sum(not r["schema"] for r in rows)}
    splits = {name: [dict(r, split=name) for r in records if r["dataset"] + ":" + r["db_id"] in dbs] for name, dbs in assignments.items()}
    splits, removed = deduplicate(splits)
    # Identical database bytes across split boundaries are a hard leakage failure.
    hashes = {name: {stats[r["dataset"]]["database_sha256"].get(r["db_id"]) for r in rows} - {None} for name, rows in splits.items()}
    if hashes["train"] & hashes["val"]:
        raise ValueError("Identical database bytes appear across splits")
    manifests = {}
    for name, rows in splits.items():
        write_jsonl(Path("data/processed") / (name + ".jsonl"), rows)
        manifest = Path("data/manifests") / (name + ".jsonl")
        write_jsonl(manifest, [{"id":r["id"], "dataset":r["dataset"], "db_id":r["db_id"], "question_sha256":question_digest(r["question"])} for r in rows])
        manifests[name] = {"path":str(manifest), "sha256":sha256(manifest), "questions":len(rows), "databases": sorted({r["dataset"] + ":" + r["db_id"] for r in rows})}
    write_json("data/manifests/checksums.json", manifests)
    write_json("data/manifests/final_test.plan.json", {
        "status":"sealed_not_downloaded", "phase_available":9, "datasets":["Spider 1.0 official dev", "BIRD-SQL 2023 official dev"],
        "sha256":None, "reason":"No held-out annotations or DBs opened before Phase 9",
        "cross_final_question_dedup":"not_performed: requires access forbidden before Phase 9"})
    write_json("results/data_audit.json", {"seed":seed, "val_fraction_per_dataset":0.15, "sources":stats,
        "splits":manifests, "removed_duplicates":removed, "duplicate_count":len(removed),
        "final_test_accessed":False, "cross_final_dedup_verified":False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    prepare(parser.parse_args().seed)
