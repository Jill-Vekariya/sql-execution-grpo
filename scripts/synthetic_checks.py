"""Produce real, reproducible verifier/reward evidence on a synthetic fixture.

These hand-authored predictions are not any model's benchmark results.
"""
import sqlite3
import argparse
import tempfile
from dataclasses import asdict
from pathlib import Path

from sqlrl.common import start_run, write_json, write_jsonl
from sqlrl.eval import summarize
from sqlrl.eval.run import evaluate
from sqlrl.reward import score


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="results/synthetic_checks")
    args=parser.parse_args()
    directory = start_run(args.output, {"seed":42, "kind":"synthetic_verifier_and_attack_fixture", "model_used":False})
    with tempfile.TemporaryDirectory() as temporary:
        db = Path(temporary) / "fixture.sqlite"
        with sqlite3.connect(db) as con:
            con.executescript("CREATE TABLE people(id INTEGER,name TEXT); INSERT INTO people VALUES(1,'Ada'),(2,'Bob'),(3,'Ada');")
        gold = "SELECT id FROM people"
        cases = {
            "correct":"SELECT id FROM people", "empty":"SELECT 1 WHERE 0",
            "hardcoded":"SELECT 1 UNION SELECT 2 UNION SELECT 3",
            "limit_trick":"SELECT id FROM people LIMIT 1", "column_permutation":"SELECT name,id FROM people",
            "write_attempt":"DELETE FROM people", "extension":"SELECT load_extension('x')",
            "multi_statement":"SELECT id FROM people; DROP TABLE people",
            "timeout":"WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x) SELECT sum(n) FROM x",
        }
        rewards = []
        for name, sql in cases.items():
            answer = "```sql\n" + sql + "\n```"
            result = score(db, gold, answer)
            rewards.append({"case":name, "gold_sql":gold, "generated_sql":sql, **asdict(result), "total":result.total})
        result = score(db, "SELECT id FROM people WHERE id > 99", "```sql\nSELECT 1 WHERE 0\n```", robust=True)
        rewards.append({"case":"empty_gold_ambiguity", "gold_sql":"SELECT id FROM people WHERE id > 99", "generated_sql":"SELECT 1 WHERE 0", **asdict(result), "total":result.total})
        write_jsonl(directory / "attack_rewards.jsonl", rewards)
        rows, predictions = [], []
        for name, sql in cases.items():
            rows.append({"id":"synthetic:" + name, "dataset":"synthetic", "db_id":"fixture", "split":"val", "difficulty":"synthetic",
                         "question":"Return all person ids", "sql":gold, "schema":"CREATE TABLE people(id INTEGER,name TEXT);", "db_path":str(db)})
            predictions.append({"id":"synthetic:" + name, "answer":"```sql\n" + sql + "\n```"})
        results = evaluate(rows, predictions)
        write_jsonl(directory / "per_question.jsonl", results)
        write_json(directory / "summary.json", dict(summarize(results), kind="synthetic_verifier_not_model_evaluation"))


if __name__ == "__main__":
    main()
