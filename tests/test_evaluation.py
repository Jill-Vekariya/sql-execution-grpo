import pytest

from sqlrl.eval.run import evaluate
from sqlrl.eval.generate import select_examples


def row(db):
    return {"id":"synthetic:a", "db_id":"fixture", "dataset":"synthetic", "split":"val",
            "question":"Names ordered by descending id", "sql":"SELECT name FROM people ORDER BY id DESC",
            "db_path":str(db), "schema":"CREATE TABLE people(id INTEGER,name TEXT,score REAL);"}


def test_end_to_end_verifier(db):
    r = row(db)
    prediction = {"id":r["id"], "answer":"```sql\nSELECT name FROM people ORDER BY id DESC\n```", "output_tokens":12}
    out = evaluate([r], [prediction])
    assert out[0]["correct"] and out[0]["valid"]
    assert out[0]["output_tokens"] == 12
    prediction["answer"] = "```sql\nSELECT name FROM people ORDER BY name\n```"
    assert not evaluate([r], [prediction])[0]["correct"]


def test_all_predictions_required(db):
    with pytest.raises(ValueError, match="exactly"):
        evaluate([row(db)], [])


def test_few_shot_database_diversity_and_seed():
    rows = [{"id":str(i),"dataset":"spider","db_id":str(i//2)} for i in range(20)]
    first = select_examples(rows, 3, 42)
    assert first == select_examples(list(reversed(rows)), 3, 42)
    assert len({r["db_id"] for r in first}) == 3


def test_missing_db_keeps_record_as_failure(tmp_path):
    r = row(tmp_path / "missing.sqlite")
    results = evaluate([r], [{"id":r["id"],"answer":"```sql\nSELECT 1\n```"}])
    assert len(results) == 1
    assert results[0]["status"] == "gold_error"
