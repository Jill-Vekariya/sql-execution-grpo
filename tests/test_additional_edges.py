import sqlite3

import pytest

from sqlrl.executor import execute, matches
from sqlrl.reward import score


def test_blob_comparison(db):
    result = execute(db, "SELECT X'00ff'")
    assert result.rows == [[{"blob_hex":"00ff"}]]
    assert matches(result.rows, [[{"blob_hex":"00ff"}]])
    assert not matches(result.rows, [["00ff"]])


def test_missing_database_is_not_created(tmp_path):
    missing = tmp_path / "missing.sqlite"
    assert execute(missing, "SELECT 1").status == "missing_database"
    assert not missing.exists()


def test_uri_special_characters(db):
    new = db.with_name("question?#.sqlite")
    db.rename(new)
    assert execute(new, "SELECT 1").status == "ok"


def test_wal_sidecar_rejected(db):
    sidecar = db.with_name(db.name + "-wal")
    sidecar.write_bytes(b"fixture")
    assert execute(db, "SELECT 1").status == "rejected"


def test_pathological_parser_input_is_bounded(db):
    sql = "SELECT " + "(" * 4000 + "1" + ")" * 4000
    assert execute(db, sql).status != "ok"


def test_large_approximate_comparison_fails_explicitly():
    with pytest.raises(ValueError, match="comparison_limit"):
        matches([[float(i)] for i in range(501)], [[i + 1e-8] for i in range(501)])


def test_counterfactual_rejects_coincidental_answer(db, tmp_path):
    alternate = tmp_path / "counterfactual.sqlite"
    with sqlite3.connect(alternate) as con:
        con.executescript("CREATE TABLE people(id INT); INSERT INTO people VALUES(50);")
    attack = "```sql\nSELECT id FROM people WHERE id < 10\n```"
    reward = score(db, "SELECT id FROM people", attack, counterfactual_databases=[alternate])
    assert reward.raw_execution == 1.0 and reward.execution == 0.0
    assert "counterfactual_mismatch" in reward.flags
