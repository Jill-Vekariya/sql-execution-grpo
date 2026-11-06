import hashlib

import pytest

from sqlrl.executor import Limits, execute, matches, ordered_gold


@pytest.mark.parametrize("sql", [
    "SELECT name FROM people", "WITH x AS (SELECT * FROM people) SELECT * FROM x",
    "/* comment */ SELECT 1", "SELECT 1 UNION ALL SELECT 2",
])
def test_read_queries(db, sql):
    assert execute(db, sql).status == "ok"


@pytest.mark.parametrize("sql", [
    "DROP TABLE people", "DELETE FROM people", "UPDATE people SET id=0",
    "INSERT INTO people VALUES(4,'X',0)", "PRAGMA table_info(people)",
    "ATTACH DATABASE ':memory:' AS other", "VACUUM", "SELECT 1; SELECT 2",
    "SELECT load_extension('x')", "SELECT readfile('/etc/passwd')",
    "WITH x AS (SELECT 1) DELETE FROM people", "SELECT * FROM pragma_database_list",
])
def test_denied_and_database_unchanged(db, sql):
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    assert execute(db, sql).status != "ok"
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


def test_bounds(db):
    assert execute(db, "SELECT * FROM people", Limits(max_rows=2)).status == "row_limit"
    assert execute(db, "SELECT '" + "x" * 1000 + "'", Limits(max_sql_chars=100)).status == "rejected"
    assert execute(db, "SELECT randomblob(100000000)").status != "ok"
    assert execute(db, "SELECT name FROM people", Limits(max_result_bytes=5)).status == "result_limit"


def test_timeout_and_next_query_survives(db):
    sql = "WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x) SELECT sum(n) FROM x"
    assert execute(db, sql, Limits(timeout_s=0.15)).status == "timeout"
    assert execute(db, "SELECT 1").rows == [[1]]


def test_memory_bound(db):
    # A tiny address-space cap must fail closed, rather than fall back unsandboxed.
    assert execute(db, "SELECT 1", Limits(memory_mb=8)).status != "ok"


@pytest.mark.parametrize("gold,pred,ordered,expected", [
    ([[1],[2]], [[2],[1]], False, True),
    ([[1],[2]], [[2],[1]], True, False),
    ([[1],[1]], [[1]], False, False),
    ([[None]], [[None]], False, True),
    ([[None]], [[0]], False, False),
    ([[1.0]], [[1.00000001]], False, True),
    ([[1.0]], [[1.01]], False, False),
    ([[1]], [["1"]], False, False),
    ([[1000000]], [[1000001]], False, False),
    ([[1000000]], [[1000001.0]], False, False),
    ([[1]], [[1.0]], False, True),
    ([[1,2]], [[2,1]], False, False),
    ([], [], False, True),
    ([[1]], [], False, False),
    ([[float("inf")]], [[float("inf")]], False, False),
    ([[float("nan")]], [[float("nan")]], False, False),
    # Requires a real one-to-one assignment, not set conversion or greedy matching.
    ([[1.0],[1.000001]], [[1.0000005],[0.9999995]], False, True),
])
def test_comparison(gold, pred, ordered, expected):
    assert matches(gold, pred, ordered=ordered) is expected


def test_order_is_top_level_only():
    assert ordered_gold("SELECT id FROM people ORDER BY id")
    assert not ordered_gold("SELECT * FROM (SELECT id FROM people ORDER BY id)")
    assert not ordered_gold("SELECT 'ORDER BY' FROM people")


def test_empty_results_keep_column_count(db):
    a = execute(db, "SELECT id FROM people WHERE 0")
    b = execute(db, "SELECT id,name FROM people WHERE 0")
    assert a.columns == 1 and b.columns == 2
    assert not matches(a.rows, b.rows, gold_columns=a.columns, pred_columns=b.columns)
