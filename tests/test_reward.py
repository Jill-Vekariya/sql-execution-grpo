import pytest

from sqlrl.executor import execute
from sqlrl.reward import score


def test_execution_dominates_format(db):
    good = score(db, "SELECT id FROM people", "```sql\nSELECT id FROM people\n```", robust=False)
    bad = score(db, "SELECT id FROM people", "```sql\nSELECT 999\n```", robust=False)
    assert good.execution == 1.0
    assert good.total > bad.total + 0.8


@pytest.mark.parametrize("attack", [
    "```sql\nDROP TABLE people\n```", "```sql\nSELECT id FROM people LIMIT 1\n```",
    "```sql\nSELECT 1 UNION SELECT 2 UNION SELECT 3\n```", "```sql\nSELECT 1 WHERE 0\n```",
    "```sql\nSELECT id FROM people; DELETE FROM people\n```",
])
def test_attacks_do_not_receive_execution_reward(db, attack):
    r = score(db, "SELECT id FROM people", attack, robust=True)
    assert r.execution == 0
    assert r.total < 0.1


def test_empty_gold_ambiguity(db):
    # Single-snapshot EX cannot distinguish two empty answers. Robust mode refuses
    # execution credit unless independently supplied counterfactual DBs also match.
    r = score(db, "SELECT id FROM people WHERE id > 99", "```sql\nSELECT 1 WHERE 0\n```", robust=True)
    assert r.raw_execution == 1.0 and r.execution == 0.0
    assert "empty_result_unverified" in r.flags


def test_long_output_rejected_before_execution(db):
    r = score(db, "SELECT 1", "```sql\nSELECT 1 " + " " * 25000 + "\n```", robust=True)
    assert r.execution == 0 and r.total <= 0


def test_reward_sql_errors_are_zero(db):
    assert score(db, "SELECT id FROM people", "```sql\nSELECT nope FROM people\n```").execution == 0


def test_large_integer_near_miss_gets_no_execution_credit(db):
    reward = score(db,"SELECT 1000000 FROM people WHERE id=1","```sql\nSELECT 1000001.0\n```",robust=False)
    assert reward.raw_execution == reward.execution == 0


def test_invalid_gold_is_an_error(db):
    with pytest.raises(ValueError, match="Gold"):
        score(db, "SELECT missing FROM people", "```sql\nSELECT 1\n```")
