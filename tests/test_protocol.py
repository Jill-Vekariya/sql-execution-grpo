import json

import pytest

from sqlrl.data import database_split, deduplicate, normalized_question
from sqlrl.eval import bootstrap_ci, extract_sql, summarize, paired_comparison
from sqlrl.eval.prompts import messages, count_chat_tokens
from sqlrl.data.guard import load_records


@pytest.mark.parametrize("answer,sql,formatted", [
    ("```sql\nSELECT 1;\n```", "SELECT 1;", True),
    ("SELECT 1", "SELECT 1", False),
    ("Here you go:\n```sql\nSELECT 1\n```", None, False),
    ("```sql\nSELECT 1\n```\n```sql\nSELECT 2\n```", None, False),
    ("```python\nprint(1)\n```", None, False),
    ("```SQL\nSELECT 1\n```", "SELECT 1", True),
])
def test_extractor(answer, sql, formatted):
    assert extract_sql(answer) == (sql, formatted)


def test_database_splits_reproducible():
    rows = [{"dataset":"spider", "db_id":str(i), "id":str(i)} for i in range(20)]
    a = database_split(rows, seed=42)
    b = database_split(list(reversed(rows)), seed=42)
    assert a == b
    assert set(a["train"]).isdisjoint(a["val"])


def test_cross_split_dedup_preserves_validation():
    rows = {"train":[{"id":"a","question":" How many people? "}],
            "val":[{"id":"b","question":"how MANY people?"}]}
    clean, removed = deduplicate(rows)
    assert not clean["train"] and len(clean["val"]) == 1
    assert removed == [{"id":"a","split":"train","reason":"duplicate_question"}]


def test_final_data_fail_closed(tmp_path):
    final = tmp_path / "final_test.jsonl"
    final.write_text('{"split":"final_test"}\n')
    with pytest.raises(PermissionError):
        load_records(final)
    sneaky = tmp_path / "val.jsonl"
    sneaky.write_text('{"split":"final_test"}\n')
    with pytest.raises(PermissionError):
        load_records(sneaky)


def test_ci_and_paired_identity():
    assert bootstrap_ci([1]*5, seed=42, samples=200) == [1.0,1.0]
    assert bootstrap_ci([0]*5, seed=42, samples=200) == [0.0,0.0]
    result = paired_comparison([0,1,1], [0,1,1], ["a","a","b"], samples=200)
    assert result["wins"] == result["losses"] == 0
    assert result["delta_ci95"] == [0.0,0.0]


def test_summary_keeps_failed_gold_in_denominator():
    rows = [{"id":"a","db_id":"x","correct":False,"status":"gold_error",
             "valid":False,"difficulty":"unknown","output_tokens":None,"latency_s":0.1,"cost_usd":None}]
    s = summarize(rows, seed=42, samples=50)
    assert s["n"] == 1 and s["execution_accuracy"] == 0
    assert s["gold_errors"] == 1 and s["average_output_tokens"] is None


def test_prompt_has_only_schema_question_and_evidence():
    row = {"schema":"CREATE TABLE people(id INT);", "question":"How many?", "evidence":"count rows", "sql":"SECRET GOLD"}
    text = json.dumps(messages(row))
    assert "SECRET GOLD" not in text
    assert "count rows" in text


def test_chat_length_counts_tokens_instead_of_encoding_fields():
    class CurrentTokenizer:
        def apply_chat_template(self,conversation,**kwargs):
            assert kwargs["return_dict"] is True
            return {"input_ids":[11,12,13],"attention_mask":[1,1,1]}
    assert count_chat_tokens(CurrentTokenizer(),[{"role":"user","content":"x"}]) == 3
