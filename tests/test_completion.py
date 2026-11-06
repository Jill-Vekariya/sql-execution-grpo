import json
import sys

import pytest

from sqlrl import complete_phase03 as completion
from sqlrl.common import sha256, write_json, write_jsonl
from sqlrl.data import question_digest
from sqlrl.data.restore_eligible import restore


def test_cost_is_per_percentage_point_and_does_not_invent_a_gain():
    assert completion.cost_per_point(12, 0.05) == pytest.approx(2.4)
    assert completion.cost_per_point(0, 0.05) == 0
    assert completion.cost_per_point(None, 0.05) is None
    assert completion.cost_per_point(12, 0) is None
    assert completion.cost_per_point(12, -0.05) is None
    with pytest.raises(ValueError):
        completion.cost_per_point(float("inf"), 0.05)


def test_provider_configuration_cannot_replace_the_evaluation_split(tmp_path):
    with pytest.raises(ValueError, match="protocol field"):
        completion.plan_configs(tmp_path, 0, {"data":"data/processed/test.jsonl"})
    with pytest.raises(ValueError, match="protocol field"):
        completion.plan_configs(tmp_path, 0, {"api_key":"never-save-this"})


def test_blocked_execution_starts_no_model_or_api(tmp_path, monkeypatch):
    root = tmp_path / "attempt"
    monkeypatch.setattr(completion, "readiness", lambda configs:{"ready":False, "blockers":["GPU missing"]})
    def forbidden(*args):
        pytest.fail("Blocked execution must not reach the model/API pipeline")
    monkeypatch.setattr(completion, "execute_plan", forbidden)
    monkeypatch.setattr(sys, "argv", ["complete", "--output", str(root), "--execute"])
    assert completion.main() == 2
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "blocked" and not status["model_runs_started"]
    with pytest.raises(FileExistsError):
        completion.main()


def test_changed_original_validation_label_blocks_completion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train = {"id":"spider:00000", "dataset":"spider", "db_id":"train_db", "split":"train", "question":"Return one", "sql":"SELECT 1", "schema":"CREATE TABLE t(x);", "db_path":str(tmp_path / "train.sqlite")}
    val = dict(train, id="spider:00001", db_id="val_db", split="val", question="Return two", sql="SELECT 2", db_path=str(tmp_path / "val.sqlite"))
    for row, name in [(train,"train_eligible"),(val,"val")]:
        (tmp_path / ("train.sqlite" if row["split"] == "train" else "val.sqlite")).write_bytes(b"fixture")
        write_jsonl(f"data/processed/{name}.jsonl", [row])
        write_jsonl(f"data/manifests/{name}.jsonl", [{"id":row["id"], "dataset":"spider", "db_id":row["db_id"], "question_sha256":question_digest(row["question"])}])
    source = [{"question":r["question"], "query":r["sql"]} for r in [train,val]]
    write_json("data/raw/spider/train.json", source)
    write_json("results/data_audit.json", {"sources":{"spider":{"source_sha256":sha256("data/raw/spider/train.json")}}})
    write_json("results/training_eligibility.json", {"processed_train_sha256":sha256("data/processed/train_eligible.jsonl")})
    config = {"train_data":"data/processed/train_eligible.jsonl", "data":"data/processed/val.jsonl"}
    assert completion.data_blockers(config)[0] == []
    val["sql"] = "SELECT 999"
    write_jsonl(config["data"], [val])
    assert any("Original annotation changed" in x for x in completion.data_blockers(config)[0])


def test_paired_report_rejects_different_question_ids(tmp_path):
    for name, identity in [("base_zero","a"),("base_few","b"),("sft_eval","a"),("frontier","a")]:
        directory = tmp_path / (name + "_evaluation")
        write_json(directory / "summary.json", {})
        write_jsonl(directory / "per_question.jsonl", [{"id":identity}])
    with pytest.raises(ValueError, match="same unique"):
        completion.report(tmp_path, {})


def test_measured_report_keeps_nonpositive_sft_improvement_undefined(tmp_path):
    for name in ["base_zero", "base_few", "sft_eval", "frontier"]:
        directory = tmp_path / (name + "_evaluation")
        write_json(directory / "summary.json", {"execution_accuracy":1.0, "execution_accuracy_ci95":[1.0,1.0], "total_cost_usd":0.5})
        write_jsonl(directory / "per_question.jsonl", [{"id":"a", "dataset":"fixture", "db_id":"one", "correct":True}])
    report = completion.report(tmp_path, {"base_zero":1.0, "base_few":1.0, "sft":10.0, "sft_eval":1.0})
    assert report["measured_costs_usd"]["sft_eval"] == 11
    assert report["sft_cost_per_accuracy_percentage_point_vs_base"] is None
    assert report["paired_vs_base_zero"]["sft_eval"]["delta"] == 0
    assert not report["rl_trained"] and not report["final_test_accessed"]


def test_restore_reproduces_the_exact_published_bytes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    row = {"id":"spider:00000", "dataset":"spider", "db_id":"one", "split":"train", "question":"Find café", "sql":"SELECT 1"}
    write_jsonl("data/processed/train.jsonl", [row])
    write_jsonl("data/manifests/train_eligible.jsonl", [{"id":row["id"], "dataset":"spider", "db_id":"one", "question_sha256":question_digest(row["question"])}])
    write_json("results/training_eligibility.json", {"raw_train_sha256":sha256("data/processed/train.jsonl"), "manifest":{"sha256":sha256("data/manifests/train_eligible.jsonl")}, "processed_train_sha256":sha256("data/processed/train.jsonl")})
    restore()
    assert sha256("data/processed/train_eligible.jsonl") == sha256("data/processed/train.jsonl")
    # Repeating an identical restore is safe; a different file is preserved.
    restore()
    path = tmp_path / "data/processed/train_eligible.jsonl"
    path.write_text("preserve this changed file")
    with pytest.raises(FileExistsError, match="preserving"):
        restore()
    assert path.read_text() == "preserve this changed file"


def test_restore_refuses_changed_training_labels(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_jsonl("data/processed/train.jsonl", [{"id":"a", "split":"train", "sql":"SELECT 999"}])
    write_json("results/training_eligibility.json", {"raw_train_sha256":"different published hash"})
    with pytest.raises(ValueError, match="training bytes differ"):
        restore()
    assert not (tmp_path / "data/processed/train_eligible.jsonl").exists()
