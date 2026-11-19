"""Imported results must preserve the full cohort and saved input hashes."""
import json
from pathlib import Path
import runpy
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
verify = runpy.run_path(str(ROOT / "scripts/verify_saved_run.py"))["verify"]
RUN = ROOT / "results/free_runs_batch8_v2/base_zero"
EVALUATION = ROOT / "results/free_runs_batch8_v2/base_zero_evaluation"


def test_original_complete_baseline_replays():
    report = verify(RUN, EVALUATION)
    assert report["questions"] == 2304
    assert report["correct"] == 1492
    assert report["summary_and_bootstrap_reproduced"]
    assert not report["sql_executed"]
    assert not report["final_test_accessed"]


def test_incomplete_run_is_rejected(tmp_path):
    copied = tmp_path / "run"
    shutil.copytree(RUN, copied)
    state = json.loads((copied / "status.json").read_text())
    state["status"] = "paused"
    (copied / "status.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="completed full-cohort"):
        verify(copied, EVALUATION)


def test_held_out_data_path_is_rejected_before_it_can_be_read(tmp_path):
    copied = tmp_path / "run"
    shutil.copytree(RUN, copied)
    config = json.loads((copied / "config.json").read_text())
    config["data"] = str(tmp_path / "must_never_be_opened_final.jsonl")
    (copied / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Only the fixed project VAL"):
        verify(copied, EVALUATION)


def test_lossless_compressed_predictions_replay(tmp_path):
    import gzip
    copied = tmp_path / "run"
    shutil.copytree(RUN, copied)
    raw = copied / "predictions.jsonl"
    compressed = copied / "predictions.jsonl.gz"
    compressed.write_bytes(gzip.compress(raw.read_bytes(), mtime=0))
    raw.unlink()
    report = verify(copied, EVALUATION)
    assert report["questions"] == 2304 and report["correct"] == 1492
    assert report["prediction_compression"] == "gzip"
