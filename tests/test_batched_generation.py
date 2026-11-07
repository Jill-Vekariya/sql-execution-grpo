"""Resume order and token accounting for the optional batched profile."""
import pytest
from scripts.run_batched_generation import batch_size_from_config, trim_eos, validate_prefix

def test_multiple_stop_ids_preserve_first_stop_token():
    assert trim_eos([8,9,2,4,3,3],{2,3}) == [8,9,2]
    assert trim_eos([8,9,3,4],{2,3}) == [8,9,3]

def test_capped_output_without_stop_is_preserved():
    tokens = list(range(512))
    assert trim_eos(tokens,{999}) == tokens
    assert trim_eos([],{999}) == []

@pytest.mark.parametrize("saved",[
    [{"id":"b"}],
    [{"id":"a"},{"id":"a"}],
    [{"id":"a"},{"id":"b"},{"id":"extra"}],
])
def test_resume_rejects_skipped_duplicate_or_extra_rows(saved):
    with pytest.raises(ValueError,match="exact validation prefix"):
        validate_prefix(saved,[{"id":"a"},{"id":"b"}])

def test_resume_accepts_only_an_ordered_prefix():
    rows=[{"id":"a"},{"id":"b"}]
    saved=[{"id":"a","answer":"SELECT 1"}]
    assert validate_prefix(saved,rows) is saved
    assert validate_prefix([],rows) == []

@pytest.mark.parametrize("value",[0,17,-1,True,2.5,"8"])
def test_invalid_batch_size_cannot_silently_change_execution(value):
    with pytest.raises(ValueError,match="batch size"):
        batch_size_from_config({"generation_batch_size":value})

def test_declared_batch_size_is_retained():
    assert batch_size_from_config({"generation_batch_size":8}) == 8

def test_batch_profile_retains_all_frozen_experiment_settings():
    import json
    from pathlib import Path
    for name in ["base_zero","base_few","sft","sft_eval","gemini"]:
        serial=json.loads(Path("configs/free",name+".json").read_text())
        batched=json.loads(Path("configs/free_batch8_v2",name+".json").read_text())
        operational={"profile_id","output","adapter","generation_batch_size"}
        assert {k:v for k,v in serial.items() if k not in operational} == {k:v for k,v in batched.items() if k not in operational}
        assert batched["profile_id"] == "student_free_nf4_fp16_batch8_v2"
        if name in {"base_zero","base_few","sft_eval"}:
            assert batched["generation_batch_size"] == 8
