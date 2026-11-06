import io
import json
from types import SimpleNamespace
import urllib.error

import pytest

from sqlrl.common import resume_run, start_run, write_jsonl
from sqlrl.eval import free_gemini
from sqlrl.model_loading import model_kwargs


def api_config(db, tmp_path, monkeypatch):
    monkeypatch.setenv("SQLRL_GEMINI_FREE_PROJECT","1")
    monkeypatch.setenv("GEMINI_API_KEY","synthetic-test-secret")
    rows=[{"id":str(i),"dataset":"fixture","db_id":"one","split":"val","source_split":"train","question":"Count rows","sql":"SECRET GOLD","schema":"CREATE TABLE people(id INT);","db_path":str(db)} for i in range(2)]
    path=tmp_path/"val.jsonl";write_jsonl(path,rows)
    directory=tmp_path/"output";directory.mkdir()
    config={"model":free_gemini.MODEL,"endpoint":free_gemini.ENDPOINT,"data":str(path),"budget_usd":0,"input_usd_per_million":0,"output_usd_per_million":0,"max_new_tokens":512,"min_interval_s":0}
    return rows,config,directory


def response(version="version-one"):
    return io.BytesIO(json.dumps({"modelVersion":version,"candidates":[{"content":{"parts":[{"thought":True,"text":"not answer"},{"text":"```sql\nSELECT COUNT(*) FROM people\n```"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":20,"candidatesTokenCount":10}}).encode())


def test_free_api_requires_account_owner_confirmation(monkeypatch):
    monkeypatch.delenv("SQLRL_GEMINI_FREE_PROJECT",raising=False)
    config={"model":free_gemini.MODEL,"endpoint":free_gemini.ENDPOINT,"budget_usd":0,"input_usd_per_million":0,"output_usd_per_million":0}
    with pytest.raises(RuntimeError,match="Verify Free tier"):
        free_gemini.validate(config)
    config["budget_usd"]=1
    with pytest.raises(ValueError,match="zero prices"):
        free_gemini.validate(config)


def test_quota_pause_preserves_prefix_and_resume_does_not_repeat_it(db,tmp_path,monkeypatch):
    rows,config,directory=api_config(db,tmp_path,monkeypatch)
    calls=[]
    def limited(request,**kwargs):
        calls.append(json.loads(request.data))
        if len(calls)==2:
            raise urllib.error.HTTPError(free_gemini.ENDPOINT,429,"Quota",{},None)
        return response()
    monkeypatch.setattr(free_gemini.urllib.request,"urlopen",limited)
    free_gemini.generate(config,directory,max_requests=2)
    saved=[json.loads(line) for line in (directory/"predictions.jsonl").read_text().splitlines()]
    assert [r["id"] for r in saved]==["0"]
    assert json.loads((directory/"status.json").read_text())["http_status"]==429
    assert "SECRET GOLD" not in json.dumps(calls)
    assert "not answer" not in saved[0]["answer"]
    assert "synthetic-test-secret" not in (directory/"predictions.jsonl").read_text()
    monkeypatch.setattr(free_gemini.urllib.request,"urlopen",lambda *a,**k:response())
    free_gemini.generate(config,directory,resume=True,max_requests=1)
    assert len((directory/"predictions.jsonl").read_text().splitlines())==2
    assert json.loads((directory/"status.json").read_text())["status"]=="completed"


def test_provider_model_change_does_not_mix_results(db,tmp_path,monkeypatch):
    rows,config,directory=api_config(db,tmp_path,monkeypatch)
    versions=iter(["version-one","version-two"])
    monkeypatch.setattr(free_gemini.urllib.request,"urlopen",lambda *a,**k:response(next(versions)))
    free_gemini.generate(config,directory,max_requests=2)
    assert len((directory/"predictions.jsonl").read_text().splitlines())==1
    assert json.loads((directory/"status.json").read_text())["reason"]=="provider_model_version_changed"


def test_resume_refuses_changed_data_or_source(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    directory=start_run("run",{"data_sha256":"original"})
    resume_run(directory,{"data_sha256":"original"})
    with pytest.raises(ValueError,match="configuration"):
        resume_run(directory,{"data_sha256":"changed"})
    (tmp_path/"src").mkdir();(tmp_path/"src/changed.py").write_text("# changed source")
    with pytest.raises(ValueError,match="source files"):
        resume_run(directory,{"data_sha256":"original"})


def test_interrupted_checkpoint_write_preserves_previous_predictions(tmp_path,monkeypatch):
    path=tmp_path/"predictions.jsonl";write_jsonl(path,[{"id":"saved"}])
    original=path.read_bytes()
    def interrupted(*args):raise OSError("simulated interruption")
    monkeypatch.setattr("sqlrl.common.os.replace",interrupted)
    with pytest.raises(OSError):write_jsonl(path,[{"id":"new"}])
    assert path.read_bytes()==original


def test_nf4_profile_uses_the_same_fp16_base_for_sft_and_inference(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(float16="FP16",bfloat16="BF16"))
    monkeypatch.setitem(sys.modules,"transformers",SimpleNamespace(BitsAndBytesConfig=lambda **kwargs:kwargs))
    config={"precision":"fp16","quantization":"nf4"}
    train=model_kwargs(config,training=True);inference=model_kwargs(config)
    assert train==inference
    assert train["device_map"]=={"":0}
    assert train["quantization_config"]["bnb_4bit_compute_dtype"]=="FP16"
    assert train["quantization_config"]["bnb_4bit_quant_type"]=="nf4"
