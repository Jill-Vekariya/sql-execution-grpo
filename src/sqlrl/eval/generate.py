"""Generation with complete raw-output logging and fail-closed preflights.

HTTP backend works with a separately hosted vLLM or a frontier endpoint. Prices
and a hard budget must be explicit before any paid API request is made.
"""
import argparse
import json
import os
import shutil
import time
import urllib.error
import urllib.request

from sqlrl.common import seed_everything, sha256, start_run, resume_run, write_json, write_jsonl
from sqlrl.data.guard import load_records
from sqlrl.eval.prompts import messages


def select_examples(rows, n, seed):
    import random
    choices = sorted(rows, key=lambda r:r["id"])
    random.Random(seed).shuffle(choices)
    selected, databases = [], set()
    for row in choices:
        key = (row["dataset"], row["db_id"])
        if key not in databases:
            selected.append(row)
            databases.add(key)
        if len(selected) >= n:
            break
    if len(selected) != n:
        raise ValueError("Not enough distinct training databases for few-shot examples")
    return selected


def generate(config, directory, *, resume=False):
    if config["backend"] == "transformers" and not shutil.which("nvidia-smi"):
        raise RuntimeError("No NVIDIA CUDA runtime detected; benchmark generation was not started")
    if config["backend"] == "http" and (not config.get("model") or not config.get("endpoint")):
        raise RuntimeError("Frontier/API model and endpoint have not been configured; no API request made")
    rows = load_records(config["data"], expected_split="val")
    if not rows or any(not r.get("schema") or not os.path.isfile(r["db_path"]) for r in rows):
        raise RuntimeError("Full validation schemas and databases are required; no silent subset evaluation")
    examples = []
    if config.get("few_shot", 0):
        train = load_records(config["train_data"], expected_split="train")
        val_databases = {(r["dataset"],r["db_id"]) for r in rows}
        if val_databases & {(r["dataset"],r["db_id"]) for r in train}:
            raise ValueError("Few-shot source databases overlap validation")
        examples = select_examples(train, config["few_shot"], config["seed"])
    write_json(directory / "few_shot_ids.json", [r["id"] for r in examples])
    backend = config["backend"]
    if backend == "transformers":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen benchmark generation requires a CUDA GPU in this configuration")
        if not config.get("revision"):
            raise RuntimeError("Pin an immutable model revision first")
        tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
        from sqlrl.model_loading import model_kwargs
        model = AutoModelForCausalLM.from_pretrained(config["model"], revision=config["revision"], **model_kwargs(config))
        if config.get("adapter"):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, config["adapter"])
        model.eval()
    elif backend == "http":
        if not config.get("model") or not config.get("endpoint"):
            raise RuntimeError("API model and endpoint have not been configured")
        key = os.environ.get(config.get("api_key_env", "SQLRL_API_KEY"))
        if not key:
            raise RuntimeError("API credentials are not configured")
        prices = [config.get("input_usd_per_million"), config.get("output_usd_per_million")]
        if config.get("budget_usd", 0) <= 0 or any(not isinstance(p, (int,float)) or p <= 0 for p in prices):
            raise RuntimeError("An explicit API budget and verified token prices are required")
    else:
        raise ValueError("Unknown backend")
    outputs, spent = [], 0.0
    if resume and (directory / "predictions.jsonl").exists():
        if backend != "transformers":
            raise ValueError("Paid HTTP runs cannot be automatically resumed")
        outputs = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines() if line]
        if [p["id"] for p in outputs] != [r["id"] for r in rows[:len(outputs)]]:
            raise ValueError("Saved predictions do not match the evaluation prefix")
    completed_ids = {p["id"] for p in outputs}
    for row in rows:
        if row["id"] in completed_ids:
            continue
        prompt = messages(row, examples)
        begin = time.monotonic()
        output = {"id":row["id"], "messages":prompt, "answer":"", "input_tokens":None,
                  "output_tokens":None, "cost_usd":None, "latency_s":None}
        try:
            if backend == "transformers":
                inputs = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
                if inputs["input_ids"].shape[-1] > config["max_prompt_tokens"]:
                    raise ValueError("Prompt exceeds preregistered context budget")
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=config["max_new_tokens"], do_sample=False, pad_token_id=tokenizer.eos_token_id)
                tokens = generated[0, inputs["input_ids"].shape[-1]:]
                output.update(answer=tokenizer.decode(tokens, skip_special_tokens=True), input_tokens=inputs["input_ids"].shape[-1], output_tokens=len(tokens))
            else:
                # Conservative pre-call bound: at most one token per UTF-8 byte
                # plus chat overhead. Reject if the remaining budget cannot cover it.
                max_input = sum(len(m["content"].encode()) for m in prompt) + 1024
                upper = (max_input*config["input_usd_per_million"] + config["max_new_tokens"]*config["output_usd_per_million"])/1e6
                if spent + upper > config["budget_usd"]:
                    raise RuntimeError("API budget exhausted before request")
                body = {"model":config["model"], "messages":prompt, "temperature":0, "max_tokens":config["max_new_tokens"]}
                request = urllib.request.Request(config["endpoint"], data=json.dumps(body).encode(), headers={"Authorization":"Bearer " + key, "Content-Type":"application/json"})
                # No retries: a network error can still incur provider charges.
                try:
                    with urllib.request.urlopen(request, timeout=120) as response:
                        raw = json.load(response)
                except Exception:
                    spent += upper
                    raise
                output["raw_response"] = raw
                usage = raw.get("usage", {})
                output.update(answer=raw["choices"][0]["message"]["content"] or "", input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"))
                if output["input_tokens"] is None or output["output_tokens"] is None:
                    spent += upper
                    raise RuntimeError("Missing usage; budget charged at conservative bound")
                output["cost_usd"] = (output["input_tokens"]*config["input_usd_per_million"] + output["output_tokens"]*config["output_usd_per_million"])/1e6
                spent += output["cost_usd"]
        except Exception as error:
            output["error"] = type(error).__name__ + ": " + str(error)[:300]
        output["latency_s"] = time.monotonic()-begin
        outputs.append(output)
        # Flush after every generation so interruption preserves evidence.
        write_jsonl(directory / "predictions.jsonl", outputs)
        if output.get("error") and backend == "http":
            raise RuntimeError("HTTP run stopped after first error; partial output preserved")
    write_json(directory / "status.json", {"status":"completed", "queries":len(outputs), "api_budget_debited_usd":spent,
               "generation_failures":sum("error" in x for x in outputs)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.load(open(args.config))
    config["data_sha256"] = sha256(config["data"]) if os.path.isfile(config["data"]) else None
    if config.get("train_data"):
        config["train_data_sha256"] = sha256(config["train_data"]) if os.path.isfile(config["train_data"]) else None
    directory = resume_run(config["output"], config) if args.resume else start_run(config["output"], config)
    seed_everything(config["seed"])
    try:
        generate(config, directory, resume=args.resume)
    except Exception as error:
        write_json(directory / "status.json", {"status":"blocked_or_failed", "error_type":type(error).__name__, "reason":str(error)[:500]})
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
