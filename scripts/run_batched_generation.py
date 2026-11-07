"""Frozen-data Qwen generation in recorded, fixed-size inference batches."""
import argparse
import json
import shutil
import time
from pathlib import Path

from sqlrl.common import seed_everything, sha256, start_run, resume_run, write_json, write_jsonl
from sqlrl.data.guard import load_records
from sqlrl.eval.generate import select_examples
from sqlrl.eval.prompts import messages


def batch_size_from_config(config):
    value = config.get("generation_batch_size", 8)
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 16:
        raise ValueError("Inference batch size must be an integer between 1 and 16")
    return value


def validate_prefix(outputs, rows):
    if [r["id"] for r in outputs] != [r["id"] for r in rows[:len(outputs)]]:
        raise ValueError("Saved predictions do not match the exact validation prefix")
    return outputs


def trim_eos(tokens, eos_ids):
    end = next((index+1 for index, token in enumerate(tokens) if token in eos_ids), len(tokens))
    return tokens[:end]


def generate(config, directory, *, resume=False):
    if config.get("backend") != "transformers" or not shutil.which("nvidia-smi"):
        raise RuntimeError("This batched profile requires a confirmed local NVIDIA GPU")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sqlrl.model_loading import model_kwargs
    if not torch.cuda.is_available() or not config.get("revision"):
        raise RuntimeError("A CUDA GPU and pinned model revision are required")
    batch_size = batch_size_from_config(config)
    rows = load_records(config["data"], expected_split="val")
    if len(rows) != 2304 or any(not r.get("schema") or not Path(r["db_path"]).is_file() for r in rows):
        raise RuntimeError("The full frozen validation set, schemas and databases are required")
    examples = []
    if config.get("few_shot", 0):
        train = load_records(config["train_data"], expected_split="train")
        if {(r["dataset"],r["db_id"]) for r in rows} & {(r["dataset"],r["db_id"]) for r in train}:
            raise ValueError("Few-shot training databases overlap validation")
        examples = select_examples(train,config["few_shot"],config["seed"])
    write_json(directory/"few_shot_ids.json",[r["id"] for r in examples])
    outputs = []
    if resume and (directory/"predictions.jsonl").exists():
        outputs = [json.loads(line) for line in (directory/"predictions.jsonl").read_text().splitlines() if line]
        validate_prefix(outputs,rows)
    tokenizer = AutoTokenizer.from_pretrained(config["model"],revision=config["revision"])
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["model"],revision=config["revision"],**model_kwargs(config))
    if config.get("adapter"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model,config["adapter"])
    model.eval()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos,list) else [eos])
    for offset in range(len(outputs),len(rows),batch_size):
        batch = rows[offset:offset+batch_size]
        began = time.monotonic()
        pending, valid = [], []
        for row in batch:
            prompt = messages(row,examples)
            item = {"id":row["id"],"messages":prompt,"answer":"","input_tokens":None,"output_tokens":None,
                    "cost_usd":None,"latency_s":None,"batch_index":offset//batch_size,"requested_batch_size":batch_size}
            try:
                encoded = tokenizer.apply_chat_template(prompt,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors="pt")
                ids = encoded["input_ids"][0].tolist()
                if len(ids) > config["max_prompt_tokens"]:
                    raise ValueError("Prompt exceeds preregistered context budget")
                valid.append((len(pending),ids))
                item["input_tokens"] = len(ids)
            except Exception as error:
                item["error"] = type(error).__name__+": "+str(error)[:300]
            pending.append(item)
        if valid:
            batch_ids = [ids for _,ids in valid]
            inputs = tokenizer.pad({"input_ids":batch_ids,"attention_mask":[[1]*len(ids) for ids in batch_ids]},padding=True,return_tensors="pt").to(model.device)
            torch.cuda.synchronize()
            # A CUDA failure stops this batch, preserving the last atomic prefix.
            with torch.inference_mode():
                generated = model.generate(**inputs,max_new_tokens=config["max_new_tokens"],do_sample=False,pad_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            for index,(position,_) in enumerate(valid):
                tokens = trim_eos(generated[index,inputs["input_ids"].shape[-1]:].tolist(),eos_ids)
                pending[position].update(answer=tokenizer.decode(tokens,skip_special_tokens=True),output_tokens=len(tokens))
        seconds = time.monotonic()-began
        for item in pending:
            item.update(latency_s=seconds,actual_batch_size=len(valid),amortized_batch_seconds=seconds/len(pending))
        outputs.extend(pending)
        write_jsonl(directory/"predictions.jsonl",outputs)
        print(json.dumps({"saved_predictions":len(outputs),"total":len(rows),"batch_seconds":round(seconds,3),
                          "generation_errors":sum("error" in r for r in outputs)}),flush=True)
    write_json(directory/"status.json",{"status":"completed","queries":len(outputs),"api_budget_debited_usd":0,
        "generation_failures":sum("error" in r for r in outputs),"generation_batch_size":batch_size,
        "profile_id":config["profile_id"],"latency_definition":"Each request records its shared batch wall time; amortized throughput is a separate field"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    batch_size_from_config(config)
    config["data_sha256"] = sha256(config["data"])
    if config.get("train_data"):
        config["train_data_sha256"] = sha256(config["train_data"])
    config["generation_script_sha256"] = sha256(__file__)
    directory = resume_run(config["output"],config) if args.resume else start_run(config["output"],config)
    seed_everything(config["seed"])
    try:
        generate(config,directory,resume=args.resume)
    except Exception as error:
        write_json(directory/"status.json",{"status":"blocked_or_failed","error_type":type(error).__name__,"reason":str(error)[:500]})
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
