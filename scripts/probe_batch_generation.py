"""Bounded GPU throughput diagnostic; never a benchmark accuracy result."""
import argparse
import hashlib
import json
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--reference", default="results/free_runs/base_zero/predictions.jsonl")
    args = parser.parse_args()
    if not 1 <= args.count <= 16:
        raise ValueError("Diagnostic count must be between 1 and 16")
    output = Path(args.output)
    if output.exists():
        raise ValueError("Preserve the prior diagnostic; choose a new output")
    from sqlrl.common import seed_everything, write_json
    from sqlrl.data.guard import load_records
    from sqlrl.eval.prompts import messages
    from sqlrl.model_loading import model_kwargs
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    config = json.loads(Path("configs/free/base_zero.json").read_text())
    seed_everything(config["seed"])
    rows = load_records(config["data"], expected_split="val")[:args.count]
    reference = {r["id"]:r for r in map(json.loads,Path(args.reference).read_text().splitlines())}
    tokenizer = AutoTokenizer.from_pretrained(config["model"],revision=config["revision"])
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    ids = [tokenizer.apply_chat_template(messages(r),add_generation_prompt=True,tokenize=True,return_dict=True)["input_ids"] for r in rows]
    if any(len(x) > config["max_prompt_tokens"] for x in ids):
        raise ValueError("Diagnostic prompt exceeds original context budget")
    model = AutoModelForCausalLM.from_pretrained(config["model"],revision=config["revision"],**model_kwargs(config))
    model.eval()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos,list) else [eos])
    receipt = {"diagnostic_only":True,"phase3_complete":False,"model":config["model"],"revision":config["revision"],"ids":[r["id"] for r in rows],"runs":[]}
    for batch_size in [4,8]:
        torch.cuda.reset_peak_memory_stats()
        outputs = []
        total_seconds = 0.0
        for start in range(0,len(rows),batch_size):
            current = rows[start:start+batch_size]
            batch_ids = ids[start:start+batch_size]
            inputs = tokenizer.pad({"input_ids":batch_ids,"attention_mask":[[1]*len(x) for x in batch_ids]},padding=True,return_tensors="pt")
            for index, expected in enumerate(ids[start:start+batch_size]):
                actual = inputs["input_ids"][index][inputs["attention_mask"][index].bool()].tolist()
                if actual != expected:
                    raise ValueError("Batched prompt tokens differ from original inputs")
            inputs = inputs.to(model.device)
            torch.cuda.synchronize()
            began = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(**inputs,max_new_tokens=config["max_new_tokens"],do_sample=False,pad_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            seconds = time.monotonic()-began
            total_seconds += seconds
            for index,row in enumerate(current):
                tokens = generated[index,inputs["input_ids"].shape[-1]:].tolist()
                end = next((i+1 for i,t in enumerate(tokens) if t in eos),len(tokens))
                tokens = tokens[:end]
                answer = tokenizer.decode(tokens,skip_special_tokens=True)
                prior = reference.get(row["id"])
                outputs.append({"id":row["id"],"input_tokens":len(ids[start+index]),"output_tokens":len(tokens),"answer_sha256":hashlib.sha256(answer.encode()).hexdigest(),"matches_reference":answer==prior["answer"] if prior and "error" not in prior else None})
        prior_seconds = sum(reference[r["id"]]["latency_s"] for r in rows if r["id"] in reference)
        receipt["runs"].append({"batch_size":batch_size,"questions":len(outputs),"generation_seconds":total_seconds,"amortized_seconds_per_question":total_seconds/len(outputs),"reference_seconds_for_same_questions":prior_seconds,"speedup_vs_saved_serial_latencies":prior_seconds/total_seconds,"exact_answer_matches":sum(r["matches_reference"] is True for r in outputs),"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),"outputs":outputs})
        write_json(output,receipt)
        print(json.dumps(receipt["runs"][-1]),flush=True)
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
