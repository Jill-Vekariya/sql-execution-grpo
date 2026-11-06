"""LoRA SFT using TRL's verified prompt-completion API.

The final epoch is the preregistered SFT baseline. Validation loss is logged;
it is never presented as execution accuracy. No final-test loader exists here.
"""
import argparse
import gc
import json
import shutil
import time
from pathlib import Path

from sqlrl.common import seed_everything, sha256, start_run, write_json
from sqlrl.data.guard import load_records
from sqlrl.eval.prompts import messages, count_chat_tokens


def train(config, directory):
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("No NVIDIA CUDA runtime detected; Qwen SFT was not started and no adapter exists")
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; full Qwen SFT was not started")
    if torch.cuda.device_count()!=1:
        raise RuntimeError("This configuration is for one CUDA GPU; expose one device with CUDA_VISIBLE_DEVICES")
    if not config.get("revision"):
        raise RuntimeError("Model revision must be an immutable Hub commit")
    rows = load_records(config["train_data"], expected_split="train")
    validation = load_records(config["val_data"], expected_split="val")
    if {(r["dataset"],r["db_id"]) for r in rows} & {(r["dataset"],r["db_id"]) for r in validation}:
        raise ValueError("Training/validation databases overlap")
    if any(not r.get("schema") or not Path(r["db_path"]).is_file() for r in rows + validation):
        raise RuntimeError("The declared corpus is incomplete; refusing silent training-data changes")
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    seed_everything(config["seed"])
    excluded = []

    def convert(records):
        converted = []
        for row in records:
            prompt = messages(row)
            completion = [{"role":"assistant", "content":"```sql\n" + row["sql"] + "\n```"}]
            length = count_chat_tokens(tokenizer,prompt+completion)
            if length > config["max_length"]:
                excluded.append({"id":row["id"], "split":row["split"], "tokens":length, "reason":"context_limit"})
                continue
            converted.append({"prompt":prompt, "completion":completion})
        if not converted:
            raise ValueError("No examples fit the context budget")
        return Dataset.from_list(converted)

    train_dataset, val_dataset = convert(rows), convert(validation)
    write_json(directory / "preprocessing.json", {"train_examples":len(train_dataset), "val_examples":len(val_dataset), "excluded":excluded})
    model = AutoModelForCausalLM.from_pretrained(config["model"], revision=config["revision"], dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    args = SFTConfig(
        output_dir=str(directory / "checkpoints"), num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"], per_device_train_batch_size=1,
        per_device_eval_batch_size=1, gradient_accumulation_steps=config["gradient_accumulation_steps"],
        warmup_steps=20, weight_decay=0.01, max_length=config["max_length"],
        completion_only_loss=True, packing=False, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False}, optim="adamw_torch",
        seed=config["seed"], data_seed=config["seed"], logging_steps=10,
        save_strategy="steps", save_steps=100, save_total_limit=2,
        eval_strategy="steps", eval_steps=100, report_to="none", load_best_model_at_end=False,
    )

    class JsonLog(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            with (directory / "training_metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(logs or {}, step=state.global_step)) + "\n")

    trainer = SFTTrainer(model=model, args=args, train_dataset=train_dataset,
                         eval_dataset=val_dataset, processing_class=tokenizer, peft_config=lora,
                         callbacks=[JsonLog()])
    started = time.monotonic()
    trainer.train()
    elapsed = time.monotonic()-started
    adapter = directory / "adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    price = config.get("gpu_price_usd_per_hour")
    write_json(directory / "status.json", {"status":"trained", "selection_rule":"final_epoch_preregistered",
        "gpu_hours":elapsed/3600, "gpu_count":1, "gpu_price_usd_per_hour":price,
        "training_cost_usd":elapsed/3600*price if price is not None else None,
        "seed":config["seed"], "model":config["model"], "revision":config["revision"],
        "validation_execution_evaluation":"pending separate generation/evaluation commands"})
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft.json")
    args = parser.parse_args()
    config = json.load(open(args.config))
    for key in ["train_data", "val_data"]:
        config[key + "_sha256"] = sha256(config[key]) if Path(config[key]).exists() else None
    directory = start_run(config["output"], config)
    try:
        train(config, directory)
    except Exception as error:
        write_json(directory / "status.json", {"status":"blocked_or_failed", "error_type":type(error).__name__, "reason":str(error)[:500]})
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
