"""Two real LoRA optimizer steps on a random tiny Qwen3 model.

This verifies current training APIs and adapter saving. It is NOT a pretrained
Qwen benchmark, a trained research adapter, or evidence of SQL improvement.
"""
import math
import time

from sqlrl.common import start_run, seed_everything, write_json


def main():
    config = {"kind":"random_tiny_qwen3_api_integration", "benchmark_result":False,
              "seed":42, "max_steps":2, "hidden_size":64, "layers":1, "device":"cpu"}
    directory = start_run("results/cpu_sft_smoke", config)
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
        from trl import SFTConfig, SFTTrainer
        torch.set_num_threads(2)
        seed_everything(42)
        vocab = {token:i for i,token in enumerate(["[UNK]","[PAD]","[EOS]","Question",":","count","people","SQL","SELECT","COUNT","(","*",")","FROM",";","names","name"])}
        raw = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
        raw.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(vocab), hidden_size=64, intermediate_size=128,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            max_position_embeddings=128, pad_token_id=1, eos_token_id=2, tie_word_embeddings=True))
        data = Dataset.from_list([
            {"prompt":"Question: count people SQL:","completion":" SELECT COUNT(*) FROM people; [EOS]"},
            {"prompt":"Question: people names SQL:","completion":" SELECT name FROM people; [EOS]"},
        ] * 2)
        args = SFTConfig(output_dir=str(directory/"checkpoints"), max_steps=2, max_length=64,
            per_device_train_batch_size=1, gradient_accumulation_steps=1, learning_rate=0.001,
            use_cpu=True, bf16=False, fp16=False, gradient_checkpointing=False, packing=False,
            completion_only_loss=True, optim="adamw_torch", report_to="none", save_strategy="no",
            logging_steps=1, seed=42, data_seed=42)
        lora = LoraConfig(r=4, lora_alpha=8, lora_dropout=0, task_type="CAUSAL_LM", target_modules=["q_proj","v_proj"])
        trainer = SFTTrainer(model=model,args=args,train_dataset=data,processing_class=tokenizer,peft_config=lora)
        begin = time.monotonic()
        result = trainer.train()
        elapsed = time.monotonic()-begin
        if not math.isfinite(result.training_loss):
            raise AssertionError("Nonfinite smoke training loss")
        changed = any(torch.count_nonzero(p).item() > 0 for name,p in trainer.model.named_parameters() if "lora_B" in name)
        if not changed:
            raise AssertionError("LoRA B weights did not update")
        trainer.save_model(str(directory/"adapter"))
        adapter_exists = (directory/"adapter/adapter_model.safetensors").is_file()
        if not adapter_exists:
            raise AssertionError("Adapter file not saved")
        write_json(directory/"status.json", {"status":"passed", "kind":config["kind"],"benchmark_result":False,
            "optimizer_steps":trainer.state.global_step,"training_loss":result.training_loss,
            "lora_weights_updated":changed,"adapter_file_saved":adapter_exists,"elapsed_s":elapsed,
            "device":"cpu","training_cost_usd":None})
    except Exception as error:
        write_json(directory/"status.json", {"status":"failed", "benchmark_result":False,
            "error_type":type(error).__name__,"reason":str(error)[:1000]})
        raise


if __name__ == "__main__":
    main()
