"""Read official pinned source signatures without importing a GPU stack."""
import ast
import hashlib
import urllib.request

from sqlrl.common import write_json

SOURCES = {
    "SFTConfig":("https://raw.githubusercontent.com/huggingface/trl/v1.17.1/trl/trainer/sft_config.py", ["max_length","completion_only_loss","packing"]),
    "SFTTrainer":("https://raw.githubusercontent.com/huggingface/trl/v1.17.1/trl/trainer/sft_trainer.py", ["processing_class","peft_config","train_dataset"]),
    "GRPOConfig":("https://raw.githubusercontent.com/huggingface/trl/v1.17.1/trl/trainer/grpo_config.py", ["use_vllm","num_generations","beta"]),
    "LoraConfig":("https://raw.githubusercontent.com/huggingface/peft/v0.21.2/src/peft/tuners/lora/config.py", ["r","lora_alpha","target_modules"]),
}


def main():
    checks = []
    for name, (url, required) in SOURCES.items():
        try:
            data = urllib.request.urlopen(url, timeout=45).read()
            tree = ast.parse(data)
            names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
            names |= {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}
            found = {field:field in names for field in required}
            checks.append({"class":name,"url":url,"sha256":hashlib.sha256(data).hexdigest(),"required_fields_found":found,"passed":all(found.values())})
        except Exception as error:
            checks.append({"class":name,"url":url,"passed":False,"error":str(error)})
    write_json("results/upstream_api_checks.json", {"checks":checks,"gpu_integration_tested":False})
    print(checks)
    if not all(check["passed"] for check in checks):
        raise SystemExit("Upstream API verification did not pass")


if __name__ == "__main__":
    main()
