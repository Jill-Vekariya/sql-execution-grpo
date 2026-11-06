"""Run the frozen Phase 2–3 experiment on an already connected CUDA machine.

Default mode only prepares configs and reports blockers. --execute never rents
hardware, changes the training pool, retries paid requests, or opens final data.
Use a new output directory for every attempt so failed runs remain evidence.
"""
import argparse
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

from sqlrl.common import sha256, start_run, write_json
from sqlrl.data import question_digest
from sqlrl.data.guard import load_records
from sqlrl.eval.statistics import paired_comparison


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def nonnegative(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def plan_configs(root, gpu_rate, frontier_override=None):
    if gpu_rate is not None and not nonnegative(gpu_rate):
        raise ValueError("GPU hourly price must be finite and nonnegative; use 0 only for known free compute")
    configs = {name: read(f"configs/{name}.json") for name in ["base_zero", "base_few", "sft", "sft_eval", "frontier"]}
    allowed = {"model", "revision", "endpoint", "api_key_env", "budget_usd", "input_usd_per_million", "output_usd_per_million"}
    for key, value in (frontier_override or {}).items():
        if key not in allowed:
            raise ValueError(f"Frontier override cannot change protocol field: {key}")
        configs["frontier"][key] = value
    for name, config in configs.items():
        config["output"] = str(root / name)
        config["gpu_price_usd_per_hour"] = gpu_rate
    configs["sft_eval"]["adapter"] = str(root / "sft" / "adapter")
    return configs


def data_blockers(config):
    problems, records = [], []
    try:
        for split, path in [("train", config["train_data"]), ("val", config["data"])]:
            rows = load_records(path, expected_split=split)
            manifest = Path("data/manifests") / ("train_eligible.jsonl" if split == "train" else "val.jsonl")
            expected = [json.loads(line) for line in manifest.read_text().splitlines() if line]
            actual = [{"id":r["id"], "dataset":r["dataset"], "db_id":r["db_id"], "question_sha256":question_digest(r["question"])} for r in rows]
            if not rows or actual != expected:
                problems.append(f"{split} records differ from the frozen manifest")
            records.extend(rows)
        eligibility = read("results/training_eligibility.json")
        if sha256(config["train_data"]) != eligibility["processed_train_sha256"]:
            problems.append("Frozen eligible training bytes changed")
        train_dbs = {(r["dataset"], r["db_id"]) for r in records if r["split"] == "train"}
        val_dbs = {(r["dataset"], r["db_id"]) for r in records if r["split"] == "val"}
        if train_dbs & val_dbs:
            problems.append("Train/validation databases overlap")
        missing = {r["db_path"] for r in records if not Path(r["db_path"]).is_file() or not r.get("schema")}
        if missing:
            problems.append(f"Missing database/schema coverage: {len(missing)} database paths")
        # Match original labels as well as question IDs. No official dev is read.
        audit = read("results/data_audit.json")
        sources = {}
        for dataset in {r["dataset"] for r in records}:
            path = Path("data/raw") / dataset / "train.json"
            if sha256(path) != audit["sources"][dataset]["source_sha256"]:
                problems.append(f"{dataset} source annotation hash changed")
            sources[dataset] = read(path)
        for row in records:
            original = sources[row["dataset"]][int(row["id"].split(":")[1])]
            if (row["question"], row["sql"], row.get("evidence", "")) != (original["question"], original.get("SQL", original.get("query")), original.get("evidence", "")):
                problems.append(f"Original annotation changed: {row['id']}")
                break
    except (OSError, ValueError, KeyError, IndexError) as error:
        problems.append("Data preflight: " + str(error))
    return problems, records


def readiness(configs):
    problems, rows = data_blockers(configs["base_zero"])
    gpu = None
    if sys.platform != "linux":
        problems.append("Linux/WSL is required for executor resource limits")
    if configs["sft"]["gpu_price_usd_per_hour"] is None:
        problems.append("Provide the actual GPU hourly rate, including 0 if known free")
    required = read("configs/protocol.json")["environment"]
    required.update({"datasets":"5.1.0", "accelerate":"1.15.0"})
    for name in ["torch", "transformers", "peft", "trl", "datasets", "accelerate"]:
        try:
            if importlib.metadata.version(name).split("+")[0] != required[name]:
                problems.append(f"Install the locked {name}=={required[name]}")
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"Missing training dependency: {name}=={required[name]}")
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            problems.append("Expose exactly one CUDA GPU")
        else:
            device = torch.cuda.get_device_properties(0)
            gpu = {"name":device.name, "memory_bytes":device.total_memory, "compute_capability":[device.major,device.minor]}
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                problems.append("The frozen BF16 training configuration needs native BF16 support")
    except ImportError:
        problems.append("CUDA availability cannot be checked without the locked torch package")
    if not shutil.which("nvidia-smi"):
        problems.append("nvidia-smi is required by the frozen generation/training entry points")
    api = configs["frontier"]
    endpoint = urlsplit(api.get("endpoint") or "")
    if not api.get("model") or "REPLACE" in str(api.get("model")).upper() or "REPLACE" in endpoint.netloc.upper() or endpoint.scheme != "https" or not endpoint.netloc or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        problems.append("Configure a frontier model and HTTPS endpoint without embedded credentials")
    if not os.environ.get(api.get("api_key_env", "SQLRL_API_KEY")):
        problems.append("Frontier API credential is not set in the execution environment")
    if any(not nonnegative(api.get(k)) or api[k] <= 0 for k in ["budget_usd", "input_usd_per_million", "output_usd_per_million"]):
        problems.append("Configure a positive frontier spending cap and verified token prices")
    return {"ready":not problems, "blockers":problems, "gpu":gpu, "train_examples":sum(r["split"] == "train" for r in rows), "val_examples":sum(r["split"] == "val" for r in rows), "final_test_accessed":False}


def cost_per_point(cost, improvement):
    if cost is None or improvement is None or improvement <= 0:
        return None
    if not nonnegative(cost):
        raise ValueError("Cost must be finite and nonnegative")
    return cost / (100 * improvement)


def report(root, stage_costs):
    summaries, results = {}, {}
    for name in ["base_zero", "base_few", "sft_eval", "frontier"]:
        evaluation = root / (name + "_evaluation")
        summaries[name] = read(evaluation / "summary.json")
        rows = [json.loads(line) for line in (evaluation / "per_question.jsonl").read_text().splitlines() if line]
        results[name] = {r["id"]:r for r in rows}
        if len(results[name]) != len(rows) or set(results[name]) != set(results["base_zero"]):
            raise ValueError("Every comparison must cover the same unique validation IDs")
    keys = sorted(results["base_zero"])
    base = results["base_zero"]
    comparisons = {}
    for name in ["base_few", "sft_eval", "frontier"]:
        candidate = results[name]
        if any((candidate[k]["dataset"],candidate[k]["db_id"]) != (base[k]["dataset"],base[k]["db_id"]) for k in keys):
            raise ValueError("Paired database labels differ")
        comparisons[name] = paired_comparison([int(base[k]["correct"]) for k in keys], [int(candidate[k]["correct"]) for k in keys], [base[k]["dataset"] + ":" + base[k]["db_id"] for k in keys])
    sft_cost = stage_costs["sft"] + stage_costs["sft_eval"]
    api_cost = summaries["frontier"]["total_cost_usd"]
    costs = {"base_zero":stage_costs["base_zero"], "base_few":stage_costs["base_few"], "sft_eval":sft_cost, "frontier":api_cost}
    value = {"status":"completed", "summaries":summaries, "measured_costs_usd":costs, "paired_vs_base_zero":comparisons,
             "sft_cost_per_accuracy_percentage_point_vs_base":cost_per_point(sft_cost, comparisons["sft_eval"]["delta"]),
             "cost_definition":"Per-stage process wall time times declared GPU rate; SFT includes training and its validation generation. Frontier cost uses provider-reported tokens. Setup, downloads, idle rented time outside these stages, storage and taxes are excluded; this is not a provider invoice.",
             "nonpositive_gain_policy":"No cost-per-improvement ratio is reported when improvement is zero or negative", "rl_trained":False, "final_test_accessed":False}
    write_json(root / "comparison.json", value)
    lines = ["# Measured Phase 2–3 validation results", "", "| System | Execution accuracy | 95% CI | Measured USD |", "|---|---:|---|---:|"]
    for name, summary in summaries.items():
        ci = summary["execution_accuracy_ci95"]
        cost = costs[name]
        lines.append(f"| {name} | {100*summary['execution_accuracy']:.2f}% | {100*ci[0]:.2f}–{100*ci[1]:.2f}% | {cost:.4f} |" if cost is not None else f"| {name} | {100*summary['execution_accuracy']:.2f}% | {100*ci[0]:.2f}–{100*ci[1]:.2f}% | Unknown |")
    lines += ["", value["cost_definition"], "", "RL has not been trained. Official final-test data remains sealed.", ""]
    (root / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return value


def execute_plan(root, configs):
    # Verify the actual static database bytes immediately before model work.
    audit = read("results/data_audit.json")
    _, rows = data_blockers(configs["base_zero"])
    for dataset, db_id, path in sorted({(r["dataset"],r["db_id"],r["db_path"]) for r in rows}):
        if sha256(path) != audit["sources"][dataset]["database_sha256"][db_id]:
            raise ValueError(f"Frozen database bytes changed: {dataset}:{db_id}")
    costs = {}
    stages = []

    def run(name, arguments, gpu=False):
        begin = time.monotonic()
        with (root / (name + ".log")).open("w") as stream:
            process = subprocess.run([sys.executable, *arguments], stdout=stream, stderr=subprocess.STDOUT)
        seconds = time.monotonic() - begin
        stages.append({"name":name, "returncode":process.returncode, "elapsed_s":seconds, "gpu_cost_usd":seconds / 3600 * configs["sft"]["gpu_price_usd_per_hour"] if gpu else None})
        write_json(root / "stages.json", stages)
        if process.returncode:
            raise RuntimeError(f"Stage {name} failed; see its preserved log")
        if gpu:
            costs[name] = stages[-1]["gpu_cost_usd"]

    for name in ["base_zero", "base_few", "sft", "sft_eval", "frontier"]:
        module = "sqlrl.train_sft.run" if name == "sft" else "sqlrl.eval.generate"
        run(name, ["-m", module, "--config", str(root / "configs" / (name + ".json"))], gpu=name != "frontier")
        if name != "sft":
            run(name + "_evaluation", ["-m", "sqlrl.eval.run", "--data", configs[name]["data"], "--predictions", str(root / name / "predictions.jsonl"), "--output", str(root / (name + "_evaluation"))])
    return report(root, costs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New evidence directory; existing runs are never overwritten")
    parser.add_argument("--gpu-usd-per-hour", type=float, default=None)
    parser.add_argument("--frontier-config", help="JSON overrides containing provider/model/prices/budget only; no API key")
    parser.add_argument("--execute", action="store_true", help="Run on the connected, preconfigured GPU only after every preflight passes")
    args = parser.parse_args()
    root = Path(args.output)
    configs = plan_configs(root, args.gpu_usd_per_hour, read(args.frontier_config) if args.frontier_config else None)
    start_run(root, {"phase_scope":[0,1,2,3], "execute_requested":args.execute, "gpu_usd_per_hour":args.gpu_usd_per_hour})
    for name, config in configs.items():
        write_json(root / "configs" / (name + ".json"), config)
    ready = readiness(configs)
    write_json(root / "readiness.json", ready)
    if not args.execute or not ready["ready"]:
        write_json(root / "status.json", {"status":"prepared" if ready["ready"] else "blocked", "model_runs_started":False, "blockers":ready["blockers"]})
        print(json.dumps(ready, indent=2))
        return 2 if args.execute else 0
    try:
        execute_plan(root, configs)
    except Exception as error:
        write_json(root / "status.json", {"status":"failed", "reason":str(error), "completed":False})
        raise
    write_json(root / "status.json", {"status":"completed", "completed":["base_zero", "base_few", "sft", "sft_eval", "frontier", "paired_comparisons", "measured_cost_report"], "rl_trained":False, "final_test_accessed":False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
