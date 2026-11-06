"""Resumable zero-shot Gemini baseline on an owner-confirmed free API project.

No paid fallback, automatic retries, grounding, or extra tools. Billing status
cannot be inferred from an API key: the owner must verify Free tier in AI Studio.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from sqlrl.common import resume_run, sha256, start_run, write_json, write_jsonl
from sqlrl.data.guard import load_records
from sqlrl.eval.prompts import messages

MODEL = "gemini-3.8-flash"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/" + MODEL + ":generateContent"


def validate(config):
    if config.get("model") != MODEL or config.get("endpoint") != ENDPOINT:
        raise ValueError("The free profile permits only its documented Gemini model/endpoint")
    if config.get("budget_usd") != 0 or config.get("input_usd_per_million") != 0 or config.get("output_usd_per_million") != 0:
        raise ValueError("The free profile requires zero prices and zero paid budget")
    if os.getenv("SQLRL_GEMINI_FREE_PROJECT") != "1":
        raise RuntimeError("Verify Free tier with billing disabled in AI Studio, then set SQLRL_GEMINI_FREE_PROJECT=1")
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not configured; no request made")


def request_body(row, config):
    conversation = messages(row)
    return {"systemInstruction":{"parts":[{"text":conversation[0]["content"]}]},
            "contents":[{"role":"user", "parts":[{"text":conversation[-1]["content"]}]}],
            "generationConfig":{"maxOutputTokens":config["max_new_tokens"], "thinkingConfig":{"thinkingLevel":"LOW"}}}


def generate(config, directory, *, resume=False, max_requests=20):
    validate(config)
    rows = load_records(config["data"], expected_split="val")
    if not rows or any(not r.get("schema") or not Path(r["db_path"]).is_file() for r in rows):
        raise ValueError("Complete validation database/schema coverage is required")
    path = directory / "predictions.jsonl"
    saved = [json.loads(line) for line in path.read_text().splitlines() if line] if resume and path.exists() else []
    if [p["id"] for p in saved] != [r["id"] for r in rows[:len(saved)]]:
        raise ValueError("Saved predictions are not the exact validation prefix")
    version_path = directory / "model_version.json"
    pinned = json.loads(version_path.read_text())["model_version"] if version_path.exists() else None
    attempted = 0
    for row in rows[len(saved):]:
        if attempted >= max_requests:
            break
        if attempted:
            time.sleep(config["min_interval_s"])
        attempted += 1
        begin = time.monotonic()
        request = urllib.request.Request(ENDPOINT, data=json.dumps(request_body(row, config)).encode(),
            headers={"x-goog-api-key":os.environ["GEMINI_API_KEY"], "Content-Type":"application/json"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = json.load(response)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            # The pending question is not scored as a model mistake. Resume on a
            # later day after quota recovers; completed predictions remain intact.
            write_json(directory / "status.json", {"status":"paused", "completed_queries":len(saved),
                "required_queries":len(rows), "http_status":getattr(error,"code",None), "reason":"quota_or_transport_error", "paid_fallback":False})
            return
        version = raw.get("modelVersion")
        if not version:
            raise ValueError("Provider response lacks modelVersion; refusing an unversioned benchmark")
        if pinned is None:
            pinned = version
            write_json(version_path, {"model_version":version, "requested_alias":MODEL})
        elif version != pinned:
            write_json(directory / "model_drift_response.json", raw)
            write_json(directory / "status.json", {"status":"paused", "reason":"provider_model_version_changed", "completed_queries":len(saved)})
            return
        candidate = (raw.get("candidates") or [{}])[0]
        parts = candidate.get("content",{}).get("parts",[])
        answer = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        usage = raw.get("usageMetadata", {})
        saved.append({"id":row["id"], "messages":messages(row), "answer":answer,
            "input_tokens":usage.get("promptTokenCount"), "output_tokens":usage.get("candidatesTokenCount"),
            "thinking_tokens":usage.get("thoughtsTokenCount"), "finish_reason":candidate.get("finishReason"),
            "model_version":version, "cost_usd":0.0, "cost_basis":"owner-confirmed free-tier project",
            "latency_s":time.monotonic()-begin, "raw_response":raw})
        write_jsonl(path, saved)
        write_json(directory / "status.json", {"status":"completed" if len(saved)==len(rows) else "paused",
            "completed_queries":len(saved), "required_queries":len(rows), "paid_fallback":False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/free/gemini.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-requests", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.max_requests <= 2304:
        raise ValueError("Choose a positive request cap within the fixed cohort")
    config = json.loads(Path(args.config).read_text())
    validate(config)  # Before creating an experiment or sending any request.
    config["data_sha256"] = sha256(config["data"])
    directory = resume_run(config["output"],config) if args.resume else start_run(config["output"],config)
    generate(config,directory,resume=args.resume,max_requests=args.max_requests)


if __name__ == "__main__":
    main()
