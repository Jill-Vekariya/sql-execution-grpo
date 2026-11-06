"""Pin Hub weights/tokenizer metadata without downloading model weights."""
import json
import urllib.request
from pathlib import Path

from sqlrl.common import write_json

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
info = json.load(urllib.request.urlopen("https://huggingface.co/api/models/" + MODEL, timeout=30))
revision = info["sha"]
for path in Path("configs").glob("*.json"):
    value = json.loads(path.read_text())
    if value.get("model") == MODEL:
        value["revision"] = revision
        write_json(path, value)
write_json("results/model_selection.json", {
    "selected":MODEL, "revision":revision, "license":info.get("cardData",{}).get("license"),
    "alternative":"Qwen/Qwen3.5-4B", "selection_reason":"Text-only non-thinking instruction model; simple SQL-only completion and mature Qwen3 integration",
    "sql_superiority_measured":False, "weights_downloaded":False,
    "sources":["https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507", "https://huggingface.co/Qwen/Qwen3.5-4B"]})
print("Pinned", MODEL, revision)
