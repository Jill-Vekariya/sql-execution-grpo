"""Result files are evidence. Never silently overwrite an experiment."""
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def seed_everything(seed):
    random.seed(seed)
    # Core data/evaluation do not require torch or numpy.
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        from transformers import set_seed
        set_seed(seed)
    except ImportError:
        pass


def start_run(directory, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "config.json", config)
    packages = {}
    for name in ["torch", "transformers", "peft", "trl", "vllm", "sqlglot"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    write_json(directory / "environment.json", {
        "utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "platform": platform.platform(), "packages": packages,
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "source_sha256":{str(p):sha256(p) for p in sorted(Path("src").rglob("*.py"))},
    })
    return directory
