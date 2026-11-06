"""The Phase 0–3 entry points cannot load official dev or final-test data."""
import json
from pathlib import Path


def load_records(path, expected_split=None):
    path = Path(path)
    denied = {"dev", "validation", "final", "final_test", "test"}
    tokens = set(path.stem.casefold().replace("-", "_").split("_"))
    if tokens & denied or "final_test" in path.stem.casefold():
        raise PermissionError("Final/official-dev data is sealed until Phase 9")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if row.get("split") not in {"train", "val"}:
            raise PermissionError("Only project train/val records are permitted")
        if row.get("source_split", "train") != "train":
            raise PermissionError("Official dev/test cannot be relabeled as project val")
        if expected_split and row["split"] != expected_split:
            raise PermissionError("Unexpected split")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate record IDs")
    return rows
