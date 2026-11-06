"""Deterministic splits from official TRAIN data only."""
import hashlib
import random
import re
import unicodedata


def normalized_question(question):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", question).casefold()).strip()


def database_split(rows, seed=42, val_fraction=0.15):
    databases = sorted({row["dataset"] + ":" + row["db_id"] for row in rows})
    random.Random(seed).shuffle(databases)
    n_val = max(1, round(len(databases) * val_fraction))
    return {"train": sorted(databases[n_val:]), "val": sorted(databases[:n_val])}


def deduplicate(splits):
    """Keep validation; remove matching train questions. Keep first within split."""
    seen, clean, removed = set(), {}, []
    for name in ["val", "train"]:
        clean[name] = []
        for row in sorted(splits.get(name, []), key=lambda x: x["id"]):
            key = normalized_question(row["question"])
            if key in seen:
                removed.append({"id": row["id"], "split": name, "reason": "duplicate_question"})
            else:
                clean[name].append(row)
                seen.add(key)
    return clean, removed


def question_digest(question):
    return hashlib.sha256(normalized_question(question).encode()).hexdigest()
