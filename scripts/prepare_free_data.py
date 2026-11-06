"""Download only TRAIN data and restore the exact published cohort on Kaggle."""
import json
from pathlib import Path

from fetch_train import fetch
from sqlrl.common import sha256, write_json
from sqlrl.data.prepare import prepare
from sqlrl.data.restore_eligible import restore


def main():
    expected = json.loads(Path("results/data_audit.json").read_text())
    manifests = json.loads(Path("data/manifests/checksums.json").read_text())
    for dataset, budget in [("spider",512),("bird",12000)]:
        source = expected["sources"][dataset]
        annotation = Path("data/raw") / dataset / "train.json"
        paths = {db:Path("data/databases")/dataset/db/(db+".sqlite") for db in source["database_sha256"]}
        if not annotation.exists() or any(not p.exists() for p in paths.values()):
            fetch(dataset,budget)
        if sha256(annotation) != source["source_sha256"]:
            raise ValueError(f"{dataset} upstream TRAIN annotations changed")
        for db, path in paths.items():
            if sha256(path) != source["database_sha256"][db]:
                raise ValueError(f"{dataset}/{db} database bytes changed")
    prepare(seed=42)
    for split, entry in manifests.items():
        if sha256(entry["path"]) != entry["sha256"]:
            raise ValueError(f"Prepared {split} manifest differs from the published cohort")
    # Retain the original audit identity after independent byte verification.
    write_json("results/data_audit.json",expected)
    restore()


if __name__ == "__main__":
    main()
