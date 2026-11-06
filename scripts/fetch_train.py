"""Fetch only TRAIN members from official ZIPs using HTTP byte ranges.

No dev/test member is opened, downloaded or extracted. Reading a ZIP directory
reveals filenames only. A server that ignores Range fails closed.
"""
import argparse
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

from sqlrl.common import sha256, write_json

SOURCES = {
    "spider": "https://drive.usercontent.google.com/download?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download&confirm=t",
    "bird": "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip",
}


class RangeReader(io.RawIOBase):
    def __init__(self, url):
        self.url, self.position = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=60) as response:
            if response.status != 206:
                raise IOError("Source does not honor byte ranges")
            self.size = int(response.headers["Content-Range"].split("/")[-1])
            self.etag = response.headers.get("ETag")
            self.modified = response.headers.get("Last-Modified")

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        end = self.position + size - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.position}-{end}"})
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {self.position}-{end}/"):
                raise IOError("Incorrect range response; refusing an unbounded download")
            value = response.read(size)
        if len(value) != size:
            raise IOError("Short range response")
        self.position += len(value)
        return value


def fetch(dataset, database_budget_mb):
    root = Path("data/raw") / dataset
    root.mkdir(parents=True, exist_ok=True)
    source = RangeReader(SOURCES[dataset])
    opened, omitted = [], []
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        target = "train_spider.json" if dataset == "spider" else "train.json"
        candidates = [n for n in names if Path(n).name == target and "__MACOSX" not in n]
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous training annotation: {candidates}")
        member = candidates[0]
        path = root / "train.json"
        path.write_bytes(archive.read(member))
        opened.append({"member": member, "path": str(path), "sha256": sha256(path)})
        rows = json.loads(path.read_text(encoding="utf-8"))
        allowed = {r["db_id"] for r in rows}
        for name in names:
            if Path(name).name in {"tables.json", "train_tables.json"} and "__MACOSX" not in name:
                path = root / "tables.json"
                path.write_bytes(archive.read(name))
                # Tables are metadata, filtered immediately to official TRAIN DBs.
                tables = json.loads(path.read_text(encoding="utf-8"))
                write_json(path, [t for t in tables if t["db_id"] in allowed])
                opened.append({"member": name, "path": str(path), "sha256": sha256(path), "train_metadata_only": True})
        candidates = [entry for entry in archive.infolist()
                      if entry.filename.endswith(".sqlite") and Path(entry.filename).stem in allowed
                      and "__MACOSX" not in entry.filename]
        nested = [entry for entry in archive.infolist() if Path(entry.filename).name == "train_databases.zip" and "__MACOSX" not in entry.filename]
        if not candidates and nested:
            entry = nested[0]
            if entry.compress_size > database_budget_mb * 1024 * 1024:
                omitted.extend({"db_id":db_id,"reason":"nested_database_archive_exceeds_download_budget",
                                "archive_compressed_bytes":entry.compress_size} for db_id in sorted(allowed))
            else:
                # The outer BIRD file contains only TRAIN data. A single streaming
                # download is much faster than thousands of small range requests.
                outer = root / "official_train.zip"
                if not outer.exists() or outer.stat().st_size != source.size:
                    temporary = outer.with_suffix(".part")
                    with urllib.request.urlopen(SOURCES[dataset], timeout=120) as reader, temporary.open("wb") as writer:
                        if reader.status != 200:
                            raise IOError("Expected complete TRAIN archive for full database download")
                        shutil.copyfileobj(reader, writer, length=16 * 1024 * 1024)
                    if temporary.stat().st_size != source.size:
                        raise IOError("Incomplete TRAIN archive")
                    temporary.replace(outer)
                nested_path = root / "train_databases.zip"
                if not nested_path.exists() or nested_path.stat().st_size != entry.file_size:
                    with zipfile.ZipFile(outer) as local_outer, local_outer.open(entry.filename) as reader, nested_path.open("wb") as writer:
                        shutil.copyfileobj(reader, writer, length=16 * 1024 * 1024)
                with zipfile.ZipFile(nested_path) as inner:
                    for db_entry in inner.infolist():
                        db_id = Path(db_entry.filename).stem
                        if not db_entry.filename.endswith(".sqlite") or db_id not in allowed or "__MACOSX" in db_entry.filename:
                            continue
                        destination = Path("data/databases") / dataset / db_id / (db_id + ".sqlite")
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with inner.open(db_entry) as reader, destination.open("wb") as writer:
                            shutil.copyfileobj(reader, writer, length=16 * 1024 * 1024)
                        opened.append({"member":entry.filename + "/" + db_entry.filename,"path":str(destination),"sha256":sha256(destination)})
                        print(dataset, db_id, "ready", flush=True)
        used = 0
        for entry in sorted(candidates, key=lambda e: (e.compress_size, e.filename)):
            db_id = Path(entry.filename).stem
            destination = Path("data/databases") / dataset / db_id / (db_id + ".sqlite")
            if used + entry.compress_size > database_budget_mb * 1024 * 1024:
                omitted.append({"db_id": db_id, "reason": "download_byte_budget", "compressed_bytes": entry.compress_size})
                continue
            used += entry.compress_size
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or destination.stat().st_size != entry.file_size:
                temporary = destination.with_suffix(".part")
                with archive.open(entry) as reader, temporary.open("wb") as writer:
                    shutil.copyfileobj(reader, writer, length=4 * 1024 * 1024)
                temporary.replace(destination)
            opened.append({"member": entry.filename, "path": str(destination), "sha256": sha256(destination)})
            print(dataset, db_id, "ready", flush=True)
    report = {"dataset": dataset, "url": SOURCES[dataset], "archive_size_bytes": source.size,
              "etag": source.etag, "last_modified": source.modified, "source_split": "train",
              "opened_members": opened, "omitted_databases": omitted, "final_data_opened": False,
              "database_download_budget_mb": database_budget_mb}
    write_json(Path("results/sources") / (dataset + ".json"), report)
    print(dataset, "annotations", len(rows), "DBs", len(allowed), "available", len(opened), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(SOURCES), required=True)
    parser.add_argument("--database-budget-mb", type=int, default=512)
    args = parser.parse_args()
    fetch(args.dataset, args.database_budget_mb)
