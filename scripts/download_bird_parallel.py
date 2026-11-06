"""Resume the official TRAIN archive using verified, bounded HTTP ranges.

Only the training archive is accessed. Extracting it with zipfile later checks
member CRCs; a full-length file is not treated as complete until all chunks land.
"""
import concurrent.futures
import json
import os
import threading
import time
import urllib.request
from pathlib import Path

from sqlrl.common import write_json

URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"
ROOT = Path("data/raw/bird")
PART = ROOT / "official_train.part"
TARGET = ROOT / "official_train.zip"
STATE = ROOT / "parallel_download_state.json"
CHUNK = 8 * 1024 * 1024
WORKERS = 8


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(URL,headers={"Range":"bytes=0-0"}),timeout=60) as response:
        if response.status != 206:
            raise IOError("Range download unsupported")
        total = int(response.headers["Content-Range"].split("/")[-1])
        etag = response.headers.get("ETag")
    if TARGET.exists():
        if TARGET.stat().st_size != total:
            raise IOError("Existing archive has the wrong size")
        print("Complete archive already present; extraction verifies member CRCs",flush=True)
        return
    if STATE.exists() and not PART.exists():
        raise IOError("Resume state exists without its partial archive")
    state = json.loads(STATE.read_text()) if STATE.exists() else {
        "url":URL,"etag":etag,"size":total,"prefix_bytes":PART.stat().st_size if PART.exists() else 0,"completed":[]}
    if state["etag"] != etag or state["size"] != total:
        raise IOError("Archive changed; refusing to combine different snapshots")
    started = time.monotonic()
    fd = os.open(PART, os.O_RDWR | os.O_CREAT, 0o600)
    os.ftruncate(fd,total)
    completed = set(state["completed"])
    tasks = [(begin,min(begin+CHUNK,total)) for begin in range(state["prefix_bytes"],total,CHUNK) if begin not in completed]
    stop = threading.Event()

    def fetch(bounds):
        begin,end = bounds
        if stop.is_set():
            return None
        error = None
        for attempt in range(3):
            try:
                headers = {"Range":f"bytes={begin}-{end-1}"}
                if etag:
                    headers["If-Range"] = etag
                with urllib.request.urlopen(urllib.request.Request(URL,headers=headers),timeout=90) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {begin}-{end-1}/{total}":
                        raise IOError("Wrong byte-range response")
                    body = response.read(end-begin)
                if len(body) != end-begin:
                    raise IOError("Incomplete chunk")
                offset = 0
                while offset < len(body):
                    written = os.pwrite(fd,body[offset:],begin+offset)
                    if written <= 0:
                        raise IOError("Archive write made no progress")
                    offset += written
                return begin
            except Exception as exception:
                error = exception
        stop.set()
        raise IOError(f"Chunk {begin} failed: {error}")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for i,future in enumerate(concurrent.futures.as_completed([pool.submit(fetch,t) for t in tasks])):
                begin = future.result()
                if begin is not None:
                    completed.add(begin)
                if i % 10 == 0:
                    state["completed"] = sorted(completed)
                    write_json(STATE,state)
                    done = state["prefix_bytes"] + sum(min(CHUNK,total-b) for b in completed)
                    print(f"Verified {done}/{total} bytes ({100*done/total:.1f}%)",flush=True)
        os.fsync(fd)
    finally:
        os.close(fd)
        state["completed"] = sorted(completed)
        write_json(STATE,state)
    if len(completed) != len(range(state["prefix_bytes"],total,CHUNK)):
        raise IOError("Not all chunks completed")
    PART.replace(TARGET)
    write_json("results/bird_parallel_download.json",{"status":"complete","url":URL,"etag":etag,"archive_bytes":total,
        "resumed_prefix_bytes":state["prefix_bytes"],"workers":WORKERS,"chunk_bytes":CHUNK,"elapsed_s":time.monotonic()-started})


if __name__ == "__main__":
    main()
