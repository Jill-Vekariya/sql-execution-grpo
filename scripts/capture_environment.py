"""Save public environment facts, never credentials or full environment vars."""
import importlib.metadata
import json
import platform
import shutil
import sqlite3
import subprocess

from sqlrl.common import write_json


def main():
    info = {"python":platform.python_version(), "platform":platform.platform(),
            "sqlite":sqlite3.sqlite_version, "cuda_gpu_present":shutil.which("nvidia-smi") is not None,
            "api_key_configured":False, "secrets_recorded":False}
    import os
    info["api_key_configured"] = bool(os.environ.get("SQLRL_API_KEY"))
    if info["cuda_gpu_present"]:
        output = subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"], capture_output=True, text=True)
        info["gpu"] = output.stdout.strip()
    info["packages"] = {d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}
    write_json("results/environment.json", info)
    print(json.dumps({k:v for k,v in info.items() if k != "packages"},indent=2))


if __name__ == "__main__":
    main()
