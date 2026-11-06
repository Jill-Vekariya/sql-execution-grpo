"""Hash evidence and source files. Do not include this index in its own hash."""
from pathlib import Path
import subprocess

from sqlrl.common import sha256, write_json

roots = [Path("src"), Path("scripts"), Path("tests"), Path("configs"), Path("docs"), Path("data/manifests"), Path("results")]
listed = subprocess.run(["git","ls-files","--cached","--others","--exclude-standard","-z"],capture_output=True,text=True,check=True)
publishable = set(listed.stdout.split("\0"))
files = []
for root in roots:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.as_posix() in publishable and "__pycache__" not in path.parts and path != Path("results/evidence_index.json"):
            files.append(path)
for path in [Path("README.md"), Path("pyproject.toml"), Path("uv.lock"),Path("requirements-cpu-smoke.in"),Path("requirements-cpu-smoke.lock")]:
    if path.exists():
        files.append(path)
write_json("results/evidence_index.json", {"algorithm":"sha256", "files":{str(path):sha256(path) for path in files}})
