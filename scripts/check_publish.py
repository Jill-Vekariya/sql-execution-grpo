"""Check staged paths and likely credentials without printing secret values."""
import re
import subprocess
from pathlib import Path

from sqlrl.common import write_json

listed = subprocess.run(["git","diff","--cached","--name-only","-z"],capture_output=True,text=True,check=True)
paths = [Path(x) for x in listed.stdout.split("\0") if x]
problems = []
patterns = [rb"gh[pousr]_[A-Za-z0-9]{30,}", rb"sk-[A-Za-z0-9_-]{24,}", rb"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"]
for path in paths:
    name = path.as_posix()
    if name.startswith(("data/raw/","data/databases/",".venv/",".tools/")) or path.suffix in {".sqlite",".safetensors",".bin"}:
        problems.append({"path":name,"reason":"private_or_large_runtime_artifact"})
    if path.stat().st_size > 20*1024*1024:
        problems.append({"path":name,"reason":"file_size_over_20_MiB"})
    if any(re.search(pattern,path.read_bytes()) for pattern in patterns):
        problems.append({"path":name,"reason":"credential_pattern"})
write_json("results/publish_check.json", {"staged_files_checked":len(paths),"problems":problems,"passed":not problems})
if problems:
    raise SystemExit(str(problems))
print("Staged path/credential check passed for",len(paths),"files")
