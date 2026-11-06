"""Attempt declared runs with argument arrays, preserving every exit/log."""
import json
import subprocess
import sys
from pathlib import Path

from sqlrl.common import write_json

attempts = []
for name, module in [("base_zero","sqlrl.eval.generate"),("base_few","sqlrl.eval.generate"),
                     ("frontier","sqlrl.eval.generate"),("sft","sqlrl.train_sft.run")]:
    config_path = Path("configs") / (name + ".json")
    config = json.loads(config_path.read_text())
    if Path(config["output"]).exists():
        attempts.append({"config":str(config_path),"status":"existing_evidence_preserved"})
        continue
    command = [sys.executable,"-m",module,"--config",str(config_path)]
    completed = subprocess.run(command, capture_output=True, text=True)
    log = Path("results") / ("attempt-" + name + ".log")
    log.write_text(completed.stdout + completed.stderr)
    attempts.append({"config":str(config_path),"returncode":completed.returncode,"log":str(log)})
write_json("results/run_attempts.json", {"attempts":attempts})
print(json.dumps(attempts,indent=2))
