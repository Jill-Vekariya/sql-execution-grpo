"""Run the free profile in bounded notebook sessions; keep completed stages."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from sqlrl.common import sha256, write_json
from sqlrl.complete_phase03 import data_blockers, report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["local","gemini","report"],required=True)
    parser.add_argument("--root",default="results/free_runs")
    parser.add_argument("--max-seconds",type=int,default=30000)
    parser.add_argument("--max-requests",type=int,default=20)
    args=parser.parse_args()
    if args.max_seconds <= 0:
        raise ValueError("Session duration must be positive")
    root=Path(args.root);root.mkdir(parents=True,exist_ok=True)
    configs={}
    for name in ["base_zero","base_few","sft","sft_eval","gemini"]:
        config=json.loads(Path("configs/free",name+".json").read_text())
        config["output"]=str(root/("frontier" if name=="gemini" else name))
        if name=="sft_eval":config["adapter"]=str(root/"sft"/"adapter")
        path=root/"configs"/(name+".json")
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError("Existing free-run configuration differs; use a new root")
        write_json(path,config);configs[name]=config
    blockers,_=data_blockers(configs["base_zero"])
    if blockers:
        raise RuntimeError("; ".join(blockers))
    deadline=time.monotonic()+args.max_seconds

    def run(name, command):
        seconds=deadline-time.monotonic()
        if seconds <= 0:raise TimeoutError("Notebook session budget exhausted")
        begin=time.monotonic()
        with (root/(name+".log")).open("a") as log:
            process=subprocess.Popen([sys.executable,*command],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=process.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
                write_json(root/"session_status.json",{"status":"paused","reason":"session_time_limit","stage":name})
                raise TimeoutError("Saved stages/checkpoints preserved for the next notebook session")
        with (root/"sessions.jsonl").open("a") as log:
            log.write(json.dumps({"stage":name,"elapsed_s":time.monotonic()-begin,"returncode":code,"cash_cost_usd":0,"cost_basis":"user-selected free allocation"})+"\n")
        if code:raise RuntimeError(f"{name} failed; inspect its log. Completed stages remain saved")

    def evaluate(name):
        output=root/(name+"_evaluation")
        predictions=root/name/"predictions.jsonl"
        if (output/"summary.json").exists():
            saved=json.loads((output/"config.json").read_text())
            if saved["predictions_sha256"] != sha256(predictions) or saved["data_sha256"] != sha256("data/processed/val.jsonl"):
                raise ValueError("Saved evaluation inputs changed")
            return
        if output.exists():raise RuntimeError("Interrupted evaluation preserved; use a new run root to recompute")
        run(name+"_evaluation",["-m","sqlrl.eval.run","--predictions",str(predictions),"--output",str(output)])

    if args.stage=="local":
        if os.getenv("SQLRL_FREE_GPU") != "1":
            raise RuntimeError("Confirm a free GPU allocation and set SQLRL_FREE_GPU=1")
        for name in ["base_zero","base_few","sft","sft_eval"]:
            directory=Path(configs[name]["output"])
            state=json.loads((directory/"status.json").read_text()) if (directory/"status.json").exists() else {}
            done=state.get("status") in {"trained","completed"}
            if not done:
                command=["-m","sqlrl.train_sft.run" if name=="sft" else "sqlrl.eval.generate","--config",str(root/"configs"/(name+".json"))]
                if directory.exists():
                    if name=="sft":
                        checkpoints=list((directory/"checkpoints").glob("checkpoint-*"))
                        if not checkpoints:raise RuntimeError("No complete SFT checkpoint exists; preserve this attempt and select a new root")
                        checkpoint=max(checkpoints,key=lambda p:int(p.name.split("-")[-1]))
                        command += ["--resume-from-checkpoint",str(checkpoint)]
                    else:command += ["--resume"]
                run(name,command)
            if name!="sft":evaluate(name)
    elif args.stage=="gemini":
        directory=Path(configs["gemini"]["output"])
        command=["-m","sqlrl.eval.free_gemini","--config",str(root/"configs/gemini.json"),"--max-requests",str(args.max_requests)]
        if directory.exists():command += ["--resume"]
        run("gemini",command)
        state=json.loads((directory/"status.json").read_text())
        if state["status"]=="completed":evaluate("frontier")
    else:
        report(root,{"base_zero":0.0,"base_few":0.0,"sft":0.0,"sft_eval":0.0})
    print("Free stage finished; check results before continuing to the next stage.")


if __name__=="__main__":
    try:main()
    except TimeoutError as error:print(str(error))
