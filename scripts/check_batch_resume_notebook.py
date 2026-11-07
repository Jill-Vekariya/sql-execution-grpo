"""Offline execution check of the exact portable source bootstrap; no model/API call."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

def main():
    notebook=json.loads(Path("notebooks/Free_Phase03_Colab_Batch8_Resume.ipynb").read_text())
    source="".join(notebook["cells"][1]["source"])
    original=Path.cwd()
    with tempfile.TemporaryDirectory(prefix="sqlrl-resume-check-") as directory:
        root=Path(directory)
        source=source.replace("/content/sqlrl_temp/sql-execution-grpo",str(root/"project"))
        source=source.replace("/content/sqlrl_free_run",str(root/"persist"))
        source=source.replace("/content/sqlrl_temp/sqlrl_hf_cache",str(root/"hf_cache"))
        values={"__name__":"__main__"}
        try:
            exec(compile(source,"portable_resume_bootstrap","exec"),values)
            project=values["PROJECT"]
            assert hashlib.sha256((project/"src/sqlrl/train_sft/run.py").read_bytes()).hexdigest()=="9c9fab2b38c75dc85ed3d4acbebb06125cebd3875353ecd50b4e0c8586ef84f7"
            assert json.loads((project/"configs/free/sft.json").read_text())["checkpoint_steps"]==10
            assert values["RUN_GEMINI"] is False and values["BILLING_DISABLED_CONFIRMED"] is False
            exec(compile(source,"portable_resume_bootstrap_repeated","exec"),values)
            script=project/"scripts/run_batched_generation.py"
            script.write_text(script.read_text()+"# incompatible change\n")
            try:
                exec(compile(source,"portable_resume_bootstrap_changed","exec"),values)
            except AssertionError:
                changed_source_rejected=True
            else:
                raise AssertionError("Changed operational source was not rejected")
        finally:
            os.chdir(original)
    receipt={"bootstrap_executed_offline":True,"identical_bootstrap_repeat_passed":True,
             "changed_operational_source_rejected":changed_source_rejected,"base_source_sha256":notebook["metadata"]["source_snapshot_sha256"],
             "operational_bundle_sha256":notebook["metadata"]["operational_bundle_sha256"],"model_or_api_called":False,"phase3_complete":False}
    Path("results/colab_resume_bootstrap_check.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt))

if __name__=="__main__":
    main()
