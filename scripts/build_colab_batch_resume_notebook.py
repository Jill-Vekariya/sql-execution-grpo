"""Build a clean resume notebook for the already-running, frozen Colab experiment."""
import ast
import base64
import re
import hashlib
import json
import subprocess
from pathlib import Path

BASE_SHA="bf30e56913aabd272ee8c8d759c038e62246e6a66105602e98645a73b099b48e"
SFT_SHA="9c9fab2b38c75dc85ed3d4acbebb06125cebd3875353ecd50b4e0c8586ef84f7"

def main():
    revisions=subprocess.run(["git","log","--format=%H","--","notebooks/Free_Phase03_Colab.ipynb"],capture_output=True,text=True,check=True).stdout.splitlines()
    legacy=None
    for revision in revisions:
        candidate=json.loads(subprocess.run(["git","show",revision+":notebooks/Free_Phase03_Colab.ipynb"],capture_output=True,text=True,check=True).stdout)
        if candidate["metadata"]["source_snapshot_sha256"]==BASE_SHA:
            legacy=candidate
            break
    if legacy is None:
        revision=revisions[0]
        legacy=json.loads(subprocess.run(["git","show",revision+":notebooks/Free_Phase03_Colab.ipynb"],capture_output=True,text=True,check=True).stdout)
    names=["scripts/run_batched_generation.py","scripts/run_free_batched.py","scripts/probe_batch_generation.py",
           "scripts/colab_private_backup.py","scripts/colab_restore_private.py","tests/test_batched_generation.py",
           "tests/test_colab_restore.py","docs/BATCHED_FREE_RUN.md"]
    names += ["configs/free_batch8_v2/"+name+".json" for name in ["base_zero","base_few","sft","sft_eval","gemini"]]
    bundle={name:Path(name).read_text() for name in names}
    bundle_json=json.dumps(bundle)
    bundle_sha=hashlib.sha256(bundle_json.encode()).hexdigest()
    boot="import json, time\nSESSION_STARTED=time.monotonic()\n"+"".join(legacy["cells"][1]["source"])
    if legacy["metadata"]["source_snapshot_sha256"]!=BASE_SHA:
        pinned=Path("notebooks/frozen_phase03_colab_source.zip").read_bytes()
        assert hashlib.sha256(pinned).hexdigest()==BASE_SHA
        replacement="payload = base64.b64decode("+repr(base64.b64encode(pinned).decode())+")"
        boot,count=re.subn(r"payload = base64\.b64decode\([^\n]+\)",lambda unused:replacement,boot,count=1)
        assert count==1
        boot=boot.replace(repr(legacy["metadata"]["source_snapshot_sha256"]),repr(BASE_SHA))

    start=boot.index("with zipfile.ZipFile(io.BytesIO(payload)) as archive:")
    end=boot.index("os.chdir(PROJECT)",start)
    boot=boot[:start]+'''with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for name in archive.namelist():
        expected=archive.read(name)
        if name=='src/sqlrl/train_sft/run.py':
            expected=expected.decode().replace('save_strategy="steps", save_steps=100, save_total_limit=2,','save_strategy="steps", save_steps=config.get("checkpoint_steps",100), save_total_limit=2,').encode()
            assert hashlib.sha256(expected).hexdigest()=='SFT_DIGEST'
        if name=='configs/free/sft.json':
            value=json.loads(expected)
            value['checkpoint_steps']=10
            expected=(json.dumps(value,indent=2)+chr(10)).encode()
        path=PROJECT/name
        if path.exists(): assert path.read_bytes()==expected, 'Different existing source: '+name
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(expected)
'''.replace("SFT_DIGEST",SFT_SHA)+boot[end:]
    boot+="\nACTIVE_ROOT='results/free_runs_batch8_v2'\nOPERATIONAL_BUNDLE=json.loads("+repr(bundle_json)+")\n"
    boot+="""for name,content in OPERATIONAL_BUNDLE.items():
    target=PROJECT/name
    if target.exists(): assert target.read_text()==content, 'Different operational source: '+name
    else:
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(content)
(PERSIST/'resume_bundle_BUNDLE_DIGEST.json').write_text(json.dumps(OPERATIONAL_BUNDLE))
""".replace("BUNDLE_DIGEST",bundle_sha[:16])
    cells=[]
    def add(kind,source):
        if kind=="code": ast.parse(source)
        cell={"cell_type":kind,"metadata":{},"source":source.splitlines(keepends=True)}
        if kind=="code": cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    add("markdown","# Free Phases 0–3: recorded batch-eight run\nUse only a free GPU and keep this notebook private. This notebook restores the exact original source plus the documented inference-batching amendment. It retains 12,908 training examples and all 2,304 validation questions. No final-test access or RL is included. Run on a fresh runtime, not alongside the active notebook. Source, results, complete trainer checkpoints and restore receipts are backed up privately through the owner's existing Drive authorization. The hosted baseline still needs an owner-confirmed Free tier Gemini project with billing disabled. No completed result is claimed by this template.\n")
    add("code",boot)
    cells[-1]["metadata"]["cellView"]="form"
    setup="".join(legacy["cells"][2]["source"])
    add("code",setup)
    add("code",'''# Public Google APIs only. Credential values remain inside this runtime.
from google.colab import auth
auth.authenticate_user()
import google.auth
from googleapiclient.discovery import build
drive_credentials,unused_project=google.auth.default(scopes=['https://www.googleapis.com/auth/drive'])
drive_service=build('drive','v3',credentials=drive_credentials,cache_discovery=False)
drive_quota=drive_service.about().get(fields='storageQuota').execute()['storageQuota']
from scripts.colab_restore_private import restore_private_checkpoint
restore_receipt=restore_private_checkpoint(PERSIST,drive_service)
print(json.dumps(restore_receipt,indent=2))
''')
    add("code",'''# Private archive contains only project outputs, processed records and source.
backup_helper_source=(PROJECT/'scripts/colab_private_backup.py').read_text()
(PERSIST/'colab_private_backup.py').write_text(backup_helper_source)
exec(compile(backup_helper_source,'colab_private_backup.py','exec'),globals())
''')
    add("code",'''# TRAIN-only preparation; never run concurrently with the local model controller.
assert not globals().get('model_controller') or not model_controller.is_alive()
assert not globals().get('prep_process') or prep_process.poll() is not None
scratch=pathlib.Path('/content/sqlrl_temp/sqlrl_data')
scratch.mkdir(parents=True,exist_ok=True)
needed_gib=10 if any((scratch/'databases').glob('*/*/*.sqlite')) else 70
assert shutil.disk_usage(scratch).free>needed_gib*1024**3,'Insufficient free scratch space'
for name in ['raw','databases']:
    target=scratch/name
    target.mkdir(exist_ok=True)
    link=PROJECT/'data'/name
    if not link.exists():
        link.parent.mkdir(parents=True,exist_ok=True)
        if link.is_symlink(): link.unlink()
        link.symlink_to(target,target_is_directory=True)
prep_log=PERSIST/'prepare-data.log'
prep_stream=prep_log.open('a')
prep_process=subprocess.Popen([PY,'-u','scripts/prepare_free_data.py'],cwd=PROJECT,stdout=prep_stream,stderr=subprocess.STDOUT,start_new_session=True)
print('TRAIN preparation PID:',prep_process.pid)
''')
    add("code",'''# Bounded local work waits for verified TRAIN data and resumes exact saved prefixes.
import threading
assert not globals().get('model_controller') or not model_controller.is_alive()
def controller_status(value):
    path=PERSIST/'controller_status.json'
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+chr(10))
    temporary.replace(path)
def continue_batch8_experiments():
    try:
        controller_status({'status':'waiting_for_verified_train_data','run_root':ACTIVE_ROOT,'phase3_complete':False})
        assert prep_process.wait()==0,'TRAIN preparation failed; inspect prepare-data.log'
        seconds=min(10800,int(14400-(time.monotonic()-SESSION_STARTED)))
        assert seconds>=900,'Not enough remaining session time'
        with (PERSIST/'local-batch8-run.log').open('a') as stream:
            process=subprocess.Popen([PY,'-u','scripts/run_free_batched.py','--stage','local','--root',ACTIVE_ROOT,'--max-seconds',str(seconds)],cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            controller_status({'status':'running_local_experiments','run_root':ACTIVE_ROOT,'pid':process.pid,'bounded_seconds':seconds,'cash_budget_usd':0,'phase3_complete':False})
            code=process.wait()
        required={'base_zero':'completed','base_few':'completed','sft':'trained','sft_eval':'completed'}
        states={}
        for name in required:
            path=PROJECT/ACTIVE_ROOT/name/'status.json'
            states[name]=json.loads(path.read_text()).get('status') if path.exists() else 'pending'
        done=all(states[name]==expected for name,expected in required.items())
        controller_status({'status':'local_stages_completed' if done else 'paused_or_failed','run_root':ACTIVE_ROOT,'returncode':code,'stages':states,'hosted_baseline':'pending','phase3_complete':False})
    except Exception as error:
        controller_status({'status':'blocked','run_root':ACTIVE_ROOT,'reason':type(error).__name__+': '+str(error)[:500],'phase3_complete':False})
if RUN_LOCAL:
    model_controller=threading.Thread(target=continue_batch8_experiments,daemon=True,name='free_batch8_model_controller')
    model_controller.start()
print('Inspect the next cell for actual progress.')
''')
    add("code",'''# Re-run this inspection cell as needed; it does not launch model work.
print('TRAIN return code:',prep_process.poll())
print(prep_log.read_text(errors='replace')[-600:])
for filename in ['controller_status.json','backup_status.json']:
    path=PERSIST/filename
    print(filename,path.read_text() if path.exists() else 'pending')
root=PROJECT/ACTIVE_ROOT
for stage in ['base_zero','base_few','sft','sft_eval','frontier']:
    state=root/stage/'status.json'
    predictions=root/stage/'predictions.jsonl'
    count=sum(1 for line in predictions.open() if line.strip()) if predictions.exists() else 0
    print(stage,'predictions',count,'status',state.read_text() if state.exists() else 'pending')
    log=root/(stage+'.log')
    if log.exists(): print(log.read_text(errors='replace')[-600:])
print('Complete trainer checkpoints:',[p.name for p in (root/'sft/checkpoints').glob('checkpoint-*') if (p/'trainer_state.json').is_file()])
''')
    add("code",'''# Optional separate hosted stage. Never paste a key in notebook code or chat.
if RUN_GEMINI:
    assert BILLING_DISABLED_CONFIRMED,'Verify a Free tier project with billing disabled first'
    from google.colab import userdata
    os.environ['GEMINI_API_KEY']=userdata.get('GEMINI_API_KEY')
    os.environ['SQLRL_GEMINI_FREE_PROJECT']='1'
    try:
        subprocess.run([PY,'scripts/run_free_batched.py','--stage','gemini','--root',ACTIVE_ROOT,'--max-requests',str(MAX_REQUESTS)],cwd=PROJECT,check=True)
    finally:
        os.environ.pop('GEMINI_API_KEY',None)
''')
    add("code",'''# Comparison is produced only after every full-cohort evaluation exists.
needed=[PROJECT/ACTIVE_ROOT/(name+'_evaluation/summary.json') for name in ['base_zero','base_few','sft_eval','frontier']]
if all(path.exists() for path in needed):
    subprocess.run([PY,'scripts/run_free_batched.py','--stage','report','--root',ACTIVE_ROOT],cwd=PROJECT,check=True)
    print((PROJECT/ACTIVE_ROOT/'comparison.md').read_text())
else:
    print('Phases 0–3 are still incomplete. Preserve the verified private backup and resume unfinished stages.')
''')
    notebook={"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.12"},"accelerator":"GPU","colab":{"name":"Free_Phase03_Colab_Batch8_Resume.ipynb","provenance":[]},"source_snapshot_sha256":BASE_SHA,"operational_bundle_sha256":bundle_sha,"profile_id":"student_free_nf4_fp16_batch8_v2","parent_source_notebook_commit":revision},"cells":cells}
    output=Path("notebooks/Free_Phase03_Colab_Batch8_Resume.ipynb")
    output.write_text(json.dumps(notebook,indent=1)+"\n")
    assert output.stat().st_size<1000000,"Keep the portable notebook below 1 MB"
    print(json.dumps({"notebook":str(output),"bytes":output.stat().st_size,"base_source_sha256":BASE_SHA,"operational_bundle_sha256":bundle_sha,"code_cells_validated":sum(c["cell_type"]=="code" for c in cells),"phase3_complete":False}))

if __name__=="__main__":
    main()
