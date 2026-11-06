"""Build a private-upload notebook containing a pinned source snapshot, no keys."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile


def main():
    names=[]
    for directory in ["src","scripts","tests","configs","docs","data/manifests"]:
        names.extend(p for p in Path(directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    names += [Path(p) for p in ["pyproject.toml","uv.lock","requirements-free.in","requirements-free.lock","LICENSE","DATA_LICENSES.md","README.md","results/data_audit.json","results/training_eligibility.json","results/model_selection.json"]]
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in sorted(set(names)):
            info=zipfile.ZipInfo(str(path),(2026,11,7,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,path.read_bytes())
    data=buffer.getvalue();encoded=base64.b64encode(data).decode();digest=hashlib.sha256(data).hexdigest()
    cells=[]
    def cell(kind,text):
        value={"cell_type":kind,"metadata":{},"source":text.splitlines(keepends=True)}
        if kind=="code":value.update(execution_count=None,outputs=[])
        cells.append(value)
    cell("markdown","# Free Qwen text-to-SQL: Phases 0–3\nKeep this notebook private. Use a free Kaggle GPU allocation. No paid resource is created. Read docs/FREE_STUDENT_RUN.md in the extracted project. This notebook has not yet produced a model benchmark.\n")
    cell("code",'''import os, pathlib, shutil, zipfile, io, base64, hashlib, subprocess, sys
PROJECT = pathlib.Path('/kaggle/temp/sql-execution-grpo')
PERSIST = pathlib.Path('/kaggle/working/sqlrl_free_run')
# Attach your previous PRIVATE output and enter its sqlrl_free_run path.
RESUME_FROM = ''
RUN_LOCAL = True
RUN_GEMINI = False
BILLING_DISABLED_CONFIRMED = False
MAX_REQUESTS = 20
if RESUME_FROM and not PERSIST.exists():
    shutil.copytree(RESUME_FROM, PERSIST)
PERSIST.mkdir(parents=True, exist_ok=True)
PROJECT.mkdir(parents=True, exist_ok=True)
payload = base64.b64decode('''+repr(encoded)+''')
assert hashlib.sha256(payload).hexdigest() == '''+repr(digest)+'''
snapshot = PERSIST / 'source_snapshot.zip'
if snapshot.exists(): assert snapshot.read_bytes() == payload, 'Resume requires the same source snapshot'
else: snapshot.write_bytes(payload)
# No DB/archive symlink is placed under /kaggle/working.
for relative, target in [('results', PERSIST/'results'), ('data/processed', PERSIST/'processed')]:
    target.mkdir(exist_ok=True)
    link = PROJECT / relative
    if not link.exists():
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for name in archive.namelist():
        path = PROJECT / name
        if path.exists():
            assert path.read_bytes() == archive.read(name), f'Changed source or metadata: {name}'
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(name))
os.chdir(PROJECT)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['SQLRL_FREE_GPU'] = '1'  # Use only on your confirmed free allocation.
os.environ['HF_HOME'] = '/kaggle/temp/sqlrl_hf_cache'
print('Source SHA256:', hashlib.sha256(payload).hexdigest())
''')
    cell("code",'''# Official PyPI and PyTorch CUDA 12.6 wheels; no cloud billing setup.
subprocess.run([sys.executable, '-m', 'pip', 'install', 'uv==0.13.7'], check=True)
ENV_DIR = '/kaggle/temp/sqlrl_venv'
if not pathlib.Path(ENV_DIR, 'bin/python').exists():
    subprocess.run(['uv', 'venv', '--python', '3.12', ENV_DIR], check=True)
PY = str(pathlib.Path(ENV_DIR, 'bin/python'))
subprocess.run(['uv', 'pip', 'sync', '--python', PY, 'requirements-free.lock', '--extra-index-url', 'https://download.pytorch.org/whl/cu126', '--index-strategy', 'unsafe-best-match'], check=True)
subprocess.run(['uv', 'pip', 'install', '--python', PY, '--no-deps', '-e', '.'], check=True)
subprocess.run([PY, '-m', 'pytest'], check=True)
if RUN_LOCAL:
    subprocess.run([PY, '-c', "import torch; assert torch.cuda.is_available(), 'Select a free GPU in Kaggle Settings'; p=torch.cuda.get_device_properties(0); print(torch.__version__,p.name,p.total_memory,torch.cuda.get_arch_list()); x=torch.ones(8,device='cuda'); print((x+x).sum().item())"], check=True)
''')
    cell("code",'''# Downloads only TRAIN. Temporary DBs/archives are not persistent output.
scratch = pathlib.Path('/kaggle/temp/sqlrl_data')
scratch.mkdir(parents=True, exist_ok=True)
assert shutil.disk_usage(scratch).free > 70 * 1024**3, 'Need scratch space for 31 GB DBs, archives and model cache'
for name in ['raw', 'databases']:
    target = scratch / name
    target.mkdir(exist_ok=True)
    link = PROJECT / 'data' / name
    if not link.exists():
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink(): link.unlink()
        link.symlink_to(target, target_is_directory=True)
subprocess.run([PY, 'scripts/prepare_free_data.py'], check=True)
''')
    cell("code",'''# Zero-shot, few-shot, QLoRA SFT and SFT evaluation. Resume completed stages.
if RUN_LOCAL:
    subprocess.run([PY, 'scripts/run_free.py', '--stage', 'local', '--max-seconds', '28000'], check=True)
''')
    cell("code",'''# Optional separate CPU/API session. Never paste your key into this cell.
if RUN_GEMINI:
    assert BILLING_DISABLED_CONFIRMED, 'Verify the API project is Free tier with billing disabled'
    from kaggle_secrets import UserSecretsClient
    os.environ['GEMINI_API_KEY'] = UserSecretsClient().get_secret('GEMINI_API_KEY')
    os.environ['SQLRL_GEMINI_FREE_PROJECT'] = '1'
    try:
        subprocess.run([PY, 'scripts/run_free.py', '--stage', 'gemini', '--max-requests', str(MAX_REQUESTS)], check=True)
    finally:
        os.environ.pop('GEMINI_API_KEY', None)
''')
    cell("code",'''# Publish no accuracy comparison until all four full-cohort evaluations exist.
needed = [PROJECT/'results/free_runs'/f'{name}_evaluation/summary.json' for name in ['base_zero','base_few','sft_eval','frontier']]
if all(path.exists() for path in needed):
    subprocess.run([PY, 'scripts/run_free.py', '--stage', 'report'], check=True)
    print((PROJECT/'results/free_runs/comparison.md').read_text())
else:
    print('Experiment pending. Preserve this PRIVATE notebook output and resume unfinished stages.')
''')
    notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.12"},"source_snapshot_sha256":digest},"nbformat":4,"nbformat_minor":5}
    output=Path("notebooks/Free_Phase03_Kaggle.ipynb");output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(notebook,indent=1)+"\n")
    print(json.dumps({"notebook":str(output),"source_snapshot_sha256":digest,"source_files":len(names),"notebook_bytes":output.stat().st_size}))


if __name__=="__main__":main()
