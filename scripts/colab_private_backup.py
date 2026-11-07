# Colab-only private backup helper. Uses the owner's existing Drive authorization.
import hashlib, json, pathlib, threading, time, zipfile
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

def save_private_checkpoint():
    api=build('drive','v3',credentials=drive_credentials,cache_discovery=False)
    folder_name='SQLRL-free-phase03-9c9fab2b'
    folders=api.files().list(q="name='"+folder_name+"' and mimeType='application/vnd.google-apps.folder' and trashed=false and 'root' in parents",fields='files(id,name)').execute()['files']
    assert len(folders)<=1,'Ambiguous project folders; preserve files for review'
    folder_id=folders[0]['id'] if folders else api.files().create(body={'name':folder_name,'mimeType':'application/vnd.google-apps.folder'},fields='id').execute()['id']
    checkpoint_roots=[PERSIST/'results'/profile/'sft/checkpoints' for profile in ['free_runs','free_runs_batch8_v2']]
    latest={}
    for root in checkpoint_roots:
        complete=[]
        for directory in root.glob('checkpoint-*'):
            required=['trainer_state.json','optimizer.pt','scheduler.pt','rng_state.pth','adapter_model.safetensors','adapter_config.json']
            if all((directory/name).is_file() for name in required) and max(p.stat().st_mtime for p in directory.rglob('*') if p.is_file())<time.time()-30:
                complete.append(directory)
        latest[root]=max(complete,key=lambda p:int(p.name.rsplit('-',1)[1])) if complete else None
    transfer=pathlib.Path('/content/sqlrl_checkpoint_transfer.zip')
    with zipfile.ZipFile(transfer,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for path in PERSIST.rglob('*'):
            if not path.is_file() or path.is_symlink(): continue
            if any(root in path.parents and (selected is None or selected not in path.parents) for root,selected in latest.items()): continue
            archive.write(path,path.relative_to(PERSIST))
    with zipfile.ZipFile(transfer) as archive: assert archive.testzip() is None
    size=transfer.stat().st_size
    assert size<=1024**3,'Pause above the declared 1 GiB transfer cap'
    existing=api.files().list(q="'"+folder_id+"' in parents and name='phase03-latest.zip' and trashed=false",fields='files(id,size)').execute()['files']
    assert len(existing)<=1,'Ambiguous checkpoint files; stop for review'
    quota=api.about().get(fields='storageQuota').execute()['storageQuota']
    available=int(quota.get('limit',0))-int(quota.get('usage',0))+(int(existing[0]['size']) if existing else 0)
    assert size<available,'Insufficient existing Drive storage; never purchase more'
    sha=hashlib.sha256()
    md5=hashlib.md5()
    with transfer.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):
            sha.update(block); md5.update(block)
    media=MediaFileUpload(str(transfer),mimetype='application/zip',resumable=True,chunksize=8*1024**2)
    metadata={'name':'phase03-latest.zip','appProperties':{'sha256':sha.hexdigest(),'profile':'student_free_nf4_fp16_batch8_v2'}}
    if existing:
        request=api.files().update(fileId=existing[0]['id'],body=metadata,media_body=media,fields='id,size,md5Checksum,webViewLink')
    else:
        metadata['parents']=[folder_id]
        request=api.files().create(body=metadata,media_body=media,fields='id,size,md5Checksum,webViewLink')
    uploaded=None
    while uploaded is None:
        progress,uploaded=request.next_chunk(num_retries=0)
    assert uploaded['md5Checksum']==md5.hexdigest() and int(uploaded['size'])==size
    return {'status':'verified_private_drive_backup','sha256':sha.hexdigest(),'bytes':size,'drive_file':uploaded,
            'checkpoints':{str(root.relative_to(PERSIST)):selected.name if selected else None for root,selected in latest.items()},'phase3_complete':False}

def backup_loop():
    while not backup_stop.is_set():
        if 'drive_quota' in globals():
            try: receipt=save_private_checkpoint()
            except Exception as error: receipt={'status':'backup_paused','reason':type(error).__name__+': '+str(error)[:300]}
            target=PERSIST/'backup_status.json'
            temporary=target.with_suffix('.tmp')
            temporary.write_text(json.dumps(receipt,indent=2)+chr(10))
            temporary.replace(target)
        backup_stop.wait(300)

if 'backup_controller' in globals() and backup_controller.is_alive():
    backup_stop.set()
    backup_controller.join(timeout=30)
    if backup_controller.is_alive():
        backup_stop.clear()
        raise RuntimeError('Existing backup is still finishing; retry worker replacement later')
backup_stop=threading.Event()
backup_controller=threading.Thread(target=backup_loop,daemon=True,name='private_checkpoint_backup_v2')
backup_controller.start()
print('Private backup worker covers both recorded runs; newest full checkpoint per run only.',flush=True)
