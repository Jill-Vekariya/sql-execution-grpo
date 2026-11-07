"""Restore only the private project archive through the owner's Drive service."""
import hashlib
import json
import pathlib
import shutil
import stat
import zipfile

BASE_SOURCE_SHA256 = "bf30e56913aabd272ee8c8d759c038e62246e6a66105602e98645a73b099b48e"

def stream_digest(stream, algorithm="sha256"):
    result=hashlib.new(algorithm)
    for block in iter(lambda:stream.read(8*1024**2),b""):
        result.update(block)
    return result.hexdigest()

def validated_members(archive):
    members=archive.infolist()
    seen=set()
    total=0
    for member in members:
        path=pathlib.PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts or not path.parts or "\\" in member.filename:
            raise ValueError("Unsafe archive member path")
        if stat.S_ISLNK(member.external_attr >> 16) or member.filename in seen:
            raise ValueError("Symlink or duplicate archive member")
        seen.add(member.filename)
        total+=member.file_size
    if total > 8*1024**3:
        raise ValueError("Archive expands beyond the declared 8 GiB restore cap")
    return members

def restore_private_checkpoint(persist,service):
    from googleapiclient.http import MediaIoBaseDownload
    persist=pathlib.Path(persist).resolve()
    folders=service.files().list(q="name='SQLRL-free-phase03-9c9fab2b' and mimeType='application/vnd.google-apps.folder' and trashed=false and 'root' in parents",fields="files(id)").execute()["files"]
    if not folders:
        return {"status":"no_prior_project_backup","phase3_complete":False}
    if len(folders)!=1:
        raise ValueError("Ambiguous project folders; do not overwrite")
    files=service.files().list(q="'"+folders[0]["id"]+"' in parents and name='phase03-latest.zip' and trashed=false",fields="files(id,size,md5Checksum,webViewLink,appProperties)").execute()["files"]
    if not files:
        return {"status":"no_prior_checkpoint_file","phase3_complete":False}
    if len(files)!=1:
        raise ValueError("Ambiguous checkpoint files; do not overwrite")
    remote=files[0]
    size=int(remote["size"])
    if not 0<size<=1024**3:
        raise ValueError("Backup exceeds the 1 GiB transfer cap")
    transfer=persist.parent/"sqlrl_restore_transfer.zip"
    with transfer.open("wb") as stream:
        download=MediaIoBaseDownload(stream,service.files().get_media(fileId=remote["id"]),chunksize=8*1024**2)
        done=False
        while not done:
            unused,done=download.next_chunk(num_retries=0)
    with transfer.open("rb") as stream: md5=stream_digest(stream,"md5")
    with transfer.open("rb") as stream: sha=stream_digest(stream)
    if transfer.stat().st_size!=size or md5!=remote["md5Checksum"] or remote.get("appProperties",{}).get("sha256",sha)!=sha:
        raise ValueError("Downloaded archive checksum differs from Drive")
    with zipfile.ZipFile(transfer) as archive:
        members=validated_members(archive)
        if hashlib.sha256(archive.read("source_snapshot.zip")).hexdigest()!=BASE_SOURCE_SHA256:
            raise ValueError("Backup belongs to a different frozen source snapshot")
        if archive.testzip() is not None:
            raise ValueError("Backup ZIP integrity check failed")
        if shutil.disk_usage(persist).free < sum(m.file_size for m in members)+1024**3:
            raise ValueError("Insufficient local restore space")
        for member in members:
            target=persist/member.filename
            if not target.resolve().is_relative_to(persist) or any(p.is_symlink() for p in [target,*target.parents] if p.is_relative_to(persist)):
                raise ValueError("Restore target escapes its project folder")
            if target.exists() and not member.is_dir():
                with target.open("rb") as local,archive.open(member) as saved:
                    if stream_digest(local)!=stream_digest(saved):
                        raise ValueError("Preserve different existing evidence: "+member.filename)
        for member in members:
            target=persist/member.filename
            if member.is_dir(): target.mkdir(parents=True,exist_ok=True)
            elif not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                temporary=target.with_name(target.name+".restore_tmp")
                with archive.open(member) as saved,temporary.open("wb") as local:
                    shutil.copyfileobj(saved,local,length=1024**2)
                temporary.replace(target)
    receipt={"status":"restored_private_drive_backup","bytes":size,"sha256":sha,"md5Checksum":md5,"drive_file":remote,"phase3_complete":False}
    target=persist/"restore_receipt_current.json"
    if target.exists():
        target=persist/("restore_receipt_"+sha[:16]+".json")
    if target.exists() and json.loads(target.read_text())!=receipt:
        raise ValueError("Preserve an existing different restore receipt")
    target.write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt
