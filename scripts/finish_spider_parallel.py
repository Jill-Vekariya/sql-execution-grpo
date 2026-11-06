"""Fetch TRAIN database members directly, checking each ZIP member's CRC."""
import concurrent.futures
import json
import struct
import urllib.request
import zipfile
import zlib
from pathlib import Path

from fetch_train import RangeReader,SOURCES
from sqlrl.common import sha256,write_json


def main():
    rows=json.loads(Path("data/raw/spider/train.json").read_text())
    allowed={r["db_id"] for r in rows}
    source=RangeReader(SOURCES["spider"])
    with zipfile.ZipFile(source) as archive:
        selected={}
        for entry in sorted(archive.infolist(),key=lambda x:(x.compress_size,x.filename)):
            db_id=Path(entry.filename).stem
            if entry.filename.endswith(".sqlite") and db_id in allowed and "__MACOSX" not in entry.filename:
                selected.setdefault(db_id,entry)

    def get_bytes(start,end):
        headers={"Range":f"bytes={start}-{end-1}"}
        if source.etag:headers["If-Range"]=source.etag
        with urllib.request.urlopen(urllib.request.Request(SOURCES["spider"],headers=headers),timeout=90) as response:
            if response.status!=206 or response.headers.get("Content-Range")!=f"bytes {start}-{end-1}/{source.size}":
                raise IOError("Wrong byte-range response")
            value=response.read(end-start)
        if len(value)!=end-start:raise IOError("Incomplete member download")
        return value

    def fetch(item):
        db_id,entry=item
        path=Path("data/databases/spider")/db_id/(db_id+".sqlite")
        if path.exists() and path.stat().st_size==entry.file_size and zlib.crc32(path.read_bytes())==entry.CRC:
            return {"db_id":db_id,"path":str(path),"member":entry.filename,"sha256":sha256(path)}
        error=None
        for attempt in range(3):
            try:
                header=get_bytes(entry.header_offset,entry.header_offset+30)
                if header[:4]!=b"PK\x03\x04":raise IOError("Invalid local ZIP header")
                name_length,extra_length=struct.unpack_from("<HH",header,26)
                start=entry.header_offset+30+name_length+extra_length
                compressed=get_bytes(start,start+entry.compress_size)
                if entry.compress_type==zipfile.ZIP_DEFLATED:
                    body=zlib.decompress(compressed,-15)
                elif entry.compress_type==zipfile.ZIP_STORED:
                    body=compressed
                else:raise IOError("Unsupported member compression")
                if len(body)!=entry.file_size or zlib.crc32(body)!=entry.CRC:
                    raise IOError("Member CRC or size mismatch")
                path.parent.mkdir(parents=True,exist_ok=True)
                temporary=path.with_suffix(".parallel.part")
                temporary.write_bytes(body)
                temporary.replace(path)
                print("spider",db_id,"CRC verified",flush=True)
                return {"db_id":db_id,"path":str(path),"member":entry.filename,"sha256":sha256(path)}
            except Exception as exception:error=exception
        raise IOError(f"{db_id}: {error}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        records=list(pool.map(fetch,sorted(selected.items())))
    if len(records)!=len(allowed):raise IOError("Missing training database entries")
    write_json("results/sources/spider_parallel.json",{"dataset":"spider","url":SOURCES["spider"],
        "archive_size_bytes":source.size,"etag":source.etag,"source_split":"train","databases":records,
        "crc_checked":True,"final_data_opened":False})


if __name__=="__main__":main()
