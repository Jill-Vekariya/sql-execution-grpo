"""Tiny fake HTTP responses exercise resume and byte-range integrity contracts."""
import importlib.util
import io
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/download_bird_parallel.py"


@pytest.fixture
def downloader(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("download_fixture",MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module,"ROOT",tmp_path)
    monkeypatch.setattr(module,"PART",tmp_path/"archive.part")
    monkeypatch.setattr(module,"TARGET",tmp_path/"archive.zip")
    monkeypatch.setattr(module,"STATE",tmp_path/"state.json")
    monkeypatch.setattr(module,"CHUNK",100)
    monkeypatch.setattr(module,"WORKERS",2)
    monkeypatch.chdir(tmp_path)
    data = bytes(range(256))*4

    class Response(io.BytesIO):
        status = 206

        def __init__(self,begin,end):
            super().__init__(data[begin:end+1])
            self.headers = {"Content-Range":f"bytes {begin}-{end}/{len(data)}","ETag":"fixed"}

    def open_range(request,timeout):
        value = next(v for k,v in request.header_items() if k.lower()=="range")
        begin,end = map(int,value.removeprefix("bytes=").split("-"))
        return Response(begin,end)

    monkeypatch.setattr(module.urllib.request,"urlopen",open_range)
    return module,data


def test_resumed_prefix_and_ranges_reconstruct_exact_bytes(downloader):
    module,data = downloader
    module.PART.write_bytes(data[:23])
    module.main()
    assert module.TARGET.read_bytes() == data
    assert not module.PART.exists()


def test_completed_download_is_not_overwritten(downloader):
    module,data = downloader
    module.TARGET.write_bytes(data)
    module.main()
    assert module.TARGET.read_bytes() == data
    assert not module.PART.exists()


def test_resume_state_without_partial_file_fails_closed(downloader):
    module,data = downloader
    module.STATE.write_text('{}')
    with pytest.raises(IOError,match="without"):
        module.main()


def test_mismatched_source_snapshot_fails_closed(downloader):
    module,data = downloader
    module.PART.write_bytes(data[:23])
    module.STATE.write_text('{"etag":"changed","size":1024}')
    with pytest.raises(IOError,match="changed"):
        module.main()
