import io, stat, zipfile
import pytest
from scripts.colab_restore_private import validated_members

@pytest.mark.parametrize("name",["../outside","/absolute","a/../../outside","a\\\\escape"])
def test_restore_rejects_escaping_paths(name):
    data=io.BytesIO()
    with zipfile.ZipFile(data,"w") as archive: archive.writestr(name,b"x")
    with zipfile.ZipFile(data) as archive:
        with pytest.raises(ValueError,match="Unsafe"):
            validated_members(archive)

def test_restore_rejects_symlinks():
    data=io.BytesIO()
    member=zipfile.ZipInfo("linked")
    member.create_system=3
    member.external_attr=(stat.S_IFLNK|0o777)<<16
    with zipfile.ZipFile(data,"w") as archive: archive.writestr(member,b"../outside")
    with zipfile.ZipFile(data) as archive:
        with pytest.raises(ValueError,match="Symlink"):
            validated_members(archive)

def test_restore_accepts_ordinary_project_members():
    data=io.BytesIO()
    with zipfile.ZipFile(data,"w") as archive:
        archive.writestr("results/",b"")
        archive.writestr("results/free_runs_batch8_v2/base_zero/predictions.jsonl",b'{"id":"a"}\n')
    with zipfile.ZipFile(data) as archive:
        assert len(validated_members(archive))==2
