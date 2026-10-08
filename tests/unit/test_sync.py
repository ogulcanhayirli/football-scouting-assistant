import hashlib
from pathlib import Path

import pytest
import yaml

from scout import cli
from scout.sync import sync_data, sync_dir

BUCKET = "scout-test"


class FakeS3:
    """In memory stand in for the boto3 S3 calls the sync uses."""

    def __init__(self, page_size: int = 2) -> None:
        self.objects: dict[str, bytes] = {}
        self.page_size = page_size
        self.puts: list[str] = []
        self.downloads: list[str] = []

    def put_object(self, Bucket, Key, Body):  # noqa: N803 (boto3 names)
        assert Bucket == BUCKET
        self.objects[Key] = Body.read() if hasattr(Body, "read") else Body
        self.puts.append(Key)

    def download_file(self, bucket, key, filename):
        assert bucket == BUCKET
        with open(filename, "wb") as fh:
            fh.write(self.objects[key])
        self.downloads.append(key)

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self

    def paginate(self, Bucket, Prefix):  # noqa: N803
        assert Bucket == BUCKET
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        for i in range(0, max(len(keys), 1), self.page_size):
            chunk = keys[i : i + self.page_size]
            yield {
                "Contents": [
                    {
                        "Key": k,
                        "Size": len(self.objects[k]),
                        "ETag": f'"{hashlib.md5(self.objects[k]).hexdigest()}"',
                    }
                    for k in chunk
                ]
            }


def _write(path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@pytest.fixture
def data(tmp_path):
    raw = tmp_path / "data" / "raw"
    processed = tmp_path / "data" / "processed"
    _write(raw / "api_football" / "players" / "a.json", b'{"a": 1}')
    _write(raw / "api_football" / "players" / "b.json", b'{"b": 2}')
    _write(raw / "api_football" / "leagues" / "c.json", b"{}")
    _write(processed / "player_season_stats" / "league=epl" / "season=2015.parquet", b"PAR1")
    return {"raw": raw, "processed": processed}


def test_upload_then_second_run_skips_everything(data):
    s3 = FakeS3()
    first = sync_data(s3, BUCKET, data, "up")
    assert len(first["raw"].transferred) == 3
    assert first["processed"].transferred == [
        "processed/player_season_stats/league=epl/season=2015.parquet"
    ]
    assert "raw/api_football/players/a.json" in s3.objects

    second = sync_data(s3, BUCKET, data, "up")
    assert all(not r.transferred for r in second.values())
    assert len(second["raw"].skipped) == 3
    assert len(s3.puts) == 4


def test_parallel_transfers_match_sequential(data, tmp_path):
    for i in range(30):
        _write(data["raw"] / "api_football" / "fixtures" / f"f{i}.json", f'{{"i": {i}}}'.encode())
    s3 = FakeS3(page_size=7)
    up = sync_dir(s3, BUCKET, data["raw"], "raw", "up", workers=6)
    assert len(up.transferred) == 33
    assert up.transferred == sorted(up.transferred)
    target = tmp_path / "copy"
    down = sync_dir(s3, BUCKET, target, "raw", "down", workers=6)
    assert len(down.transferred) == 33
    for src in data["raw"].rglob("*.json"):
        assert (target / src.relative_to(data["raw"])).read_bytes() == src.read_bytes()
    assert sync_dir(s3, BUCKET, data["raw"], "raw", "up", workers=6).transferred == []


def test_upload_sends_only_new_and_changed_files(data):
    s3 = FakeS3()
    sync_data(s3, BUCKET, data, "up")
    _write(data["raw"] / "api_football" / "players" / "new.json", b"{}")
    _write(data["raw"] / "api_football" / "players" / "a.json", b'{"a": 9}')  # same size
    result = sync_dir(s3, BUCKET, data["raw"], "raw", "up")
    assert sorted(result.transferred) == [
        "raw/api_football/players/a.json",
        "raw/api_football/players/new.json",
    ]
    assert s3.objects["raw/api_football/players/a.json"] == b'{"a": 9}'


def test_download_restores_files_and_skips_existing(data, tmp_path):
    s3 = FakeS3()
    sync_data(s3, BUCKET, data, "up")
    target = {"raw": tmp_path / "copy" / "raw", "processed": tmp_path / "copy" / "processed"}
    _write(target["raw"] / "api_football" / "players" / "a.json", b'{"a": 1}')

    result = sync_data(s3, BUCKET, target, "down")
    assert result["raw"].skipped == ["raw/api_football/players/a.json"]
    assert len(result["raw"].transferred) == 2
    for name in ("raw", "processed"):
        for src in data[name].rglob("*"):
            if src.is_file():
                copy = target[name] / src.relative_to(data[name])
                assert copy.read_bytes() == src.read_bytes()
    assert not list(tmp_path.joinpath("copy").rglob("*.part"))

    again = sync_data(s3, BUCKET, target, "down")
    assert all(not r.transferred for r in again.values())


def test_download_replaces_a_damaged_local_file(data, tmp_path):
    s3 = FakeS3()
    sync_data(s3, BUCKET, data, "up")
    target = tmp_path / "copy"
    _write(target / "api_football" / "players" / "a.json", b'{"a": 0}')
    result = sync_dir(s3, BUCKET, target, "raw/", "down")
    assert "raw/api_football/players/a.json" in result.transferred
    assert (target / "api_football" / "players" / "a.json").read_bytes() == b'{"a": 1}'


def test_prefix_is_applied(data):
    s3 = FakeS3()
    sync_data(s3, BUCKET, data, "up", prefix="v1")
    assert all(k.startswith("v1/raw/") or k.startswith("v1/processed/") for k in s3.objects)


def test_dry_run_changes_nothing(data, tmp_path):
    s3 = FakeS3()
    result = sync_data(s3, BUCKET, data, "up", dry_run=True)
    assert len(result["raw"].transferred) == 3
    assert s3.objects == {}

    sync_data(s3, BUCKET, data, "up")
    target = tmp_path / "copy"
    down = sync_dir(s3, BUCKET, target, "raw", "down", dry_run=True)
    assert len(down.transferred) == 3
    assert not target.exists()


def test_download_refuses_keys_outside_the_folder(tmp_path):
    s3 = FakeS3()
    s3.objects["raw/../../evil.txt"] = b"x"
    with pytest.raises(ValueError, match="outside"):
        sync_dir(s3, BUCKET, tmp_path / "raw", "raw", "down")


def test_missing_local_folder_uploads_nothing(tmp_path):
    result = sync_dir(FakeS3(), BUCKET, tmp_path / "absent", "raw", "up")
    assert result.transferred == []


def _config(tmp_path, **paths):
    template = Path(__file__).parents[2] / "configs" / "backfill.free.yaml"
    settings = yaml.safe_load(template.read_text(encoding="utf-8"))
    settings["paths"] = {"state_dir": str(tmp_path / "state")} | {
        k: str(v) for k, v in paths.items()
    }
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(settings), encoding="utf-8")
    return config


def test_cli_sync_uses_bucket_from_environment(data, tmp_path, monkeypatch, capsys):
    boto3 = pytest.importorskip("boto3")
    s3 = FakeS3()
    monkeypatch.setattr(boto3, "client", lambda service, **kwargs: s3)
    monkeypatch.setenv("SCOUT_S3_BUCKET", BUCKET)
    config = _config(tmp_path, raw_dir=data["raw"], processed_dir=data["processed"])
    assert cli.main(["--config", str(config), "sync", "up", "--only", "raw"]) == 0
    assert len(s3.objects) == 3
    assert "raw: sync up: 3 transferred" in capsys.readouterr().out


def test_cli_sync_without_bucket_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("SCOUT_S3_BUCKET", raising=False)
    config = _config(tmp_path)
    assert cli.main(["--config", str(config), "sync", "up"]) == 1
