"""Failure-path tests for shared file-writing utilities."""

import sys
from types import SimpleNamespace

import pytest

from ftrb import run_utils


def test_save_csv_does_not_close_transferred_descriptor_twice(
    tmp_path, monkeypatch
) -> None:
    destination = tmp_path / "rows.csv"

    class FailingWriter:
        def __init__(self, *_args, **_kwargs):
            pass

        def writeheader(self):
            raise RuntimeError("write failed")

    def forbidden_close(_fd):
        raise AssertionError("save_csv must not close an fd owned by fdopen")

    monkeypatch.setattr(run_utils.csv, "DictWriter", FailingWriter)
    monkeypatch.setattr(run_utils.os, "close", forbidden_close)

    with pytest.raises(RuntimeError, match="write failed"):
        run_utils.save_csv([{"qid": "q1"}], str(destination))

    assert not destination.exists()
    assert not list(tmp_path.glob(".rows.csv.*.tmp"))


def test_cuda_request_fails_instead_of_silently_using_cpu(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(RuntimeError, match="no CUDA device is available"):
        run_utils.normalize_device("cuda")


def test_cpu_must_be_requested_explicitly() -> None:
    assert run_utils.normalize_device("cpu") == "cpu"
