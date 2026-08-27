from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from src.utils.retained_readonly_file import (
    RetainedReadonlyFileError,
    hold_readonly_file,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_retained_file_reads_exact_held_bytes_and_revalidates(tmp_path: Path) -> None:
    root = tmp_path / "authority" / "nested"
    root.mkdir(parents=True)
    target = root / "train.csv"
    data = b"molecule_id,parent_smiles,label\n1,CCO,1\n"
    target.write_bytes(data)
    with hold_readonly_file(target, expected_sha256=_sha(data)) as held:
        assert held.read_bytes() == data
        evidence = held.revalidate()
        assert evidence["path"] == str(target)
        assert evidence["sha256"] == _sha(data)
        assert type(evidence["inode"]) is int


def test_retained_file_rejects_leaf_replacement(tmp_path: Path) -> None:
    target = tmp_path / "train.csv"
    target.write_bytes(b"original\n")
    with hold_readonly_file(target) as held:
        original = tmp_path / "train-original.csv"
        target.rename(original)
        target.write_bytes(b"replacement\n")
        with pytest.raises(
            RetainedReadonlyFileError,
            match="input file identity changed|named input leaf no longer identifies",
        ):
            held.revalidate()


def test_retained_file_rejects_ancestor_rename_copy(tmp_path: Path) -> None:
    authority = tmp_path / "authority"
    authority.mkdir()
    target = authority / "train.csv"
    target.write_bytes(b"original\n")
    with hold_readonly_file(target) as held:
        moved = tmp_path / "authority-original"
        authority.rename(moved)
        authority.mkdir()
        (authority / "train.csv").write_bytes(b"replacement\n")
        with pytest.raises(
            RetainedReadonlyFileError,
            match="named input ancestor no longer identifies",
        ):
            held.read_bytes()


def test_retained_file_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_bytes(b"rows\n")
    symlink = tmp_path / "symlink.csv"
    symlink.symlink_to(source)
    with pytest.raises(OSError):
        hold_readonly_file(symlink)
    hardlink = tmp_path / "hardlink.csv"
    os.link(source, hardlink)
    with pytest.raises(RetainedReadonlyFileError, match="single-link regular"):
        hold_readonly_file(source)


@pytest.mark.parametrize("path", ("relative.csv", "/tmp/../tmp/train.csv"))
def test_retained_file_requires_normalized_absolute_path(path: str) -> None:
    with pytest.raises(RetainedReadonlyFileError, match="normalized absolute"):
        hold_readonly_file(path)
