from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.utils import retained_output_directory as retained


def _portable_link(directory_fd: int, _source_fd: int, target: str) -> None:
    os.link(
        ".PASS.prepared",
        target,
        src_dir_fd=directory_fd,
        dst_dir_fd=directory_fd,
    )


def test_fresh_output_retains_tree_and_commits_marker_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = retained.FreshOutputDirectory.create(tmp_path / "fresh")
    try:
        output.write_new("manifest.json", b'{"status":"PASS"}\n').close()
        nested = output.path / "checkpoint-5"
        nested.mkdir(mode=0o700)
        (nested / "adapter.bin").write_bytes(b"adapter")
        prepared = retained.prepare_terminal_output(
            output,
            marker_name="PASS",
            marker_payload=b"[TASTE_T6_OURS_PPO_SMOKE_PASS]\n",
        )
        monkeypatch.setattr(retained, "_link_held_noreplace", _portable_link)
        assert not (output.path / "PASS").exists()
        assert (output.path / ".PASS.prepared").is_file()
        prepared.commit(retained_input_closure=lambda: None)
        assert (output.path / "PASS").read_bytes() == (
            b"[TASTE_T6_OURS_PPO_SMOKE_PASS]\n"
        )
        assert not (output.path / ".PASS.prepared").exists()
        prepared.close()
    finally:
        if output.descriptor >= 0:
            output.close()


def test_terminal_callback_failure_leaves_no_authorizing_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = retained.FreshOutputDirectory.create(tmp_path / "fresh")
    output.write_new("manifest.json", b"{}\n").close()
    prepared = retained.prepare_terminal_output(
        output,
        marker_name="PASS",
        marker_payload=b"PASS\n",
    )
    monkeypatch.setattr(retained, "_link_held_noreplace", _portable_link)

    def fail() -> None:
        raise RuntimeError("input drift")

    with pytest.raises(RuntimeError, match="input drift"):
        prepared.commit(retained_input_closure=fail)
    assert not (output.path / "PASS").exists()
    assert (output.path / ".PASS.prepared").is_file()
    prepared.close()


def test_retained_tree_rejects_equal_byte_leaf_replacement(tmp_path: Path) -> None:
    output = retained.FreshOutputDirectory.create(tmp_path / "fresh")
    output.write_new("manifest.json", b"same\n").close()
    tree = retained.RetainedOutputTree.capture(output.descriptor)
    original = output.path / "manifest.json"
    parked = output.path / "parked"
    original.rename(parked)
    original.write_bytes(b"same\n")
    with pytest.raises(retained.RetainedOutputDirectoryError, match="identity changed"):
        tree.revalidate()
    tree.close()
    output.close()


def test_fresh_output_rejects_ancestor_rename_copy(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    output = retained.FreshOutputDirectory.create(parent / "fresh")
    parked = tmp_path / "parked"
    parent.rename(parked)
    parent.mkdir()
    (parent / "fresh").mkdir()
    with pytest.raises(retained.RetainedOutputDirectoryError, match="named output ancestor"):
        output.revalidate()
    output.close()


def test_terminal_commit_revalidates_equal_byte_replacement_after_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = retained.FreshOutputDirectory.create(tmp_path / "fresh")
    output.write_new("manifest.json", b"same\n").close()
    prepared = retained.prepare_terminal_output(
        output,
        marker_name="PASS",
        marker_payload=b"PASS\n",
    )
    monkeypatch.setattr(retained, "_link_held_noreplace", _portable_link)

    def replace() -> None:
        source = output.path / "manifest.json"
        parked = output.path / "manifest.parked"
        source.rename(parked)
        source.write_bytes(b"same\n")

    with pytest.raises(retained.RetainedOutputDirectoryError):
        prepared.commit(retained_input_closure=replace)
    assert not (output.path / "PASS").exists()
    prepared.close()


def test_terminal_prepare_flush_failure_creates_no_terminal_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = retained.FreshOutputDirectory.create(tmp_path / "fresh")
    output.write_new("manifest.json", b"{}\n").close()

    def fail(_self: object) -> None:
        raise OSError("durability failure")

    monkeypatch.setattr(retained.RetainedOutputTree, "durably_flush", fail)
    with pytest.raises(OSError, match="durability failure"):
        retained.prepare_terminal_output(
            output,
            marker_name="PASS",
            marker_payload=b"PASS\n",
        )
    assert not (output.path / "output_hashes.json").exists()
    assert not (output.path / ".PASS.prepared").exists()
    assert not (output.path / "PASS").exists()
    output.close()


def test_terminal_inventory_binds_fresh_root_identity(tmp_path: Path) -> None:
    output = retained.FreshOutputDirectory.create(tmp_path / "fresh")
    output.write_new("manifest.json", b"{}\n").close()
    prepared = retained.prepare_terminal_output(
        output,
        marker_name="PASS",
        marker_payload=b"PASS\n",
    )
    assert prepared.tree.inventory["root"] == output.identity.evidence()
    prepared.close()


def test_terminal_link_without_prepared_unlink_remains_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = retained.FreshOutputDirectory.create(tmp_path / "fresh")
    output.write_new("manifest.json", b"{}\n").close()
    prepared = retained.prepare_terminal_output(
        output,
        marker_name="PASS",
        marker_payload=b"PASS\n",
    )
    monkeypatch.setattr(retained, "_link_held_noreplace", _portable_link)
    original_unlink = retained.os.unlink

    def fail_prepared_unlink(path: object, *args: object, **kwargs: object) -> None:
        if path == ".PASS.prepared":
            raise OSError("injected prepared unlink failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(retained.os, "unlink", fail_prepared_unlink)
    with pytest.raises(OSError, match="injected prepared unlink failure"):
        prepared.commit(retained_input_closure=lambda: None)
    assert (output.path / "PASS").is_file()
    assert (output.path / ".PASS.prepared").is_file()
    with pytest.raises(retained.RetainedOutputDirectoryError):
        prepared.revalidate()
    prepared.close()


def test_retained_tree_rejects_forbidden_bytes_from_held_text_leaf(
    tmp_path: Path,
) -> None:
    output = retained.FreshOutputDirectory.create(tmp_path / "fresh")
    output.write_new(
        "tokenizer_config.json",
        b'{"name_or_path":"/proc/self/fd/44"}\n',
    ).close()
    tree = retained.RetainedOutputTree.capture(output.descriptor)
    with pytest.raises(
        retained.RetainedOutputDirectoryError,
        match="tokenizer_config.json",
    ):
        tree.reject_byte_sequence(
            b"/proc/self/fd/",
            suffixes=(".json",),
        )
    tree.close()
    output.close()


def test_published_terminal_consumer_rejects_prepared_collision_and_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = retained.FreshOutputDirectory.create(tmp_path / "fresh")
    output.write_new("manifest.json", b"{}\n").close()
    prepared = retained.prepare_terminal_output(
        output,
        marker_name="PASS",
        marker_payload=b"PASS\n",
    )
    monkeypatch.setattr(retained, "_link_held_noreplace", _portable_link)
    prepared.commit(retained_input_closure=lambda: None)
    prepared.close()
    with retained.HeldPublishedTerminalOutput.open(
        tmp_path / "fresh",
        marker_name="PASS",
        marker_payload=b"PASS\n",
    ) as published:
        assert published.read_bytes("manifest.json") == b"{}\n"
    marker = tmp_path / "fresh/PASS"
    parked = tmp_path / "fresh/PASS.parked"
    marker.rename(parked)
    marker.write_bytes(b"PASS\n")
    with pytest.raises(retained.RetainedOutputDirectoryError):
        retained.HeldPublishedTerminalOutput.open(
            tmp_path / "fresh",
            marker_name="PASS",
            marker_payload=b"PASS\n",
        )
