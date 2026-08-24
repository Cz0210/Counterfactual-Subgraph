from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from src.baselines.comrecgc.external_memory_recourse import PAIR_STORE_SCHEMA
from src.utils import aids_comrecgc_v5_snapshot as snapshot


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, object]:
    source = tmp_path / "old" / "pair_store"
    source.mkdir(parents=True)
    parents = 2
    candidates = 3
    rows = parents * candidates
    pairs = np.asarray(
        [(candidate, parent) for candidate in range(candidates) for parent in range(parents)],
        dtype=np.int64,
    )
    vectors = np.arange(rows * 2, dtype=np.float32).reshape(rows, 2) / 100.0
    pair_path = source / "pair_indices.npy"
    vector_path = source / "recourse_vectors.npy"
    np.save(pair_path, pairs, allow_pickle=False)
    np.save(vector_path, vectors, allow_pickle=False)
    identity = {
        "dataset": "aids",
        "parent_count": parents,
        "candidate_count": candidates,
        "pair_order": "candidate_major_parent_minor",
    }
    manifest = {
        "schema_version": PAIR_STORE_SCHEMA,
        "run_complete": True,
        "candidate_major_parent_minor_order": True,
        "row_count": rows,
        "vector_dim": 2,
        "vectors_dtype": "float32",
        "pairs_path": str(pair_path),
        "vectors_path": str(vector_path),
        "pairs_sha256": _sha(pair_path),
        "vectors_sha256": _sha(vector_path),
        "scientific_identity": identity,
        "scientific_identity_sha256": snapshot._stable_hash(identity),
        "chunk_count": 0,
        "chunks": [],
    }
    manifest_path = source / "run_manifest.json"
    _write_json(manifest_path, manifest)
    proc = tmp_path / "proc"
    proc.mkdir()
    old_output = tmp_path / "old"
    project = tmp_path / "project"
    project.mkdir()
    return {
        "source": source,
        "manifest": manifest_path,
        "manifest_sha": _sha(manifest_path),
        "pairs": pair_path,
        "vectors": vector_path,
        "proc": proc,
        "old_output": old_output,
        "project": project,
        "rows": rows,
        "parents": parents,
        "candidates": candidates,
        "dim": 2,
    }


def _install_process_gate(monkeypatch: pytest.MonkeyPatch, *, exits_after: int | None = None) -> None:
    calls = 0

    def fake_verify(**kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        alive = exits_after is None or calls <= exits_after
        return {
            "status": "PASS",
            "process_set_status": (
                "ALLOWED_OLD_READ_ONLY_PROCESS_PRESENT"
                if alive
                else "ALLOWED_OLD_PROCESS_NATURALLY_EXITED"
            ),
            "allowed_old_process_count": int(alive),
            "active_common_recourse_count": int(alive),
            "allowed_pid": int(kwargs["allowed_pid"]),
        }

    monkeypatch.setattr(snapshot, "verify_process_set", fake_verify)


def _run(
    fixture: dict[str, object],
    output: Path,
    *,
    resume: bool = False,
    min_free_after_bytes: int = 0,
) -> dict[str, object]:
    return snapshot.create_promoted_pair_store_snapshot(
        source_root=fixture["source"],
        expected_source_manifest_sha256=str(fixture["manifest_sha"]),
        output_dir=output,
        proc_root=fixture["proc"],
        allowed_pid=77,
        allowed_start_ticks=1234,
        allowed_cmdline_sha256="a" * 64,
        allowed_output_root=fixture["old_output"],
        allowed_project_root=fixture["project"],
        min_free_after_bytes=min_free_after_bytes,
        expected_row_count=int(fixture["rows"]),
        expected_vector_dim=int(fixture["dim"]),
        expected_parent_count=int(fixture["parents"]),
        expected_candidate_count=int(fixture["candidates"]),
        resume=resume,
    )


def test_physical_snapshot_is_exact_distinct_and_dbscan_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    _install_process_gate(monkeypatch, exits_after=1)
    output = tmp_path / "fresh-snapshot"

    result = _run(fixture, output)

    assert result["status"] == "PASS"
    assert (output / "PASS").read_text() == "PASS\n"
    copied_pairs = output / "pair_store/pair_indices.npy"
    copied_vectors = output / "pair_store/recourse_vectors.npy"
    assert np.array_equal(np.load(copied_pairs), np.load(fixture["pairs"]))
    assert np.array_equal(np.load(copied_vectors), np.load(fixture["vectors"]))
    assert copied_pairs.stat().st_ino != Path(fixture["pairs"]).stat().st_ino
    assert copied_vectors.stat().st_ino != Path(fixture["vectors"]).stat().st_ino
    contract = json.loads((output / "dbscan_contract.json").read_text())
    assert contract["eps"] == 0.02
    assert contract["min_samples"] == 3
    assert contract["self_neighbor_included"] is True
    assert contract["metric"] == "euclidean"
    assert contract["sklearn_version"] == "1.7.2"
    assert contract["pair_indices_role"] == "row_provenance_only_not_adjacency_or_distance_edges"
    assert result["source_post"]["allowed_writer_generation"]["allowed_old_process_count"] == 0


def test_snapshot_existing_root_requires_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    _install_process_gate(monkeypatch)
    output = tmp_path / "snapshot"
    output.mkdir()
    with pytest.raises(FileExistsError, match="must be fresh"):
        _run(fixture, output)


def test_resume_restarts_non_authoritative_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    _install_process_gate(monkeypatch)
    output = tmp_path / "snapshot"
    (output / "pair_store").mkdir(parents=True)
    partial = output / "pair_store/.pair_indices.npy.partial"
    partial.write_bytes(b"truncated")

    result = _run(fixture, output, resume=True)

    assert result["status"] == "PASS"
    assert not partial.exists()


def test_snapshot_start_accepts_already_naturally_exited_old_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    _install_process_gate(monkeypatch, exits_after=0)
    result = _run(fixture, tmp_path / "snapshot")
    assert result["status"] == "PASS"
    assert result["source"]["allowed_writer_generation"]["allowed_old_process_count"] == 0


def test_resume_discards_large_regular_partial_before_headroom_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    _install_process_gate(monkeypatch)
    output = tmp_path / "snapshot"
    pair_store = output / "pair_store"
    pair_store.mkdir(parents=True)
    partial = pair_store / ".recourse_vectors.npy.partial"
    partial.write_bytes(b"x" * 4096)
    remaining = Path(fixture["pairs"]).stat().st_size + Path(
        fixture["vectors"]
    ).stat().st_size
    floor = 10_000

    def free_space(_path: Path) -> shutil._ntuple_diskusage:
        free = floor + remaining - 1 if partial.exists() else floor + remaining
        return shutil._ntuple_diskusage(100_000, 0, free)

    monkeypatch.setattr(snapshot.shutil, "disk_usage", free_space)
    result = _run(fixture, output, resume=True, min_free_after_bytes=floor)
    assert result["status"] == "PASS"
    assert result["discarded_non_authoritative_partials"] == [str(partial)]


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_resume_rejects_nonregular_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    fixture = _fixture(tmp_path)
    _install_process_gate(monkeypatch)
    output = tmp_path / "snapshot"
    pair_store = output / "pair_store"
    pair_store.mkdir(parents=True)
    partial = pair_store / ".pair_indices.npy.partial"
    if kind == "symlink":
        partial.symlink_to(fixture["pairs"])
    else:
        partial.mkdir()
    with pytest.raises(snapshot.PairStoreSnapshotError, match="physical regular"):
        _run(fixture, output, resume=True)


def test_logical_source_and_output_symlinks_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    _install_process_gate(monkeypatch)
    source_link = tmp_path / "source-link"
    source_link.symlink_to(fixture["source"], target_is_directory=True)
    with pytest.raises(snapshot.PairStoreSnapshotError, match="symlink"):
        snapshot.create_promoted_pair_store_snapshot(
            source_root=source_link,
            expected_source_manifest_sha256=str(fixture["manifest_sha"]),
            output_dir=tmp_path / "snapshot",
            proc_root=fixture["proc"],
            allowed_pid=77,
            allowed_start_ticks=1234,
            allowed_cmdline_sha256="a" * 64,
            allowed_output_root=fixture["old_output"],
            allowed_project_root=fixture["project"],
            min_free_after_bytes=0,
            expected_row_count=int(fixture["rows"]),
            expected_vector_dim=int(fixture["dim"]),
            expected_parent_count=int(fixture["parents"]),
            expected_candidate_count=int(fixture["candidates"]),
        )
    real_output = tmp_path / "real-output"
    real_output.mkdir()
    output_link = tmp_path / "output-link"
    output_link.symlink_to(real_output, target_is_directory=True)
    with pytest.raises(snapshot.PairStoreSnapshotError, match="symlink"):
        _run(fixture, output_link, resume=True)


def test_resume_after_first_array_promotion_only_charges_remaining_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    _install_process_gate(monkeypatch)
    output = tmp_path / "snapshot"
    original = snapshot._copy_one
    calls = 0

    def crash_second(**kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected crash")
        return original(**kwargs)

    monkeypatch.setattr(snapshot, "_copy_one", crash_second)
    with pytest.raises(OSError, match="injected crash"):
        _run(fixture, output)
    assert (output / "pair_store/pair_indices.npy").is_file()
    monkeypatch.setattr(snapshot, "_copy_one", original)
    remaining = Path(fixture["vectors"]).stat().st_size
    floor = 10_000
    monkeypatch.setattr(
        snapshot.shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(100_000, 0, floor + remaining),
    )

    result = _run(fixture, output, resume=True, min_free_after_bytes=floor)

    assert result["status"] == "PASS"
    assert result["remaining_bytes_at_headroom_gate"] == remaining


def test_terminal_manifest_to_pass_crash_window_reconciles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    _install_process_gate(monkeypatch)
    output = tmp_path / "snapshot"
    original = snapshot._publish_pass
    called = False

    def crash_once(path: Path) -> None:
        nonlocal called
        if not called:
            called = True
            raise OSError("injected PASS crash")
        original(path)

    monkeypatch.setattr(snapshot, "_publish_pass", crash_once)
    with pytest.raises(OSError, match="injected PASS crash"):
        _run(fixture, output)
    assert (output / "snapshot_manifest.json").is_file()
    assert not (output / "PASS").exists()
    monkeypatch.setattr(snapshot, "_publish_pass", original)

    assert _run(fixture, output, resume=True)["status"] == "PASS"


def test_hardlinked_existing_destination_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    _install_process_gate(monkeypatch)
    output = tmp_path / "snapshot"
    pair_store = output / "pair_store"
    pair_store.mkdir(parents=True)
    os.link(fixture["pairs"], pair_store / "pair_indices.npy")

    with pytest.raises(snapshot.PairStoreSnapshotError, match="hardlink"):
        _run(fixture, output, resume=True)


def test_source_drift_during_copy_fails_without_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    _install_process_gate(monkeypatch)
    output = tmp_path / "snapshot"
    original = snapshot._copy_one
    calls = 0

    def mutate_after_copy(**kwargs: object) -> dict[str, object]:
        nonlocal calls
        result = original(**kwargs)
        calls += 1
        if calls == 2:
            vectors = np.load(fixture["vectors"])
            vectors[0, 0] += 1
            np.save(fixture["vectors"], vectors, allow_pickle=False)
        return result

    monkeypatch.setattr(snapshot, "_copy_one", mutate_after_copy)
    with pytest.raises(snapshot.PairStoreSnapshotError, match="full-hash closure"):
        _run(fixture, output)
    assert not (output / "PASS").exists()


@pytest.mark.parametrize("artifact", ["pair_store/pair_indices.npy", "dbscan_contract.json"])
def test_terminal_resume_rejects_destination_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact: str
) -> None:
    fixture = _fixture(tmp_path)
    _install_process_gate(monkeypatch)
    output = tmp_path / "snapshot"
    _run(fixture, output)
    path = output / artifact
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.chmod(0o600)
    path.write_bytes(payload)

    with pytest.raises(snapshot.PairStoreSnapshotError):
        _run(fixture, output, resume=True)


def test_pair_formula_mismatch_is_rejected_before_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    _install_process_gate(monkeypatch)
    pairs = np.load(fixture["pairs"])
    pairs[[0, 1]] = pairs[[1, 0]]
    np.save(fixture["pairs"], pairs, allow_pickle=False)
    manifest = json.loads(Path(fixture["manifest"]).read_text())
    manifest["pairs_sha256"] = _sha(Path(fixture["pairs"]))
    _write_json(Path(fixture["manifest"]), manifest)
    fixture["manifest_sha"] = _sha(Path(fixture["manifest"]))

    with pytest.raises(snapshot.PairStoreSnapshotError, match="pair order changed"):
        _run(fixture, tmp_path / "snapshot")


def test_headroom_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    _install_process_gate(monkeypatch)
    monkeypatch.setattr(
        snapshot.shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(1000, 0, 1),
    )

    with pytest.raises(snapshot.PairStoreSnapshotError, match="headroom"):
        _run(fixture, tmp_path / "snapshot", min_free_after_bytes=1)


def test_only_exact_old_source_writer_is_permitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    _install_process_gate(monkeypatch)
    source = Path(fixture["source"])

    def writers(paths: list[Path], **_kwargs: object) -> list[dict[str, object]]:
        if any(source in path.parents or path == source for path in paths):
            return [{"pid": 77, "kind": "mapping", "path": str(paths[-1])}]
        return []

    monkeypatch.setattr(snapshot, "_find_writable_process_references", writers)
    assert _run(fixture, tmp_path / "snapshot")["status"] == "PASS"


def test_unexpected_source_or_destination_writer_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    _install_process_gate(monkeypatch)
    monkeypatch.setattr(
        snapshot,
        "_find_writable_process_references",
        lambda _paths, **_kwargs: [{"pid": 88, "kind": "fd", "path": "rogue"}],
    )
    with pytest.raises(snapshot.PairStoreSnapshotError, match="unexpected source writer"):
        _run(fixture, tmp_path / "source-writer")

    calls = 0

    def destination_writer(paths: list[Path], **_kwargs: object) -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        # Source scans occur first and may recur; the destination scan is the
        # only call whose file set includes snapshot_manifest.json.
        if any(path.name == "snapshot_manifest.json" for path in paths):
            return [{"pid": 88, "kind": "fd", "path": str(paths[0])}]
        return []

    monkeypatch.setattr(
        snapshot, "_find_writable_process_references", destination_writer
    )
    with pytest.raises(snapshot.PairStoreSnapshotError, match="live writer"):
        _run(fixture, tmp_path / "destination-writer")


def test_checkpoint_identity_tamper_is_not_a_resume_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    _install_process_gate(monkeypatch)
    output = tmp_path / "snapshot"
    original = snapshot._copy_one

    def crash_first(**_kwargs: object) -> dict[str, object]:
        raise OSError("injected before first copy")

    monkeypatch.setattr(snapshot, "_copy_one", crash_first)
    with pytest.raises(OSError):
        _run(fixture, output)
    checkpoint = json.loads((output / "snapshot_checkpoint.json").read_text())
    checkpoint["identity"]["row_count"] += 1
    _write_json(output / "snapshot_checkpoint.json", checkpoint)
    monkeypatch.setattr(snapshot, "_copy_one", original)
    with pytest.raises(snapshot.PairStoreSnapshotError, match="checkpoint identity"):
        _run(fixture, output, resume=True)


def test_cli_and_slurm_keep_frozen_full_dimensions() -> None:
    root = Path(__file__).resolve().parents[2]
    cli = (root / "scripts/autodl/snapshot_aids_comrecgc_pair_store.py").read_text()
    slurm = (root / "scripts/slurm/snapshot_aids_comrecgc_pair_store.sh").read_text()
    assert "--expected-row-count" in cli
    assert "--expected-vector-dim" in cli
    assert "#SBATCH --partition=A800" in slurm
    assert "#SBATCH --gres=gpu:a800:1" in slurm
    assert "export PYTHONPATH=$PWD" in slurm
    assert "--config configs/hpc.yaml" in slurm
    assert "exit 78" in slurm
    supervisor = (
        root / "scripts/autodl/run_aids_comrecgc_v5_snapshot_supervisor.sh"
    ).read_text()
    assert "AIDS_COMRECGC_V5_SNAPSHOT_MAX_SAME_ROOT_RESUMES" in supervisor
    assert "status != 137 && status != 143" in supervisor
    assert "--resume" in supervisor
    assert "--validate-only" in supervisor
    assert "test hooks are forbidden" in supervisor
