from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("rdkit")

from scripts.build_molecular_graph_cache import (  # noqa: E402
    build_molecular_graph_cache,
    build_parser,
)
from src.data.molecular_graph_dataset import (  # noqa: E402
    MolecularGraphDataset,
    load_molecular_graph_cache,
    save_molecular_graph_cache,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_splits(root: Path, *, num_classes: int) -> dict[str, Path]:
    root.mkdir(parents=True)
    names = {
        "train": "train.csv",
        "validation": "val.csv",
        "calibration": "calibration.csv",
        "test": "test.csv",
    }
    split_values = {
        "train": "train",
        "validation": "val",
        "calibration": "calibration",
        "test": "test",
    }
    smiles = ("C", "N", "O")
    paths: dict[str, Path] = {}
    for split_name, file_name in names.items():
        path = root / file_name
        rows = ["molecule_id,smiles,label,split"]
        rows.extend(
            f"{split_name}_{label},{smiles[label]},{label},{split_values[split_name]}"
            for label in range(num_classes)
        )
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        paths[split_name] = path
    return paths


def _config(path: Path) -> Path:
    path.write_text("runtime:\n  environment: local\n", encoding="utf-8")
    return path


def _only_safe_types(value: Any) -> bool:
    if isinstance(value, (str, int, float, bool, type(None), torch.Tensor)):
        return True
    if isinstance(value, list):
        return all(_only_safe_types(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _only_safe_types(item)
            for key, item in value.items()
        )
    return False


@pytest.mark.parametrize(("dataset", "num_classes"), [("bace", 2), ("taste", 3)])
def test_builds_four_split_safe_cache_and_manifest(
    tmp_path: Path,
    dataset: str,
    num_classes: int,
) -> None:
    data_dir = tmp_path / "data"
    sources = _write_splits(data_dir, num_classes=num_classes)
    output = tmp_path / "graphs"

    result = build_molecular_graph_cache(
        config_files=[_config(tmp_path / "hpc.yaml")],
        dataset=dataset,
        data_dir=data_dir,
        output_dir=output,
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert result["num_classes"] == num_classes
    assert manifest["num_classes"] == num_classes
    assert manifest["total_graph_count"] == num_classes * 4
    assert manifest["serialization_contract"] == {
        "custom_pickled_objects": False,
        "fresh_output_required": True,
        "payload_types": "plain_tensors_and_python_primitives",
        "torch_load_weights_only": True,
    }
    assert list(manifest["splits"]) == [
        "calibration",
        "test",
        "train",
        "validation",
    ]
    for split_name, source in sources.items():
        entry = manifest["splits"][split_name]
        cache_path = output / entry["cache_file"]
        assert entry["graph_count"] == num_classes
        assert entry["source_csv_sha256"] == _sha256(source)
        assert entry["cache_sha256"] == _sha256(cache_path)
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        assert _only_safe_types(payload)
        loaded = load_molecular_graph_cache(
            cache_path,
            expected_num_classes=num_classes,
            expected_source_sha256=entry["source_csv_sha256"],
        )
        assert len(loaded) == num_classes
        assert loaded.dataset_fingerprint == entry["dataset_fingerprint"]


def test_cache_loader_always_requests_weights_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_splits(tmp_path / "data", num_classes=2)["train"]
    dataset = MolecularGraphDataset.from_csv(
        source, num_classes=2, expected_split="train"
    )
    cache = tmp_path / "train.pt"
    save_molecular_graph_cache(dataset, cache, split_name="train")
    real_load = torch.load
    calls: list[dict[str, Any]] = []

    def recording_load(*args: Any, **kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    load_molecular_graph_cache(cache)
    assert calls and all(call.get("weights_only") is True for call in calls)


def test_cache_output_must_be_fresh(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_splits(data_dir, num_classes=2)
    config = _config(tmp_path / "hpc.yaml")
    output = tmp_path / "graphs"
    build_molecular_graph_cache(
        config_files=[config], dataset="bace", data_dir=data_dir, output_dir=output
    )
    before = _sha256(output / "manifest.json")

    with pytest.raises(FileExistsError, match="must be fresh"):
        build_molecular_graph_cache(
            config_files=[config],
            dataset="bace",
            data_dir=data_dir,
            output_dir=output,
        )

    assert _sha256(output / "manifest.json") == before


def test_all_four_split_csvs_are_required_before_output(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    sources = _write_splits(data_dir, num_classes=3)
    sources["calibration"].unlink()
    output = tmp_path / "graphs"

    with pytest.raises(FileNotFoundError, match="All four molecular splits"):
        build_molecular_graph_cache(
            config_files=[_config(tmp_path / "hpc.yaml")],
            dataset="tastemolnet",
            data_dir=data_dir,
            output_dir=output,
        )

    assert not output.exists()


def test_loader_rejects_source_digest_or_graph_fingerprint_drift(tmp_path: Path) -> None:
    source = _write_splits(tmp_path / "data", num_classes=2)["train"]
    dataset = MolecularGraphDataset.from_csv(
        source, num_classes=2, expected_split="train"
    )
    cache = tmp_path / "train.pt"
    save_molecular_graph_cache(dataset, cache, split_name="train")

    with pytest.raises(ValueError, match="source CSV SHA256 mismatch"):
        load_molecular_graph_cache(cache, expected_source_sha256="0" * 64)

    payload = torch.load(cache, map_location="cpu", weights_only=True)
    payload["graph_sha256s"][0] = "0" * 64
    torch.save(payload, cache)
    with pytest.raises(ValueError, match="graph fingerprint mismatch"):
        load_molecular_graph_cache(cache)


def test_cli_and_slurm_wrapper_expose_required_contract() -> None:
    args = build_parser().parse_args(
        [
            "--config",
            "configs/hpc.yaml",
            "--dataset",
            "tastemolnet",
            "--data-dir",
            "/data/taste",
            "--output-dir",
            "/data/cache",
        ]
    )
    assert args.config == ["configs/hpc.yaml"]
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "scripts/slurm/build_molecular_graph_cache.sh"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --partition=A800" in wrapper
    assert "#SBATCH --gres=gpu:a800:1" in wrapper
    assert "source ~/.bashrc" in wrapper
    assert "conda activate smiles_pip118" in wrapper
    assert "cd /share/home/u20526/czx/counterfactual-subgraph" in wrapper
    assert "export PYTHONPATH=$PWD" in wrapper
    assert "--config configs/hpc.yaml" in wrapper
    assert '--dataset "${DATASET}"' in wrapper
    assert '--data-dir "${DATA_DIR}"' in wrapper
    assert '--output-dir "${OUTPUT_DIR}"' in wrapper
