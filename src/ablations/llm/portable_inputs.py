"""Read exact L0 input bytes on HPC without rewriting AutoDL manifests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from src.ablations.llm.contracts import canonical_json_sha256
from src.eval.bace_frozen_gnn_contracts import sha256_file

SCHEMA = "bace_l0_cpu_portable_inputs_v1"


class PortableInputs:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve(strict=True)
        self.manifest = json.loads((self.root / "portable_manifest.json").read_text())
        manifest = self.manifest
        if (manifest.get("schema_version") != SCHEMA
                or manifest.get("manifest_sha256") != canonical_json_sha256({
                    k: v for k, v in manifest.items() if k != "manifest_sha256"})
                or manifest.get("variant") != "BRICS_FIXED"
                or manifest.get("original_manifests_modified") is not False
                or manifest.get("model_weights_copied") is not False):
            raise ValueError("L0_PORTABLE_MANIFEST_MISMATCH")

    def resolve(self, identity: Mapping[str, Any]) -> Path:
        original = str(identity["path"])
        entry = self.manifest["source_files"].get(original)
        if entry is None or entry["sha256"] != identity["sha256"]:
            raise ValueError("L0_PORTABLE_SOURCE_BINDING_MISMATCH")
        relative = Path(entry["relative"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("L0_PORTABLE_PATH_ESCAPE")
        path = self.root / relative
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("L0_PORTABLE_SYMLINK")
        path.resolve(strict=True).relative_to(self.root)
        if (not path.is_file() or path.stat().st_size != entry["size"]
                or sha256_file(path) != entry["sha256"]):
            raise ValueError("L0_PORTABLE_BYTES_CHANGED")
        return path

    def task_spec_path(self) -> Path:
        return self.resolve(self.manifest["task_spec"])
