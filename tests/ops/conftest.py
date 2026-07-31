from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml


@pytest.fixture
def base_spec(tmp_path: Path) -> dict[str, Any]:
    return {
        "task_id": "unit_task",
        "description": "Unit test task.",
        "project": {
            "local_root": str(tmp_path),
            "remote_root": "/share/home/u20526/czx/counterfactual-subgraph",
            "branch": "main",
        },
        "remote": {
            "host": "u20526@logini.tongji.edu.cn",
            "port": 10022,
            "control_socket": None,
            "conda_env": "smiles_pip118",
        },
        "git": {
            "allowed_paths": ["scripts/ops/**"],
            "commit_message": "Unit test",
            "allow_commit": False,
            "allow_push": False,
            "require_clean_allowed_paths": False,
            "dynamic_remote_paths": [],
        },
        "execution": {
            "auto_until": "local_gate",
            "stop_before": None,
            "max_auto_retries": 0,
            "poll_interval_seconds": 60,
        },
        "permissions": {
            "allow_remote_write": False,
            "allow_sbatch": False,
            "allow_gpu_smoke": False,
            "allow_full": False,
            "allow_calibration": False,
            "allow_test": False,
            "allow_finalization": False,
            "allow_overwrite": False,
            "preserve_proxy_environment": True,
        },
        "stages": [
            {
                "id": "local_gate",
                "kind": "local_command",
                "command": ["python", "-c", "print('ok')"],
                "script": None,
                "dependencies": [],
                "expected_artifacts": [],
                "gate": {
                    "json_path": None,
                    "required_marker": None,
                    "required_fields": {},
                    "forbidden_fields": {},
                },
                "resources": {
                    "name": None,
                    "tags": "local",
                    "notes": None,
                    "expected_output_root": None,
                },
                "cwd": None,
                "timeout_seconds": 60,
            }
        ],
    }


@pytest.fixture
def write_spec(tmp_path: Path):
    def _write(payload: dict[str, Any], name: str = "task.yaml") -> Path:
        path = tmp_path / name
        path.write_text(
            yaml.safe_dump(deepcopy(payload), sort_keys=False),
            encoding="utf-8",
        )
        return path

    return _write
