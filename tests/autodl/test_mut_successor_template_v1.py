from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.autodl_mut_next_stage_executor_v1 import build_successor_spec
from src.utils.autodl_mut_successor_template_v1 import (
    MutSuccessorTemplateError,
    render_successor_template,
    template_placeholders,
    validate_exact_mut_successor_template,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = PROJECT_ROOT / "configs/autodl/mut_next_stage_executor_v1.template.json"


def _template() -> dict[str, object]:
    value = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _bindings(tmp_path: Path) -> dict[str, str]:
    predecessor = tmp_path / "ab/task-spec.json"
    predecessor.parent.mkdir(parents=True)
    predecessor.write_text("{}\n", encoding="utf-8")
    control = tmp_path / "ab/control"
    output = tmp_path / "ab/output"
    return {
        "__ADOPTION_ROOT__": str(tmp_path / "successor/adoption"),
        "__EXECUTION_COMMIT__": "a" * 40,
        "__EXECUTOR_LEASE_PATH__": str(tmp_path / "successor/owner.lease"),
        "__EXECUTOR_RUNTIME_ROOT__": str(tmp_path / "successor"),
        "__EXPORT_ROOT__": str(tmp_path / "successor/export"),
        "__MATRIX_AUTHORITY_ROOT__": str(tmp_path / "control/fast16_matrix_authority"),
        "__MATRIX_OUTPUT_ROOT__": str(tmp_path / "paper_matrix"),
        "__NEXT_ACTION_PATH__": str(tmp_path / "continuation/next_action.json"),
        "__OWNER_REGISTRY__": str(
            tmp_path / "control/final16-owner-registry/current.json"
        ),
        "__PREDECESSOR_TASK_ID__": "mut_same_contract_ab",
        "__PREDECESSOR_TASK_SPEC__": str(predecessor),
        "__PREDECESSOR_TERMINAL__": str(control / "terminal.json"),
        "__PROJECT_ROOT__": str(tmp_path / "immutable-project"),
        "__PUBLISHER_ID__": "mut-successor-publisher",
        "__PUBLISHER_LEASE_PATH__": str(tmp_path / "publisher/owner.lease"),
        "__PUBLISHER_LOCATOR__": str(tmp_path / "publisher/locator.json"),
        "__PUBLISH_ROOT__": str(tmp_path / "successor/publish"),
        "__PYTHON__": str(tmp_path / "env/bin/python"),
        "__ROUTE_B_ROOT__": str(tmp_path / "successor/route-b"),
        "__STANDARDIZED_ROOT__": str(tmp_path / "successor/standardized"),
        "__TASK_ID__": "mut-post-ab-successor",
        "__TRACE_MODE_GATE__": str(
            output / "trace_on_off_500_step_equivalence.json"
        ),
    }


def test_exact_template_has_no_superseded_instrumentation_or_memory_gate(
    tmp_path: Path,
) -> None:
    rendered = render_successor_template(_template(), _bindings(tmp_path))
    validated = validate_exact_mut_successor_template(rendered)
    argv = validated["adoption_pipeline"][0]["argv"]
    joined = " ".join(argv)
    assert "run_mut_same_contract_adoption_v1.py" in joined
    assert "--same-contract-gate" in argv
    assert "--ab-task-spec" in argv
    assert "--ab-owner-terminal" in argv
    assert "--instrumentation-gate" not in argv
    assert "--memory-receipt" not in argv
    assert "--trace-code-audit" not in argv


def test_rendered_template_builds_exact_ordered_successor_spec(tmp_path: Path) -> None:
    rendered = validate_exact_mut_successor_template(
        render_successor_template(_template(), _bindings(tmp_path))
    )
    spec = build_successor_spec(rendered, check_files=False)
    assert [row["stage"] for row in spec["adoption_pipeline"]] == [
        "HISTORICAL_50K_ADOPTION",
        "STANDARDIZED_EVALUATION",
        "FIGURE_TABLE_EXPORT",
        "MATRIX_PUBLISH",
    ]
    assert spec["route_b_pipeline"][0]["expected_terminal_status"] == [
        "BLOCKED_ADAPTER_MISSING"
    ]


def test_renderer_rejects_missing_binding(tmp_path: Path) -> None:
    bindings = _bindings(tmp_path)
    bindings.pop(next(iter(template_placeholders(_template()))))
    with pytest.raises(MutSuccessorTemplateError, match="missing"):
        render_successor_template(_template(), bindings)
