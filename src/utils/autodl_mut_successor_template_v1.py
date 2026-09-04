"""Render and validate the concrete Mut post-A/B successor stage template.

The generic successor executor intentionally accepts arbitrary sealed argv
lists.  This module is the dataset-specific deployment boundary: it renders a
small placeholder template, then proves that every stage invokes the reviewed
Mut CLI and consumes the preceding stage's exact output.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping, Sequence


PLACEHOLDER = re.compile(r"__[A-Z][A-Z0-9_]*__")
REQUIRED_ENVIRONMENT = {
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "RUN_GNN_ABLATION": "0",
    "RUN_LLM_ABLATION": "0",
    "TOKENIZERS_PARALLELISM": "false",
}


class MutSuccessorTemplateError(RuntimeError):
    """The rendered Mut successor template is incomplete or not canonical."""


def _tokens(value: Any) -> set[str]:
    if isinstance(value, str):
        return set(PLACEHOLDER.findall(value))
    if isinstance(value, list):
        result: set[str] = set()
        for item in value:
            result.update(_tokens(item))
        return result
    if isinstance(value, Mapping):
        result = set()
        for key, item in value.items():
            result.update(_tokens(key))
            result.update(_tokens(item))
        return result
    return set()


def template_placeholders(template: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the complete sorted placeholder set from a template."""

    return tuple(sorted(_tokens(template)))


def render_successor_template(
    template: Mapping[str, Any], bindings: Mapping[str, Any]
) -> dict[str, Any]:
    """Replace every template token exactly once at the JSON value layer.

    Values remain individual argv elements and are never interpreted by a
    shell.  Extra bindings and recursively injected placeholders are rejected.
    """

    required = set(template_placeholders(template))
    supplied = set(bindings)
    if supplied != required:
        raise MutSuccessorTemplateError(
            "template bindings changed: "
            f"missing={sorted(required - supplied)}, extra={sorted(supplied - required)}"
        )
    normalized: dict[str, str] = {}
    for key, value in bindings.items():
        if PLACEHOLDER.fullmatch(key) is None:
            raise MutSuccessorTemplateError(f"invalid binding token: {key}")
        if not isinstance(value, str) or not value:
            raise MutSuccessorTemplateError(f"binding {key} must be a nonempty string")
        if PLACEHOLDER.search(value):
            raise MutSuccessorTemplateError(
                f"binding {key} may not inject another placeholder"
            )
        normalized[key] = value

    def replace(value: Any) -> Any:
        if isinstance(value, str):
            rendered = PLACEHOLDER.sub(
                lambda match: normalized[match.group(0)], value
            )
            if PLACEHOLDER.search(rendered):  # defensive: bindings are checked above
                raise MutSuccessorTemplateError(
                    f"unresolved placeholder after rendering: {rendered}"
                )
            return rendered
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, Mapping):
            return {replace(key): replace(item) for key, item in value.items()}
        return value

    rendered = replace(dict(template))
    if not isinstance(rendered, dict):  # pragma: no cover - Mapping above guarantees this
        raise MutSuccessorTemplateError("rendered successor template is not an object")
    return rendered


def _option(argv: Sequence[str], flag: str) -> str:
    positions = [index for index, item in enumerate(argv) if item == flag]
    if len(positions) != 1:
        raise MutSuccessorTemplateError(f"{flag} must occur exactly once")
    position = positions[0]
    if position + 1 >= len(argv) or argv[position + 1].startswith("--"):
        raise MutSuccessorTemplateError(f"{flag} lacks one value")
    return argv[position + 1]


def _base(
    row: Mapping[str, Any], *, script: str, action: str | None
) -> tuple[list[str], Path, str]:
    argv = list(row.get("argv") or [])
    if len(argv) < 4 or any(not isinstance(item, str) for item in argv):
        raise MutSuccessorTemplateError(f"{row.get('stage')} argv is malformed")
    cwd = Path(str(row.get("cwd") or ""))
    expected_script = cwd / "scripts" / "autodl" / script
    prefix = [argv[0], "-I", "-B", str(expected_script)]
    common = [
        "--config",
        "configs/hpc.yaml",
        "--set",
        "inference.fallback_to_heuristic=false",
    ]
    if action is not None:
        common.append(action)
    return prefix + common, cwd, argv[0]


def _assert_row(
    row: Mapping[str, Any],
    *,
    expected_argv: Sequence[str],
    expected_terminal: Path,
    expected_status: Sequence[str],
    expected_output_root: Path,
) -> None:
    stage = str(row.get("stage") or "")
    if list(row.get("argv") or []) != list(expected_argv):
        raise MutSuccessorTemplateError(f"{stage} argv differs from pinned CLI")
    if Path(str(row.get("expected_terminal") or "")) != expected_terminal:
        raise MutSuccessorTemplateError(f"{stage} terminal is not output-bound")
    if list(row.get("expected_terminal_status") or []) != list(expected_status):
        raise MutSuccessorTemplateError(f"{stage} terminal status changed")
    if Path(str(row.get("output_root") or "")) != expected_output_root:
        raise MutSuccessorTemplateError(f"{stage} output root changed")
    if dict(row.get("environment") or {}) != REQUIRED_ENVIRONMENT:
        raise MutSuccessorTemplateError(f"{stage} deterministic environment changed")
    for field in (
        "pair_store_recomputed",
        "dbscan_recomputed",
        "calibration_used_for_selection",
        "test_used_for_selection",
    ):
        if row.get(field) is not False:
            raise MutSuccessorTemplateError(f"{stage}.{field} must remain false")


def validate_exact_mut_successor_template(
    template: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the concrete adoption-to-publication dataflow and Route B."""

    value = dict(template)
    adoption = value.get("adoption_pipeline")
    route_b = value.get("route_b_pipeline")
    if not isinstance(adoption, list) or [row.get("stage") for row in adoption] != [
        "HISTORICAL_50K_ADOPTION",
        "STANDARDIZED_EVALUATION",
        "FIGURE_TABLE_EXPORT",
        "MATRIX_PUBLISH",
    ]:
        raise MutSuccessorTemplateError("adoption stage order changed")
    if not isinstance(route_b, list) or [row.get("stage") for row in route_b] != [
        "ROUTE_B"
    ]:
        raise MutSuccessorTemplateError("Route-B stage order changed")

    adopt, standardize, export, publish = adoption
    route = route_b[0]
    adopt_root = Path(str(adopt["output_root"]))
    standard_root = Path(str(standardize["output_root"]))
    export_root = Path(str(export["output_root"]))
    publish_root = Path(str(publish["output_root"]))
    route_root = Path(str(route["output_root"]))
    if len({adopt_root, standard_root, export_root, publish_root, route_root}) != 5:
        raise MutSuccessorTemplateError("successor stage roots are not disjoint")

    prefix, cwd, python = _base(
        adopt,
        script="run_mut_same_contract_adoption_v1.py",
        action=None,
    )
    adopt_argv = list(adopt["argv"])
    adopt_expected = prefix + [
        "--ab-task-spec",
        str(value["predecessor_task_spec"]),
        "--ab-owner-terminal",
        str(value["predecessor_terminal"]),
        "--same-contract-gate",
        _option(adopt_argv, "--same-contract-gate"),
        "--authorization-receipt",
        _option(adopt_argv, "--authorization-receipt"),
        "--historical-source-root",
        (
            "/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/"
            "recovery/mutagenicity_comrecgc_lineage_v3_20260822T025620Z"
        ),
        "--completed-common-root",
        (
            "/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/"
            "repairs/mut_comrecgc_exact_multicomponent_v1_20260830T184359Z/full"
        ),
        "--output-root",
        str(adopt_root),
        "--proc-root",
        "/proc",
    ]
    trace_mode_gate = Path(_option(adopt_argv, "--same-contract-gate"))
    if trace_mode_gate.name != "trace_on_off_500_step_equivalence.json":
        raise MutSuccessorTemplateError("adoption does not consume the future A/B gate")
    authorization = Path(_option(adopt_argv, "--authorization-receipt"))
    if (
        authorization.name != "trace_on_adoption_authorization_20260901T183543Z.json"
        or authorization.parent.name != "mut_fast_accurate_v2_20260901T043250Z"
        or authorization.parent.parent.name != "mut_fast_accurate_v2"
    ):
        raise MutSuccessorTemplateError(
            "adoption does not bind the latest existing 2026-09-01 authorization"
        )
    _assert_row(
        adopt,
        expected_argv=adopt_expected,
        expected_terminal=adopt_root / "verification.json",
        expected_status=["PASS"],
        expected_output_root=adopt_root,
    )

    standard_argv = list(standardize["argv"])
    standard_prefix, standard_cwd, standard_python = _base(
        standardize,
        script="run_mut_comrecgc_parity_standardization.py",
        action=None,
    )
    if standard_cwd != cwd or standard_python != python:
        raise MutSuccessorTemplateError("successor stages changed execution checkout")
    standard_expected = standard_prefix + [
        "--source-generation-root",
        _option(standard_argv, "--source-generation-root"),
        "--upstream-root",
        _option(standard_argv, "--upstream-root"),
        "--dataset-dir",
        _option(standard_argv, "--dataset-dir"),
        "--distance-checkpoint",
        _option(standard_argv, "--distance-checkpoint"),
        "--dataset-csv",
        _option(standard_argv, "--dataset-csv"),
        "--teacher-path",
        _option(standard_argv, "--teacher-path"),
        "--molclr-root",
        _option(standard_argv, "--molclr-root"),
        "--molclr-checkpoint",
        _option(standard_argv, "--molclr-checkpoint"),
        "--thresholds-path",
        _option(standard_argv, "--thresholds-path"),
        "--historical-adoption",
        str(adopt_root / "historical_adoption.json"),
        "--output-root",
        str(standard_root),
        "--device",
        "cpu",
    ]
    _assert_row(
        standardize,
        expected_argv=standard_expected,
        expected_terminal=standard_root / "final_gate.json",
        expected_status=["PASS"],
        expected_output_root=standard_root,
    )

    export_argv = list(export["argv"])
    export_prefix, export_cwd, export_python = _base(
        export, script="reopen_mut_successor_export_v1.py", action=None
    )
    if export_cwd != cwd or export_python != python:
        raise MutSuccessorTemplateError("export changed execution checkout")
    export_expected = export_prefix + [
        "--terminal-root",
        str(standard_root),
        "--output-root",
        str(export_root),
        "--proc-root",
        "/proc",
    ]
    _assert_row(
        export,
        expected_argv=export_expected,
        expected_terminal=export_root / "terminal.json",
        expected_status=["PASS"],
        expected_output_root=export_root,
    )

    publish_argv = list(publish["argv"])
    publish_prefix, publish_cwd, publish_python = _base(
        publish, script="publish_mut_successor_v1.py", action=None
    )
    if publish_cwd != cwd or publish_python != python:
        raise MutSuccessorTemplateError("publisher changed execution checkout")
    owner_registry = _option(publish_argv, "--owner-registry")
    if not owner_registry.endswith("/control/final16-owner-registry/current.json"):
        raise MutSuccessorTemplateError("publisher does not use the canonical owner registry")
    publish_expected = publish_prefix + [
        "--terminal-root",
        str(standard_root),
        "--export-receipt",
        str(export_root / "terminal.json"),
        "--owner-registry",
        owner_registry,
        "--publisher-id",
        str(value["publisher_id"]),
        "--publisher-locator",
        str(value["publisher_locator"]),
        "--publisher-lease-path",
        _option(publish_argv, "--publisher-lease-path"),
        "--matrix-authority-root",
        str(value["matrix_authority_root"]),
        "--matrix-output-root",
        _option(publish_argv, "--matrix-output-root"),
        "--output-root",
        str(publish_root),
        "--proc-root",
        "/proc",
    ]
    _assert_row(
        publish,
        expected_argv=publish_expected,
        expected_terminal=publish_root / "terminal.json",
        expected_status=["PASS"],
        expected_output_root=publish_root,
    )

    route_prefix, route_cwd, route_python = _base(
        route, script="run_mut_route_b_closeout_v1.py", action=None
    )
    if route_cwd != cwd or route_python != python:
        raise MutSuccessorTemplateError("Route B changed execution checkout")
    route_expected = route_prefix + ["--output-root", str(route_root)]
    _assert_row(
        route,
        expected_argv=route_expected,
        expected_terminal=route_root / "terminal.json",
        expected_status=["BLOCKED_ADAPTER_MISSING"],
        expected_output_root=route_root,
    )
    return value


__all__ = [
    "MutSuccessorTemplateError",
    "REQUIRED_ENVIRONMENT",
    "render_successor_template",
    "template_placeholders",
    "validate_exact_mut_successor_template",
]
