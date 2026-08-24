#!/usr/bin/env python3
"""Print one root-cause acceleration monitor snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from src.utils.autodl_progress_health import route_viability_for_progress_state


def _legacy_route_viability(row: Mapping[str, Any]) -> Any:
    if row.get("route_viability") is not None:
        return row.get("route_viability")
    progress_state = row.get("scientific_progress_state", row.get("health_state"))
    if not isinstance(progress_state, str):
        return None
    try:
        return route_viability_for_progress_state(progress_state)
    except ValueError:
        return None


def render_table(state: Mapping[str, Any]) -> str:
    """Render the non-owning monitor fields without conflating their meaning."""

    lines = [
        "Root-cause acceleration monitor",
        (
            f"controller_id={state.get('controller_id')} "
            f"controller_pid={state.get('controller_pid')} "
            "controller_process_alive="
            f"{state.get('controller_process_alive', 'UNKNOWN')} "
            f"updated_at={state.get('updated_at')}"
        ),
        "",
        (
            "task_id\tcontroller_process_alive\tscientific_worker_alive\t"
            "scientific_progress_state\troute_viability\tcompleted/total\teta_hours"
        ),
    ]
    tasks = state.get("tasks")
    if not isinstance(tasks, Mapping):
        tasks = {}
    for task_id, task in sorted(tasks.items(), key=lambda item: str(item[0])):
        row = task if isinstance(task, Mapping) else {}
        controller_alive = row.get(
            "controller_process_alive",
            state.get("controller_process_alive", "UNKNOWN"),
        )
        worker_alive = row.get("scientific_worker_alive", row.get("pid_alive"))
        progress_state = row.get(
            "scientific_progress_state", row.get("health_state")
        )
        lines.append(
            "\t".join(
                (
                    str(task_id),
                    str(controller_alive),
                    str(worker_alive),
                    str(progress_state),
                    str(_legacy_route_viability(row)),
                    f"{row.get('completed')}/{row.get('total')}",
                    str(row.get("eta_hours")),
                )
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--format", choices=("json", "table"), default="json")
    args = parser.parse_args()
    state = json.loads((args.root.expanduser().resolve(strict=True) / "state.json").read_text())
    if args.format == "table":
        print(render_table(state), end="")
    else:
        print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
