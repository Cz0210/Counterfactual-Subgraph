#!/usr/bin/env python3
"""Serve or print the read-only AutoDL four-by-four controller dashboard."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

from scripts.autodl.four_by_four_dashboard import (
    DEFAULT_NAMESPACE,
    DEFAULT_STALE_SECONDS,
    DashboardHTTPServer,
    DashboardServerConfig,
    collect_dashboard_snapshot,
    is_loopback_host,
    make_handler,
    render_terminal,
)
from src.utils.autodl_runtime import (
    AutoDLRuntimeError,
    build_runtime_layout,
    resolve_project_root,
    select_data_root,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--control-root", type=Path)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument(
        "--stale-seconds", type=float, default=DEFAULT_STALE_SECONDS
    )
    commands = parser.add_subparsers(dest="command", required=True)
    once = commands.add_parser("once", help="print one current snapshot")
    once.add_argument("--format", choices=("table", "json"), default="table")
    serve = commands.add_parser("serve", help="serve the loopback-only web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8766)
    serve.add_argument("--interval", type=float, default=5.0)
    return parser


def _layout(args: argparse.Namespace):
    project_root = resolve_project_root(args.project_root)
    data_root = select_data_root(project_root, explicit=args.data_root)
    return build_runtime_layout(
        project_root=project_root,
        data_root=data_root,
        control_root=args.control_root,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.stale_seconds < 1:
            raise AutoDLRuntimeError("--stale-seconds must be at least 1")
        if args.command == "serve":
            if not is_loopback_host(args.host):
                raise AutoDLRuntimeError(
                    "Dashboard refuses non-loopback --host; use an SSH tunnel"
                )
            if not 1 <= args.port <= 65535:
                raise AutoDLRuntimeError("--port must be in 1..65535")
            if args.interval < 1:
                raise AutoDLRuntimeError("--interval must be at least 1 second")
        layout = _layout(args)
        if args.command == "once":
            snapshot = collect_dashboard_snapshot(
                layout,
                namespace=args.namespace,
                stale_seconds=args.stale_seconds,
            )
            if args.format == "json":
                print(
                    json.dumps(
                        snapshot, ensure_ascii=False, indent=2, sort_keys=True
                    )
                )
            else:
                print(render_terminal(snapshot), end="")
            return 0
        config = DashboardServerConfig(
            layout=layout,
            namespace=args.namespace,
            interval_seconds=args.interval,
            stale_seconds=args.stale_seconds,
        )
        server = DashboardHTTPServer((args.host, args.port), make_handler(config))
        print(
            f"[AUTODL_FOUR_BY_FOUR_DASHBOARD] "
            f"http://{args.host}:{args.port} namespace={args.namespace}",
            flush=True,
        )
        try:
            server.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0
    except (AutoDLRuntimeError, OSError, ValueError) as exc:
        print(f"AUTODL_DASHBOARD_FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
