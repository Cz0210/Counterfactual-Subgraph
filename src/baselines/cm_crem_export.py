"""Record-only CM-CReM CSV/LaTeX/figure export and offline replot.

The driver owns scientific saved-record audit, funnel/provenance and publication.
This module never emits scientific PASS or imports oracle/OT/generator/selector
entrypoints during a replot. The selection module import provides data types only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.baselines.cm_crem_selection import K_MAX, PAPER_LABEL, PrefixEvaluation, canonical_sha256


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _csv_value(value: Any) -> Any:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value) or value == -math.inf:
            raise ValueError("Cannot export failed or malformed numeric values")
        return "inf" if value == math.inf else repr(value)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("CSV schema must not disappear for zero-candidate outcomes")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: _csv_value(value) for key, value in row.items()} for row in rows)


def _file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()


def _metadata(*, dataset: str, oracle: str) -> None:
    if dataset not in {"bace", "tastemolnet"} or oracle != "gine":
        raise ValueError("CM-CReM export is restricted to BACE/TasteMolNet original frozen GINE")


def export_results(evaluation: PrefixEvaluation | Mapping[str, Any], output_root: str | Path, *, dataset: str,
                   oracle: str = "gine", fixture: bool = False, make_figures: bool = True) -> dict[str, Any]:
    """Write new run-root/results records; never overwrite an existing export.

    Synthetic fixture outputs are explicitly labelled in every CSV/figure.
    final_audit.json is deliberately left to the independent scientific auditor.
    """
    _metadata(dataset=dataset, oracle=oracle)
    # A driver may persist/reload the typed result between compute and export.
    # Verify its binding before creating any output directory.
    if not isinstance(evaluation, PrefixEvaluation):
        evaluation = PrefixEvaluation.from_dict(evaluation)
    evaluation.selection.validate()
    root = Path(output_root).expanduser().resolve() / "results"
    root.mkdir(parents=True, exist_ok=False)
    source = root / "source_csv"
    source.mkdir()
    common = {"dataset": dataset, "oracle": oracle, "method": PAPER_LABEL,
        "fixture": fixture, "contract_sha256": evaluation.contract_sha256,
        "selection_freeze_sha256": evaluation.selection.freeze_sha256}
    def tagged(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [{**common, **row} for row in rows]
    metrics = evaluation.prefix_metrics()
    files = {"prefix_metrics.csv": tagged(metrics),
             "figure3_coverage_cost_vs_k.csv": tagged(metrics),
             "parent_best_distances.csv": tagged(evaluation.parent_best_rows())}
    for k in (10, 20):
        files[f"figure4_k{k}_exact.csv"] = tagged(evaluation.exact_ecdf(k))
        files[f"table2_k{k}.csv"] = tagged([metrics[k-1]])
    for filename, rows in files.items():
        _write_csv(source / filename, rows)
    manifest = {"schema_version": "cm_crem_export_records_v1", "status": "EXPORTED_RECORDS",
        **common, "scientific_pass_claimed": False, "frozen_pool_sha256": evaluation.selection.frozen_pool_sha256,
        "test_matrix_sha256": evaluation.test_matrix_sha256,
        "parent_ids_sha256": canonical_sha256(list(evaluation.parent_ids)),
        "base_parent_count": len(evaluation.parent_ids),
        "theta": evaluation.selection.theta, "cap": evaluation.selection.cap,
        "source_files": {filename: _file_sha(source/filename) for filename in sorted(files)},
        "unavailable_files_owned_by_driver": ["candidate_funnel.csv", "candidate_provenance.csv",
            "budget_and_timing.json", "final_audit.json"],
        "interpretation": "Generated full-graph endpoint prototypes, not reusable deletion rules."}
    _write_json(root / "export_manifest.json", manifest)
    if make_figures:
        plotted = replot(source, root / "figures", dataset=dataset, oracle=oracle)
        return {**manifest, "source_csv": str(source), "figures": plotted["output_files"]}
    return {**manifest, "source_csv": str(source), "figures": [], "figure_status": "NOT_RENDERED"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty result CSV: {path}")
    return rows


def _number(row: Mapping[str, str], name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"Invalid plotted {name}")
    return value


def _latex(value: str) -> str:
    return "".join({"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&",
        "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}"}.get(c, c) for c in value)


def _table(path: Path, row: Mapping[str, str], *, fixture: bool) -> None:
    median = row["conditional_median_cost"]
    median_text = "N/A" if median == "N/A" else format(float(median), ".6g")
    label = PAPER_LABEL + (" (synthetic fixture)" if fixture else "")
    content = (
        "% Record-only CM-CReM table; no scientific PASS is implied.\n"
        "\\begin{tabular}{lrrrr}\n\\hline\n"
        "Method & Effective K & Coverage & Capped mean & Conditional median \\\\\n\\hline\n"
        f"{_latex(label)} & {int(row['effective_k'])} & {float(row['coverage']):.4f} & "
        f"{float(row['fixed_capped_mean_cost']):.6g} & {median_text} \\\\\n"
        "\\hline\n\\end{tabular}\n"
    )
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


def replot(source_csv: str | Path, output_dir: str | Path, *, dataset: str,
           oracle: str = "gine") -> dict[str, Any]:
    """Offline, hash-bound CSV-only replot; no selection/OT/model execution."""
    _metadata(dataset=dataset, oracle=oracle)
    source = Path(source_csv).expanduser().resolve(strict=True)
    manifest_path = source.parent / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "cm_crem_export_records_v1" or manifest.get("dataset") != dataset or manifest.get("oracle") != oracle:
        raise ValueError("CSV manifest does not match the requested original-GINE panel")
    expected = {"prefix_metrics.csv", "parent_best_distances.csv", "figure3_coverage_cost_vs_k.csv",
                "figure4_k10_exact.csv", "figure4_k20_exact.csv", "table2_k10.csv", "table2_k20.csv"}
    if set(manifest.get("source_files", {})) != expected:
        raise ValueError("Incomplete or unexpected CM-CReM CSV inventory")
    records = {}
    for filename, expected_sha in manifest["source_files"].items():
        path = source / filename
        if _file_sha(path) != expected_sha:
            raise ValueError(f"Result CSV changed: {filename}")
        rows = _read_csv(path)
        for row in rows:
            if any(row[name] != str(manifest[name]) for name in ("dataset", "oracle", "method", "contract_sha256", "selection_freeze_sha256")):
                raise ValueError("CSV mixed an incompatible dataset/oracle/contract/freeze")
            if row["fixture"] != ("true" if manifest["fixture"] else "false"):
                raise ValueError("Fixture label mismatch")
        records[filename] = rows
    metrics = records["figure3_coverage_cost_vs_k.csv"]
    if metrics != records["prefix_metrics.csv"] or [int(r["k"]) for r in metrics] != list(range(1, K_MAX+1)):
        raise ValueError("Figure3 must use the exact complete K1..20 prefix table")
    for k in (10, 20):
        if records[f"table2_k{k}.csv"] != [metrics[k-1]]:
            raise ValueError("Table2 does not match the frozen prefix")
    # Matplotlib is deliberately lazy: numeric selection and CSV validation do
    # not need plotting dependencies or a display server.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixture = manifest["fixture"]
    label = f"{dataset.upper()} | original frozen GINE"
    annotation = "SYNTHETIC FIXTURE - NOT SCIENTIFIC RESULTS" if fixture else "CM-CReM-Global | generated full-graph prototypes"
    outputs: list[str] = []
    style = {"color": "#167D9A", "linestyle": (0, (6, 2, 1, 2)), "linewidth": 2.0}
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "font.family": "DejaVu Sans"}):
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
        x = [int(row["k"]) for row in metrics]
        for ax, column, ylabel in zip(axes, ("coverage", "fixed_capped_mean_cost"),
                                     ("Coverage at frozen theta", "Fixed-denominator capped mean cost")):
            y = [_number(row, column) for row in metrics]
            ax.plot(x, y, marker="o", markersize=3, label=PAPER_LABEL, **style)
            ax.set(xlabel="Requested K (at most K prototypes)", ylabel=ylabel, xlim=(1, 20))
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.grid(alpha=.2)
            ax.legend(frameon=False, fontsize=8)
            if column == "coverage":
                ax.set_ylim(0, min(1.02, max(.05, max(y)*1.12)))
            else:
                ax.set_ylim(0, max(float(manifest["cap"])*1.08, max(y)*1.12))
        fig.suptitle(f"Figure 3 - {label}", fontsize=12)
        fig.text(.5, .02, annotation, ha="center", fontsize=8, color="#5B6570")
        fig.tight_layout(rect=(0, .07, 1, .94))
        for extension in ("png", "pdf"):
            path = output/f"figure3_coverage_cost_vs_k.{extension}"
            fig.savefig(path, dpi=200)
            outputs.append(str(path))
        plt.close(fig)
        for k in (10, 20):
            rows = records[f"figure4_k{k}_exact.csv"]
            xs = [_number(row, "distance") for row in rows]
            ys = [_number(row, "coverage") for row in rows]
            if xs != sorted(set(xs)) or any(not 0 <= y <= 1 for y in ys) or ys != sorted(ys):
                raise ValueError("ECDF must contain ordered exact finite knots and valid coverage")
            fig, ax = plt.subplots(figsize=(6.3, 4.1))
            right = max(xs[-1], float(manifest["cap"]), float(manifest["theta"])) * 1.04
            plot_x = xs + [right] if xs[-1] < right else xs
            plot_y = ys + [ys[-1]] if xs[-1] < right else ys
            ax.step(plot_x, plot_y, where="post", label=PAPER_LABEL, **style)
            ax.axvline(float(manifest["theta"]), color="#747C87", linestyle=":", linewidth=1, label="Frozen theta")
            ax.set(xlabel="Uncapped parent-best valid WNode distance", ylabel="Coverage (all base parents)",
                   title=f"Figure 4 - {label} - K={k}", xlim=(0, right), ylim=(0, 1.02))
            ax.grid(alpha=.2)
            ax.legend(frameon=False, fontsize=9)
            ax.text(.99, .03, f"Finite recourse: {rows[0]['finite_recourse_count']}/{rows[0]['base_parent_count']}",
                    transform=ax.transAxes, ha="right", fontsize=9)
            fig.text(.5, .02, annotation, ha="center", fontsize=8, color="#5B6570")
            fig.tight_layout(rect=(0, .06, 1, 1))
            for extension in ("png", "pdf"):
                path = output/f"figure4_k{k}.{extension}"
                fig.savefig(path, dpi=200)
                outputs.append(str(path))
            plt.close(fig)
            path = output/f"table2_k{k}.tex"
            _table(path, records[f"table2_k{k}.csv"][0], fixture=fixture)
            outputs.append(str(path))
    receipt = {"schema_version": "cm_crem_replot_inputs_v1", "status": "REPLOTTED_RECORDS",
        "dataset": dataset, "oracle": oracle, "fixture": fixture, "scientific_pass_claimed": False,
        "source_manifest": str(manifest_path), "source_manifest_sha256": _file_sha(manifest_path),
        "source_files": {str(source/name): sha for name, sha in manifest["source_files"].items()},
        "contract_sha256": manifest["contract_sha256"], "parent_ids_sha256": manifest["parent_ids_sha256"],
        "oracle_calls": 0, "ot_calls": 0, "selector_calls": 0, "network_calls": 0,
        "output_files": outputs}
    _write_json(output/"replot_inputs.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CM-CReM saved-CSV offline replot (no oracle/OT/selector)")
    commands = parser.add_subparsers(dest="command", required=True)
    plot = commands.add_parser("replot", help="Render authenticated saved result CSVs")
    plot.add_argument("--source-csv", type=Path, required=True)
    plot.add_argument("--output-dir", type=Path, required=True)
    plot.add_argument("--dataset", choices=("bace", "tastemolnet"), required=True)
    plot.add_argument("--oracle", choices=("gine",), default="gine")
    args = parser.parse_args(argv)
    print(json.dumps(replot(args.source_csv, args.output_dir, dataset=args.dataset, oracle=args.oracle), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
