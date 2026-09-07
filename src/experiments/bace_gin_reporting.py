"""Offline saved-parent reducer shared by Figure3, Figure4 and Table2.

This checks display consistency, not molecular validity or independent science.
No oracle, selector, model, or distance solver is called here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

METHODS = ("Ours", "GlobalGCE", "GCFExplainer", "ComRecGC")
STYLES = {"Ours": ("#161616", "s"), "GlobalGCE": ("#cf3030", "x"),
          "GCFExplainer": ("#7540a4", "^"), "ComRecGC": ("#288641", "*")}
INPUTS = ("bace_gin_fixed141_table2.csv", "bace_gin_fixed141_figure3.csv",
          "bace_gin_fixed141_figure4_exact_ecdf.csv", "bace_gin_parent_best_distances.csv")


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    if value is None or str(value) in ("", "None", "N/A"):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("NONFINITE_DISPLAY_VALUE")
    return result


def assert_same(old, new, field):
    a, b = number(old), number(new)
    if (a is None) != (b is None) or (a is not None and not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-14)):
        raise ValueError(f"SOURCE_REDUCER_CONFLICT:{field}:{old!r}!={new!r}")


def reduce_parent_prefix(rows, *, theta, cap):
    """Original fixed capped mean and conditional median; missing min is +inf."""
    if not rows or len({r["parent_id"] for r in rows}) != len(rows):
        raise ValueError("EMPTY_OR_DUPLICATE_PARENT_PREFIX")
    effective = {int(r["K_effective"]) for r in rows}
    requested = {int(r["K_requested"]) for r in rows}
    if len(effective) != 1 or len(requested) != 1:
        raise ValueError("INCONSISTENT_PREFIX_BUDGET")
    k, m = requested.pop(), effective.pop()
    if not (1 <= m <= min(k, 20)):
        raise ValueError("INVALID_AT_MOST_K")
    values = []
    for row in rows:
        value = number(row["best_valid_distance"])
        if value is not None and (value < 0 or int(row["pred_before"]) != 1):
            raise ValueError("INVALID_SAVED_STRICT_FLIP_MINIMUM")
        values.append(math.inf if value is None else value)
    finite = sorted(v for v in values if math.isfinite(v))
    n = len(values)
    covered = sum(v <= theta for v in values)
    median = (finite[(len(finite)-1)//2] + finite[len(finite)//2])/2 if finite else None
    return {"cohort": "fixed141", "denominator": n, "K_requested": k, "K_effective": m,
            "covered_count": covered, "coverage": covered/n, "coverage_percent": 100*covered/n,
            "finite_strict_flip_count": len(finite), "strict_flip_availability": len(finite)/n,
            "fixed_capped_mean": sum(min(v, cap) for v in values)/n,
            "conditional_median": median, "theta_star": theta, "cost_cap": cap}


def derive(source: Path, *, expected_parents=141):
    """Fail closed on panel/source conflict before writing paper figures."""
    table, old3, old4, parents = [read_csv(source/name) for name in INPUTS]
    tables = {r["method"]: r for r in table}
    if len(tables) != len(table) or set(tables) != set(METHODS):
        raise ValueError("FOUR_METHOD_STATUS_ROWS_REQUIRED")
    ready = {m for m, r in tables.items() if r["state"] == "EVALUATED"}
    if not ready:
        raise ValueError("NO_EVALUATED_RESULTS")
    grouped = defaultdict(list)
    for row in parents:
        if row["method"] not in ready:
            raise ValueError("PARENT_ROWS_FOR_UNFINISHED_METHOD")
        grouped[row["method"], int(row["K_requested"])].append(row)
    indexed3 = {(r["method"], int(r["K_requested"])): r for r in old3}
    if len(indexed3) != len(old3) or set(indexed3) != set(grouped):
        raise ValueError("FIGURE3_PREFIX_DOMAIN_MISMATCH")
    curves, ecdf = [], {10: [], 20: []}
    canonical_ids = predictions = common_contract = None
    for method in METHODS:
        if method not in ready:
            continue
        t = tables[method]
        theta, cap = float(t["theta_star"]), float(t["cost_cap"])
        if not (0 < theta <= cap):
            raise ValueError("INVALID_FROZEN_DISTANCE_CONTRACT")
        if common_contract is not None and common_contract != (theta, cap):
            raise ValueError("CROSS_METHOD_THRESHOLD_OR_COST_CONTRACT_MISMATCH")
        common_contract = theta, cap
        if int(t["K_requested"]) != 10 or int(t["denominator"]) != expected_parents:
            raise ValueError("TABLE2_K10_FIXED_BASE_REQUIRED")
        previous = None
        for k in range(1, 21):
            current = grouped.get((method, k), [])
            ids = {r["parent_id"] for r in current}
            if len(current) != expected_parents or len(ids) != expected_parents:
                raise ValueError("INCOMPLETE_FIXED_PARENT_PREFIX")
            if canonical_ids is None:
                canonical_ids = ids
                predictions = {r["parent_id"]: int(r["pred_before"]) for r in current}
            if ids != canonical_ids or {r["parent_id"]: int(r["pred_before"]) for r in current} != predictions:
                raise ValueError("COHORT_OR_CLASSIFIER_CHANGED_ACROSS_PANELS")
            keyed = {r["parent_id"]: (math.inf if number(r["best_valid_distance"]) is None
                     else float(r["best_valid_distance"])) for r in current}
            if previous is not None and any(keyed[p] > previous[p] for p in ids):
                raise ValueError("SAVED_PREFIX_NOT_NESTED")
            result = {"method": method, **reduce_parent_prefix(current, theta=theta, cap=cap)}
            for field in result:
                if field in ("method", "cohort", "coverage_percent"):
                    continue
                assert_same(indexed3[method, k].get(field), result[field], f"figure3:{method}:{k}:{field}")
                if k == 10:
                    assert_same(t.get(field), result[field], f"table2:{method}:{field}")
            curves.append(result)
            if k in ecdf:
                old_points = [r for r in old4 if r["method"] == method and int(r["K_requested"]) == k]
                if not old_points:
                    raise ValueError("EXACT_ECDF_DOMAIN_MISSING")
                xs = [float(r["threshold"]) for r in old_points]
                if len(set(xs)) != len(xs) or min(xs) != 0 or max(xs) < theta or theta not in xs:
                    raise ValueError("EXACT_ECDF_DOMAIN_INVALID")
                points = sorted({*xs, *(v for v in keyed.values() if math.isfinite(v) and v <= max(xs))})
                for old in old_points:
                    count = sum(v <= float(old["threshold"]) for v in keyed.values())
                    assert_same(old["covered_count"], count, f"figure4:{method}:{k}:count")
                    assert_same(old["coverage"], count/expected_parents, f"figure4:{method}:{k}:coverage")
                ecdf[k].extend({"method": method, "cohort": "fixed141", "K_requested": k,
                    "K_effective": result["K_effective"], "denominator": expected_parents,
                    "threshold": x, "covered_count": sum(v <= x for v in keyed.values()),
                    "coverage": sum(v <= x for v in keyed.values())/expected_parents,
                    "coverage_percent": 100*sum(v <= x for v in keyed.values())/expected_parents}
                    for x in points)
            previous = keyed
        available = result["K_effective"]
        if any(r["K_effective"] != min(r["K_requested"], available) for r in curves if r["method"] == method):
            raise ValueError("AT_MOST_K_PLATEAU_INCONSISTENT")
    table_rows = []
    for method in METHODS:
        if method in ready:
            table_rows.append({**next(r for r in curves if r["method"] == method and r["K_requested"] == 10), "state": "EVALUATED"})
        else:
            state = tables[method]["state"]
            table_rows.append({"method": method, "state": "BLOCKED" if state.startswith("BLOCKED") else "PENDING",
                               "source_state": state, "K_requested": 10, "cohort": "fixed141"})
    return {"figure3": curves, "figure4": ecdf, "table2": table_rows, "ready": sorted(ready),
            "theta": common_contract[0], "cap": common_contract[1], "denominator": expected_parents}


def render(source: Path, output: Path, *, version_label="BACE-GIN-fixed-pool-v1", expected_parents=141, display_labels=None):
    data = derive(source, expected_parents=expected_parents)
    labels = {m: (display_labels or {}).get(m, m) for m in METHODS}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    csv_root = output/"source_csv"
    csv_root.mkdir(exist_ok=True)
    write_csv(csv_root/"figure3_k1_20.csv", data["figure3"])
    for k in (10, 20):
        write_csv(csv_root/f"figure4_k{k}_exact_ecdf.csv", data["figure4"][k])
    write_csv(csv_root/"table2_k10.csv", data["table2"])
    missing = "; ".join(f"{labels[r['method']]}: {r['state']}" for r in data["table2"] if r["state"] != "EVALUATED")
    subtitle = ("PARTIAL · " if missing else "") + f"{version_label} · fixed {expected_parents} parents"
    footer = "Not plotted: " + missing if missing else "All four methods have evaluated saved records."
    footer += "\nPost-hoc development; no unseen-test or end-to-end retraining claim."
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 7.4), sharex=True)
    for method in METHODS:
        rows = [r for r in data["figure3"] if r["method"] == method]
        if not rows:
            continue
        color, marker = STYLES[method]
        for axis, key in zip(axes, ("coverage_percent", "fixed_capped_mean")):
            axis.plot([r["K_requested"] for r in rows], [r[key] for r in rows], color=color,
                      marker=marker, markersize=4.5, linewidth=1.5, label=labels[method])
    axes[0].set(ylabel="Coverage (%)", ylim=(0, max(5, max(r["coverage_percent"] for r in data["figure3"])*1.1)))
    axes[1].set(ylabel="Fixed-capped mean WNode cost", xlabel="Rule budget K (at most)",
                ylim=(0, data["cap"]*1.05), xticks=(1, 5, 10, 15, 20))
    axes[0].legend(loc="best")
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle(subtitle, fontsize=12)
    fig.text(.03, .013, footer, fontsize=8, va="bottom")
    fig.tight_layout(rect=(0, .075, 1, .96))
    for ext in ("pdf", "png"):
        fig.savefig(output/f"figure3_bace_gin_fixed141.{ext}", dpi=180)
    plt.close(fig)
    for k in (10, 20):
        fig, axis = plt.subplots(figsize=(8.4, 4.9))
        for method in METHODS:
            rows = [r for r in data["figure4"][k] if r["method"] == method]
            if rows:
                color, marker = STYLES[method]
                axis.step([r["threshold"] for r in rows], [r["coverage_percent"] for r in rows],
                          where="post", color=color, label=labels[method], linewidth=1.5)
                theta_row = next(r for r in rows if r["threshold"] == data["theta"])
                axis.plot([data["theta"]], [theta_row["coverage_percent"]], marker=marker, color=color, markersize=5)
        upper = max(r["threshold"] for r in data["figure4"][k])
        axis.set(xlabel=f"WNode threshold · exact ECDF · K={k}", ylabel="Coverage (%)", xlim=(0, upper),
                 ylim=(0, max(5, max(r["coverage_percent"] for r in data["figure4"][k])*1.1)),
                 title=subtitle + ("\nPrimary K=10" if k == 10 else "\nAuxiliary K=20"))
        axis.axvline(data["theta"], color="0.6", linewidth=.8, linestyle=":")
        axis.grid(alpha=.2)
        axis.legend(loc="best")
        fig.text(.03, .014, footer, fontsize=8, va="bottom")
        fig.tight_layout(rect=(0, .105, 1, 1))
        for ext in ("pdf", "png"):
            fig.savefig(output/f"figure4_bace_gin_fixed141_k{k}.{ext}", dpi=180)
        plt.close(fig)
    display = []
    for row in data["table2"]:
        state = row["state"]
        def show(key, digits=".6g"):
            if state != "EVALUATED":
                return state
            return "N/A" if row.get(key) is None else format(row[key], digits)
        display.append([labels[row["method"]], state, show("K_effective", "d"), show("coverage_percent", ".4f"),
                        show("fixed_capped_mean"), show("conditional_median")])
    fig, axis = plt.subplots(figsize=(10.2, 3.3))
    axis.axis("off")
    axis.set_title(subtitle + "\nTable 2 · K=10", fontsize=12)
    table = axis.table(cellText=display, colLabels=("Method", "State", "K eff.", "Coverage (%)", "Capped mean", "Conditional median"), loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.65)
    fig.text(.03, .012, "EVALUATED is a saved-result status, not a new scientific PASS.\n" + footer, fontsize=8)
    fig.tight_layout(rect=(0, .17, 1, 1))
    fig.savefig(output/"table2_bace_gin_fixed141.pdf")
    plt.close(fig)
    tex = ["% Saved parent minima; post-hoc version explicitly named; no metric redefinition.",
           r"\begin{tabular}{llrrrr}", r"Method & State & K eff. & Coverage (\%) & Capped mean & Conditional median \\"]
    tex.extend(" & ".join(row) + r" \\" for row in display)
    tex.append(r"\end{tabular}")
    (output/"table2_bace_gin_fixed141.tex").write_text("\n".join(tex)+"\n")
    manifest = {"state": "DISPLAY_SOURCE_REDUCTION_CONSISTENT", "version_label": version_label, "display_labels": labels,
        "source_root": str(source), "primary_cohort": "fixed141", "denominator": expected_parents,
        "figure4_primary_k": 10, "figure4_auxiliary_k": 20, "table2_k": 10,
        "theta_star": data["theta"], "cost_cap": data["cap"], "cost_definition_changed": False,
        "coverage_csv_unit": "fraction_and_explicit_percent", "parent_reducer": "reduce_parent_prefix",
        "scientific_reaudit_performed": False, "main_matrix_written": False,
        "missing_methods_not_plotted": missing,
        "input_files": {name: {"path": str(source/name), "bytes": (source/name).stat().st_size,
            "sha256": hashlib.sha256((source/name).read_bytes()).hexdigest()} for name in INPUTS}}
    (output/"display_consistency_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest
