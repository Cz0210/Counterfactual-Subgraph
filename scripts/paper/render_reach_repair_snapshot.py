#!/usr/bin/env python3
"""Render frozen v1 results and parser corrections; never imply v2 science PASS."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def corrected_llm_rows(original, corrected):
    result = [dict(next(row for row in original if row['variant'] == 'L0'))]
    result[0].update(result_version='L0_AT_MOST_K_ACCEPTED', scientific_state='PASS')
    for variant, name in [('L1', 'CHEMLLM_7B_OFF_THE_SHELF'),
                          ('L2', 'CHEMLLM_7B_PPO_LORA_MAIN'),
                          ('L3', 'CHEMLLM_2B_OFF_THE_SHELF')]:
        source = corrected[name]
        if source['audit']['state'] != 'PASS' or source['audit']['test_selection']:
            raise ValueError('Require independently accepted, non-test-selected correction')
        k10 = source['table2']
        if len(k10) != 1 or int(k10[0]['k']) != 10:
            raise ValueError('Ambiguous Table2 scope')
        k10 = k10[0]
        # Auxiliary heldout fields may be K20. K10 costs come only from Table2.
        result.append(dict(variant=variant, model=name, eligible_rules=source['audit']['valid_unique_rule_count'],
            effective_k10=k10['effective_k'], cohort_size=source['metrics']['cohort_size'],
            **{'CCRCov@10': source['metrics']['CCRCov@10'], 'CCRCov@20': source['metrics']['CCRCov@20']},
            fixed_capped_mean_cost=k10['fixed_capped_mean_cost'],
            conditional_median_WNode=k10['conditional_median_cost'],
            strict_flip_reachable_k10=k10['strict_flip_parent_count'],
            attempts=source['candidate_metrics']['proposal_attempts'],
            source=source['root'], result_version='COMMON_PARSER_CORRECTION_e4073409',
            scientific_state='PASS', new_reach_v2_state='PENDING_NEW_VERSION_SPECIFIC_EVALUATION'))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--polylines-root', required=True, type=Path)
    p.add_argument('--old-llm-csv', required=True, type=Path)
    p.add_argument('--corrected-llm-json', required=True, type=Path)
    p.add_argument('--output-root', required=True, type=Path)
    args = p.parse_args()
    if not args.config.is_file() or args.output_root.exists():
        p.error('Existing config and fresh output root required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    output = args.output_root
    output.mkdir(parents=True)
    audit = json.loads((args.polylines_root / 'display_error_audit.json').read_text())
    methods = ['Ours', 'GlobalGCE', 'GCFExplainer', 'ComRecGC']
    colors = dict(zip(methods, ['#a12132', '#247b75', '#4666b0', '#8f662d']))
    datasets = ['AIDS', 'Mutagenicity', 'BACE']
    for kind in ['exact', 'display']:
        records = read(args.polylines_root / f'figure4_{kind}_polyline.csv')
        fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3))
        for ax, dataset in zip(axes, datasets):
            subset = [r for r in records if r['dataset'] == dataset and int(r['k']) == 10]
            for method in methods:
                rows = [r for r in subset if r['method'] == method]
                if not rows:
                    raise ValueError(f'Missing original {dataset}/{method}')
                ax.plot([float(r['threshold']) for r in rows], [float(r['coverage']) for r in rows],
                        color=colors[method], label=method, linewidth=1.55)
            maximum = max(float(r['coverage']) for r in subset)
            # Entire exact range, independent vertical axes, BACE 80% visible.
            ax.set_ylim(0, 1 if dataset == 'BACE' else min(1, max(.1, maximum * 1.08)))
            ax.set_xlim(0, max(float(r['threshold']) for r in subset))
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_title(dataset + (' (source cohort)' if dataset == 'AIDS' else ''))
            ax.set_xlabel('WNode distance threshold')
            ax.grid(alpha=.2)
            for item in audit['groups']:
                if item['dataset'] == dataset and item['method'] == 'Ours' and item['k'] == 10:
                    for theta in item['key_thresholds']:
                        ax.axvline(theta, color='#777777', linewidth=.6, linestyle=':')
        axes[0].set_ylabel('Coverage at fixed K=10')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(.5, .03), frameon=False)
        fig.suptitle('Frozen v1 source results — v2 repairs remain separate and pending', fontsize=12)
        note = ('Exact saved-distance ECDF; no smoothed numerical metrics.' if kind == 'exact' else
                'Display-only polyline, uniform error bound ≤0.5 percentage points; metrics use exact source.')
        fig.text(.5, .015, note + '\nAIDS ComRecGC / BACE GlobalGCE are under scientific repair, not validated by numerical consistency.',
                 ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .15, 1, .93))
        for suffix in ['pdf', 'png']:
            fig.savefig(output / f'figure4_v1_{kind}.{suffix}', dpi=180)
        plt.close(fig)
    llm = corrected_llm_rows(read(args.old_llm_csv), json.loads(args.corrected_llm_json.read_text()))
    fields = list(dict.fromkeys(key for row in llm for key in row))
    with (output / 'llm_v1_parser_corrected.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(llm)
    body = []
    for row in llm:
        body.append([row['variant'], row['eligible_rules'], row['effective_k10'], row['cohort_size'],
                     f"{100 * float(row['CCRCov@10']):.2f}", f"{100 * float(row['CCRCov@20']):.2f}",
                     f"{float(row['fixed_capped_mean_cost']):.6f}",
                     f"{float(row['conditional_median_WNode']):.6f}"])
    fig, ax = plt.subplots(figsize=(12.8, 3.5))
    ax.axis('off')
    ax.set_title('BACE LLM proposer v1: accepted common-parser correction', fontweight='bold', pad=20)
    table = ax.table(cellText=body, colLabels=['Variant', 'Eligible rules', 'K10 effective', 'N',
        'Cov@10 %', 'Cov@20 %', 'Capped mean\n(K10)', 'Conditional median\n(K10)'], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for (r, _), cell in table.get_celld().items():
        cell.set_linewidth(.35)
        if r == 0:
            cell.set_height(cell.get_height() * 1.5)
            cell.set_facecolor('#e7ebef')
            cell.set_text_props(fontweight='bold')
    fig.text(.5, .03, 'L0=BRICS; L1=7B off-the-shelf; L2=7B PPO-LoRA; L3=2B off-the-shelf. 3088 fixed attempts each.\n'
             'At-most-K, no padding. L0 remains unchanged. Corrected L3 decline is retained.\n'
             'These are v1 proposer results, not the new Reach-v2 ablation. GNN common N=96 is a different cohort.',
             ha='center', fontsize=9)
    fig.subplots_adjust(top=.82, bottom=.22, left=.02, right=.98)
    for suffix in ['pdf', 'png']:
        fig.savefig(output / f'llm_v1_parser_corrected.{suffix}', dpi=180)
    plt.close(fig)
    (output / 'version_scope_manifest.json').write_text(json.dumps({
        'state': 'DERIVED_EXISTING_RESULTS_ONLY', 'new_scientific_results_created': False,
        'source_exact_ecdf': audit['source_csv'], 'source_exact_ecdf_sha256': audit['source_csv_sha256'],
        'corrected_llm_summary': str(args.corrected_llm_json),
        'v2_scope': {'Ours-Reach-v2': 'PENDING', 'GlobalGCE-ChemAligned': 'PENDING', 'ComRecGC-RFAligned': 'PENDING'},
        'main_matrix_written': False, 'test_used_for_development': False,
        'metrics_computed_from_display': False}, indent=2) + '\n')
    print(json.dumps({'state': 'EXISTING_RESULTS_RENDERED', 'output': str(output)}))


if __name__ == '__main__':
    main()
