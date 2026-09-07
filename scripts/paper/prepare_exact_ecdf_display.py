#!/usr/bin/env python3
"""Prepare exact and display polylines from already exported full-range ECDF CSV."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.ecdf_display import staircase, simplify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--source-csv', required=True, type=Path)
    p.add_argument('--key-thresholds-json', required=True, type=Path,
                   help='dataset -> exact frozen theta list; not fitted here')
    p.add_argument('--output-root', required=True, type=Path)
    args = p.parse_args()
    if not args.config.is_file():
        p.error('Config path missing')
    if args.output_root.exists():
        p.error('Fresh output root required; original exports are immutable')
    keys = json.loads(args.key_thresholds_json.read_text())
    groups = defaultdict(list)
    for row in csv.DictReader(args.source_csv.open(newline='')):
        if not row.get('curve_kind', '').startswith('EXACT_ECDF'):
            raise ValueError('Require exact saved-distance ECDF, not sparse grid or display data')
        groups[(row['dataset'], row['method'], int(row['k']))].append(row)
    products = {'exact': [], 'display': []}
    audits = []
    for (dataset, method, k), rows in sorted(groups.items()):
        group_keys = keys[dataset]
        points = staircase(((float(r['threshold']), float(r['coverage'])) for r in rows), group_keys)
        display, audit = simplify(points, keys=group_keys)
        audit.update(dataset=dataset, method=method, k=k)
        audits.append(audit)
        for kind, values in [('exact', points), ('display', display)]:
            products[kind].extend({'dataset': dataset, 'method': method, 'k': k,
                'threshold': point.x, 'coverage': point.y,
                'curve_kind': 'EXACT_ECDF_POLYLINE' if kind == 'exact' else 'DISPLAY_ONLY_BOUNDED_POLYLINE',
                'metrics_allowed': kind == 'exact'} for point in values)
    args.output_root.mkdir(parents=True)
    for kind, rows in products.items():
        with (args.output_root / f'figure4_{kind}_polyline.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = {'schema_version': 'paper_reach_v2_display_audit_v1', 'groups': audits,
        'source_csv': str(args.source_csv.resolve()),
        'source_csv_sha256': hashlib.sha256(args.source_csv.read_bytes()).hexdigest(),
        'input_key_thresholds': str(args.key_thresholds_json.resolve()),
        'numerical_source': 'ORIGINAL_EXACT_CSV_ONLY', 'model_or_candidate_recomputed': False}
    (args.output_root / 'display_error_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'state': 'DERIVED_DISPLAY_COMPLETE', 'groups': len(audits), 'output': str(args.output_root)}))


if __name__ == '__main__':
    main()
