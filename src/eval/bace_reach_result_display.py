"""Independent numerical display checks, not an oracle/scientific acceptance."""
import math
import statistics


def checked_control_rows(report, *, parent_count, theta, cap):
    if report['cohort_size'] != parent_count or len(report['prefix_rows']) != 20:
        raise ValueError('DISPLAY_COHORT_OR_PREFIX_CONFLICT')
    groups = {k: [] for k in range(1, 21)}
    for row in report['parent_rows']:
        groups[int(row['k'])].append(row)
    previous_ids = None
    previous = {}
    result = []
    for k in range(1, 21):
        rows = groups[k]
        ids = [r['parent_id'] for r in rows]
        if len(rows) != parent_count or len(set(ids)) != parent_count:
            raise ValueError('DISPLAY_PARENT_PARTITION_CONFLICT')
        if previous_ids is not None and set(ids) != previous_ids:
            raise ValueError('DISPLAY_PREFIX_COHORT_DRIFT')
        values = []
        for row in rows:
            value = math.inf if row['best_distance'] is None else float(row['best_distance'])
            if math.isnan(value) or value < 0 or value > previous.get(row['parent_id'], math.inf):
                raise ValueError('DISPLAY_INVALID_OR_NONNESTED_DISTANCE')
            if row['strict_recourse_available'] != math.isfinite(value) or row['theta_star_covered'] != (value <= theta):
                raise ValueError('DISPLAY_MASK_DISTANCE_CONFLICT')
            if not math.isclose(row['capped_distance'], min(value, cap), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('DISPLAY_CAPPED_DISTANCE_CONFLICT')
            previous[row['parent_id']] = value
            values.append(value)
        finite = [x for x in values if math.isfinite(x)]
        expected = dict(k=k, N=parent_count, finite_reach=len(finite),
            covered_theta=sum(x <= theta for x in values), covered_cap=sum(x <= cap for x in values),
            conditional_median=statistics.median(finite) if finite else None,
            capped_mean=sum(min(x, cap) for x in values)/parent_count)
        original = report['prefix_rows'][k-1]
        if (original['k'] != k or original['strict_flip_parent_count'] != len(finite)
                or original['num_theta_star_covered'] != expected['covered_theta']
                or not math.isclose(original['ccrcov_theta_star'], expected['covered_theta']/parent_count, abs_tol=1e-12)
                or not math.isclose(original['fixed_capped_mean_cost'], expected['capped_mean'], abs_tol=1e-12)
                or original['conditional_median_cost'] != expected['conditional_median']):
            raise ValueError('DISPLAY_NUMERICAL_SOURCE_CONFLICT')
        expected['ecdf'] = [(x, sum(v <= x for v in values)/parent_count)
                            for x in sorted({0., theta, cap, *finite})]
        result.append(expected)
        previous_ids = set(ids)
    return result
