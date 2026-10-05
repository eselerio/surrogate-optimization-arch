"""Recompute the values used to describe the retained article figures."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.generate_composite_two_route_charts import (
    COMPOSITES,
    COMPOSITE_MATRIX,
    LOCATION_LABELS,
    NOMINAL_INFLUENT,
    TSS_VECTOR,
    coordinate_normalized_score,
    finite_score,
    response_composites,
)


def main():
    run = Path('results/article_v3/article_full_10000_001')
    cases = ['nominal', *[f'robustness_{i:02d}' for i in range(1, 11)]]
    labels = ['N', *[f'S{i}' for i in range(1, 11)]]
    with np.load(run / 'datasets/effective_design.npz') as data:
        test_decisions = data['test_decisions']
        development_decisions = data['development_decisions']
        influents = np.vstack([NOMINAL_INFLUENT, data['robustness_influents']])
    with np.load(run / 'predictions/post_selection_holdout.npz') as data:
        holdout = {key: response_composites(data[key], test_decisions)
                   for key in ('mechanistic', 'raw', 'projected')}
    truth = holdout['mechanistic']
    metrics = {}
    cells = {}
    for method in ('raw', 'projected'):
        metrics[method] = dict(zip(
            ('nrmse', 'nmae', 'mean_location_r2'),
            coordinate_normalized_score(truth.reshape(len(truth), -1),
                                        holdout[method].reshape(len(truth), -1)),
        ))
        metrics[method]['locations'] = {
            location: coordinate_normalized_score(truth[:, i, :], holdout[method][:, i, :])
            for i, location in enumerate(LOCATION_LABELS)
        }
        metrics[method]['composites'] = {
            composite: coordinate_normalized_score(truth[:, :, i], holdout[method][:, :, i])
            for i, composite in enumerate(COMPOSITES)
        }
        cells[method] = np.array([
            [finite_score(truth[:, i, j], holdout[method][:, i, j])[0]
             for j in range(4)] for i in range(8)
        ])
    delta = 100 * (cells['projected'] / cells['raw'] - 1)
    holdout_summary = {
        'n': len(truth), 'metrics': metrics,
        'nrmse_relative_reduction_percent': 100 * (1 - metrics['projected']['nrmse'] / metrics['raw']['nrmse']),
        'nmae_relative_reduction_percent': 100 * (1 - metrics['projected']['nmae'] / metrics['raw']['nmae']),
        'cell_percent_changes': {loc: dict(zip(COMPOSITES, delta[i].tolist()))
                                 for i, loc in enumerate(LOCATION_LABELS)},
        'cells_improved': int((delta < 0).sum()),
        'overflow_tss_underpredicted_states': int((holdout['projected'][:, 6, 3] < truth[:, 6, 3]).sum()),
        'overflow_tss_projected_nrmse': finite_score(truth[:, 6, 3], holdout['projected'][:, 6, 3])[0],
        'overflow_tp_raw_nrmse': finite_score(truth[:, 6, 2], holdout['raw'][:, 6, 2])[0],
        'overflow_tp_projected_nrmse': finite_score(truth[:, 6, 2], holdout['projected'][:, 6, 2])[0],
    }
    with np.load(run / 'datasets/development/mechanistic_accepted_v3.npz') as data:
        dev_quality = (data['targets'][:, 120:140] /
                       (1 - development_decisions[:, 6])[:, None]) @ COMPOSITE_MATRIX.T
    quality_scale = dev_quality.std(axis=0)
    selected = {}
    reference = {}
    profiles = {}
    weights = np.array([0.5, 0.15, 0.20, 0.05, 0.05, 0.05])
    for case, label, feed in zip(cases, labels, influents):
        feed_quality = COMPOSITE_MATRIX @ feed
        reference[label] = {}
        profiles[label] = {}
        for route in ('surrogate', 'direct'):
            path = run / 'optimization' / case / f'{route}_casewise_reference.npz'
            if not path.is_file() and case == 'robustness_01' and route == 'direct':
                path = run / 'report/chart_overrides/robustness_01_direct_casewise_reference.npz'
            with np.load(path) as data:
                theta = data['theta']
                response = data['exact_reference']
                effluent = COMPOSITE_MATRIX @ (response[120:140] / (1 - theta[6]))
                underflow_tss = float(TSS_VECTOR @ (response[140:160] / (theta[5] + theta[6])))
                parts = np.array([
                    np.mean(effluent / quality_scale), (theta[0] - 6) / 30,
                    theta[0] * sum(theta[1:4]) / 108, theta[4] / 4,
                    theta[5] - 0.25, theta[6] * underflow_tss / 750,
                ])
                profile = np.vstack([feed, response[:120].reshape(6, 20),
                                     response[120:140] / (1 - theta[6])]) @ COMPOSITE_MATRIX.T
                profiles[label][route] = profile.tolist()
                reference[label][route] = {
                    'controls': dict(zip(('H', 'a3', 'a4', 'a5', 'rI', 'rR', 'w'), theta.tolist())),
                    'effluent': dict(zip(COMPOSITES, effluent.tolist())),
                    'quality': float(parts[0]), 'economic': float(weights[1:] @ parts[1:]),
                    'objective': float(weights @ parts),
                    'H_per_pass': float(theta[0] / (1 + theta[4] + theta[5])),
                }
                if route == 'surrogate':
                    predicted = COMPOSITE_MATRIX @ (data['projected'][120:140] / (1 - theta[6]))
                    selected[label] = {
                        'predicted': dict(zip(COMPOSITES, predicted.tolist())),
                        'reference': dict(zip(COMPOSITES, effluent.tolist())),
                        'predicted_removal': dict(zip(COMPOSITES, (100 * (1 - predicted / feed_quality)).tolist())),
                        'reference_removal': dict(zip(COMPOSITES, (100 * (1 - effluent / feed_quality)).tolist())),
                        'removal_error_pp': dict(zip(COMPOSITES, (100 * (effluent - predicted) / feed_quality).tolist())),
                    }
    timing = pd.read_csv(run / 'metrics/robustness_case_timing.csv').pivot(
        index='case', columns='route', values='time_seconds').reindex(cases[1:])
    nominal = pd.read_csv(run / 'report/tables/selected_candidate_reference_evaluation.csv').query(
        "case == 'nominal'").pivot(index='case', columns='route', values='time_seconds')
    time_summary = {
        'nominal': nominal.loc['nominal'].to_dict(),
        'ranges': {route: [float(timing[route].min()), float(timing[route].max())]
                   for route in ('surrogate', 'direct')},
        'means': timing.mean().to_dict(),
        'ratios_direct_over_surrogate': dict(zip(labels[1:], (timing.direct / timing.surrogate).tolist())),
    }
    comparisons = {
        'direct_lower_counts': {metric: sum(reference[label]['direct'][metric] < reference[label]['surrogate'][metric]
                                            for label in labels)
                                for metric in ('objective', 'quality', 'economic')},
        'direct_lower_composite_counts': {comp: sum(reference[label]['direct']['effluent'][comp] < reference[label]['surrogate']['effluent'][comp]
                                                  for label in labels) for comp in COMPOSITES},
        'largest_objective_difference_case': max(labels, key=lambda label: reference[label]['surrogate']['objective'] - reference[label]['direct']['objective']),
    }
    audit = {'holdout': holdout_summary, 'selected_parity': selected,
             'routes': reference, 'profiles': profiles, 'time': time_summary,
             'quality_scales': quality_scale.tolist(), 'comparisons': comparisons}
    audit_path = run / 'report/manuscript_numerical_audit.json'
    audit_path.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps({key: audit[key] for key in ('holdout', 'selected_parity', 'routes', 'time', 'quality_scales', 'comparisons')}, indent=2))
    print(f'Audit saved to {audit_path}')


if __name__ == '__main__':
    main()
