#!/usr/bin/env python3
"""Independent arithmetic, split, units, bootstrap and preservation checks."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/NMH_registered_extension'
ARC = OUT / 'archive'
VL = ARC / 'validation_lock'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def independent_metrics(y, p, lp):
    error = y - p
    slope = np.dot(p - p.mean(), y - y.mean()) / np.dot(p - p.mean(), p - p.mean())
    return {'predictive_r2': 1 - np.dot(error, error) / np.dot(y - y.mean(), y - y.mean()),
            'rmse': np.sqrt(np.dot(error, error) / len(error)), 'mae': np.abs(error).sum() / len(error),
            'mean_log_predictive_density': lp.sum() / len(lp),
            'calibration_intercept': y.mean() - slope * p.mean(), 'calibration_slope': slope}


def validate():
    note = json.loads((ARC / 'pre_estimation_implementation_hash.json').read_text())
    start = json.loads((ARC / 'estimation_started.json').read_text())
    assert digest(ROOT / note['note_path']) == note['note_sha256'] == start['note_sha256']
    assert datetime.fromisoformat(note['sealed_at_utc']) < datetime.fromisoformat(start['first_fit_start_utc'])
    record = json.loads((ARC / 'registration_input_hashes.json').read_text())
    for filename, original in {**record['inputs'], **record['protected_parent_results']}.items():
        assert digest(ROOT / filename) == original, filename
    for lock in (ARC / 'sample_lock', VL):
        for f in json.loads((lock / 'manifest.json').read_text())['files']:
            assert digest(ROOT / f['path']) == f['sha256']
    targets = pd.read_parquet(ARC / 'sample_lock/target_registry.parquet')
    history = pd.read_parquet(ARC / 'sample_lock/history_registry.parquet')
    counts = pd.read_parquet(ARC / 'sample_lock/sample_counts.parquet')
    pd.testing.assert_frame_equal(counts, pd.read_csv(OUT / '01_sample_counts.csv'), check_dtype=False)
    members = pd.read_parquet(VL / 'case_memberships.parquet')
    cases = pd.read_parquet(VL / 'case_registry.parquet')
    runs = pd.read_parquet(VL / 'run_registry.parquet')
    result = pd.read_parquet(ARC / 'all_performance_results.parquet')
    fit = pd.read_parquet(ARC / 'all_fit_records.parquet')
    source = pd.concat([pd.read_parquet(ROOT / f'data_processed/NMH_{c}_event_episodes.parquet',
                       columns=['cohort', 'episode_id', 'person_id', 'outcome_pre1_raw', 'outcome_event_raw', 'outcome_year2_raw'])
                       for c in ('UKHLS', 'HRS')], ignore_index=True)
    lookup = source.set_index(['cohort', 'episode_id']).to_dict('index')
    assert len(runs) == 192 and len(cases) == 48 and len(result) == 14688 and len(fit) == 8448
    assert not members.duplicated(['case_id', 'person_id']).any()
    assert not history.duplicated(['sample_id', 'person_id', 'history_episode_id']).any()
    assert (history.history_qualification_date < history.information_cutoff_date).all()
    assert (history.history_event_date < history.information_cutoff_date).all()
    count_qa = pd.read_parquet(ARC / 'event_count_information_cutoff_QA.parquet')
    assert count_qa.n_observed_at_pre1.le(count_qa.source_preceding_index_count).all()
    for case in cases.itertuples(index=False):
        m = members.loc[members.case_id.eq(case.case_id)]
        cutoff = m.event_date.sort_values().iloc[math.ceil(0.70 * len(m)) - 1]
        assert cutoff == case.temporal_cutoff
        assert m.loc[m.temporal_assignment.eq('train'), 'event_date'].le(cutoff).all()
        assert m.loc[m.temporal_assignment.eq('test'), 'event_date'].gt(cutoff).all()
        assert m.temporal_assignment.eq('train').sum() == case.temporal_train_n
        assert m.temporal_assignment.eq('test').sum() == case.temporal_test_n
        assert case.temporal_person_overlap_excluded == 0
        assert set(m.outer_fold) == set(range(10))
    predictions_cache = {}
    max_metric_error = 0.0
    predictions_checked = 0
    boot_rows = 0
    for run in runs.to_dict('records'):
        run_id = run['run_id']
        m = members.loc[members.case_id.eq(run['case_id'])]
        fr = fit.loc[fit.run_id.eq(run_id)]
        model_names = sorted(fr.model.unique())
        for split, group in fr.groupby('split'):
            if split == 'temporal':
                training = m.loc[m.temporal_assignment.eq('train')]
                testing = m.loc[m.temporal_assignment.eq('test')]
            else:
                fold = int(split.split('_')[-1])
                training, testing = m.loc[m.outer_fold.ne(fold)], m.loc[m.outer_fold.eq(fold)]
            x = np.array([lookup[(run['cohort'], eid)]['outcome_pre1_raw'] for eid in training.target_episode_id])
            mu, sd = (x.mean(), x.std(ddof=1)) if run['scale'] == 'standardized' else (0, 1)
            assert np.allclose(group.training_pre1_mean, mu, atol=1e-13, rtol=1e-13)
            assert np.allclose(group.training_pre1_sd_ddof1, sd, atol=1e-13, rtol=1e-13)
            assert group.n_train.eq(len(training)).all() and group.n_test.eq(len(testing)).all()
            assert len(group.training_person_ids_sha256.unique()) == 1
            assert len(group.test_person_ids_sha256.unique()) == 1
            assert np.allclose(group.sigma2_train, group.rss_train / group.n_train, atol=1e-13, rtol=1e-13)
            assert not set(training.person_id) & set(testing.person_id)
        for validation in ('grouped_cv', 'temporal'):
            p = pd.read_parquet(ARC / 'predictions' / f'{run_id}_{validation}.parquet').sort_values('person_id').reset_index(drop=True)
            predictions_cache[(run_id, validation)] = p
            expected = m if validation == 'grouped_cv' else m.loc[m.temporal_assignment.eq('test')]
            assert p.person_id.is_unique and set(p.person_id) == set(expected.person_id)
            predictions_checked += len(p)
            y = p.y_observed.to_numpy()
            endpoint = 'outcome_event_raw' if run['outcome'] == 'acute' else 'outcome_year2_raw'
            raw_y = np.array([lookup[(run['cohort'], eid)][endpoint] for eid in p.target_episode_id])
            pre1 = np.array([lookup[(run['cohort'], eid)]['outcome_pre1_raw'] for eid in p.target_episode_id])
            if run['formulation'] == 'change_score':
                raw_y = raw_y - pre1
                expected_y = raw_y / p.training_pre1_sd_ddof1.to_numpy()
            else:
                expected_y = (raw_y - p.training_pre1_mean.to_numpy()) / p.training_pre1_sd_ddof1.to_numpy()
            assert np.allclose(p.y_raw, raw_y)
            assert np.allclose(y, expected_y, atol=1e-12)
            values = {}
            for model in model_names:
                point = p['prediction_' + model].to_numpy()
                sigma = p['sigma2_' + model].to_numpy()
                lp = -0.5 * (np.log(2 * np.pi * sigma) + (y - point) ** 2 / sigma)
                assert np.allclose(p['log_density_' + model], lp, atol=1e-11, rtol=1e-11)
                values[model] = independent_metrics(y, point, lp)
            r = result.loc[result.run_id.eq(run_id) & result.validation.eq(validation)]
            bootstrap = pd.read_parquet(ARC / 'bootstrap' / f'{run_id}_{validation}.parquet')
            boot_rows += len(bootstrap)
            assert len(bootstrap) == 500 and bootstrap.bootstrap_id.nunique() == 500
            for row in r.itertuples(index=False):
                if row.row_type == 'increment':
                    a, b = row.term.split('-')
                    estimate = values[a][row.metric] - values[b][row.metric]
                    assert np.allclose(bootstrap[row.term + '|' + row.metric],
                                       bootstrap[a + '|' + row.metric] - bootstrap[b + '|' + row.metric], atol=1e-12)
                else:
                    estimate = values[row.term][row.metric]
                max_metric_error = max(max_metric_error, float(abs(estimate - row.estimate)))
                assert np.isclose(row.estimate, estimate, atol=1e-10, rtol=1e-10)
                bounds = np.quantile(bootstrap[row.term + '|' + row.metric], [0.025, 0.975])
                assert np.allclose([row.ci_lower, row.ci_upper], bounds, atol=1e-11, rtol=1e-11)
                assert row.bootstrap_successful == 500 and row.n_evaluated == len(p)
    scale_checks = ancova_checks = 0
    for (case_id, formulation), group in runs.groupby(['case_id', 'formulation']):
        ids = group.set_index('scale').run_id.to_dict()
        for val in ('grouped_cv', 'temporal'):
            std, raw = predictions_cache[(ids['standardized'], val)], predictions_cache[(ids['raw'], val)]
            assert std.person_id.equals(raw.person_id)
            for col in [c for c in std.columns if c.startswith('prediction_raw_')]:
                assert np.allclose(std[col], raw[col], atol=1e-8, rtol=1e-8, equal_nan=True), (case_id, col)
            scale_checks += 1
    for (case_id, scale), group in runs.groupby(['case_id', 'scale']):
        ids = group.set_index('formulation').run_id.to_dict()
        for val in ('grouped_cv', 'temporal'):
            change, ancova = predictions_cache[(ids['change_score'], val)], predictions_cache[(ids['ancova_level'], val)]
            assert change.person_id.equals(ancova.person_id)
            prefix = 'C' if '__recency__' in case_id else ('R' if case_id.startswith('recurrence') else 'M')
            assert np.allclose(ancova['prediction_raw_' + prefix + '3'],
                               change['prediction_raw_' + prefix + '3'] + change.target_pre1_raw, atol=1e-8, rtol=1e-8)
            if prefix != 'C':
                assert np.allclose(ancova['prediction_' + prefix + '0'], ancova['prediction_' + prefix + '2'], atol=1e-10)
                assert np.allclose(ancova['prediction_' + prefix + '1'], ancova['prediction_' + prefix + '3'], atol=1e-10)
            else:
                assert np.allclose(ancova.prediction_C0, ancova.prediction_C2, atol=1e-10)
            ancova_checks += 1
    result_payload = {'passed': True, 'implementation_note_sealed_before_estimation': True,
        'protocol_note_and_source_hashes_unchanged': True, 'sample_and_validation_locks_unchanged': True,
        'case_count': len(cases), 'run_count': len(runs), 'fit_count': len(fit),
        'performance_rows_independently_recomputed': len(result),
        'prediction_rows_checked': predictions_checked, 'bootstrap_summary_rows_checked': boot_rows,
        'training_pre1_only_standardization_verified': True, 'ddof_one_verified': True,
        'observed_date_cutoff_and_ties_verified': True, 'identical_nested_model_test_persons_verified': True,
        'training_RSS_divided_by_n_and_Gaussian_log_density_verified': True,
        'all_500_paired_bootstrap_replicates_and_intervals_verified': True,
        'raw_standardized_prediction_equivalence_checks': scale_checks,
        'change_ANCOVA_prediction_identity_checks': ancova_checks,
        'maximum_metric_recomputation_difference': max_metric_error}
    (ARC / 'independent_analysis_validation.json').write_text(json.dumps(result_payload, indent=2) + '\n')
    print(json.dumps(result_payload, indent=2))


if __name__ == '__main__':
    validate()
