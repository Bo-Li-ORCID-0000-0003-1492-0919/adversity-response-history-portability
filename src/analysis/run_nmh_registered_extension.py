#!/usr/bin/env python3
"""OSF zgeb3/v2 registered extension; prepare immutable cases, then fit.

Only v2 and the user-approved pre-estimation note control this implementation.
No parent analysis module, v1 protocol, or old DRAFT is imported or read.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import platform
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/NMH_registered_extension'
ARC = OUT / 'archive'
LOCK = ARC / 'sample_lock'
VLOCK = ARC / 'validation_lock'
WORK = ROOT / 'outputs/nmh_registered_extension_20260904'
PROTOCOL = ROOT / 'OSF_registration_bundle_v2/UPLOAD_TO_OSF/preregistration.pdf'
PROTOCOL_SHA = '8077d0105f3867533ed57ab4f1075f51712ff3cad3e23a8556c53c0d27f7b8c0'
NOTE_SHA = 'b8796b26831285a0e5971b63815c9888c3d82fe45380540aab8cd1cb942d6fa2'
SEED = 20260904
BOOTSTRAPS = 500
METRICS = ('predictive_r2', 'rmse', 'mae', 'mean_log_predictive_density', 'calibration_intercept', 'calibration_slope')
FAMILIES = ('caregiving', 'financial_strain', 'health', 'unemployment', 'widowhood')
EDUCATION = ('degree', 'no_qualification', 'other_qualification', 'unknown')


def log(text):
    print(text, flush=True)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def seal_json(path, payload):
    assert not path.exists(), f'Write-once archive exists: {path}'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    path.chmod(0o444)


def seal_parquet(frame, path, key):
    assert not frame.duplicated(key).any(), str(path)
    assert not path.exists(), f'Write-once registry exists: {path}'
    frame.to_parquet(path, index=False)
    pd.testing.assert_frame_equal(frame, pd.read_parquet(path))
    path.chmod(0o444)
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'rows': len(frame), 'key': key}


def verify_sources(full=False):
    assert sha(PROTOCOL) == PROTOCOL_SHA
    assert sha(OUT / '00_pre_estimation_implementation_note.md') == NOTE_SHA
    approved = json.loads((ARC / 'pre_estimation_implementation_hash.json').read_text())
    assert approved['note_sha256'] == NOTE_SHA and approved['new_models_fitted_at_sealing'] == 0
    for folder in (LOCK, VLOCK):
        manifest = folder / 'manifest.json'
        if manifest.exists():
            for row in json.loads(manifest.read_text())['files']:
                assert sha(ROOT / row['path']) == row['sha256'], row['path']
    if full:
        registry = json.loads((ARC / 'registration_input_hashes.json').read_text())
        for name, digest in {**registry['inputs'], **registry['protected_parent_results']}.items():
            assert sha(ROOT / name) == digest, name


def load_sources(include_scores=False):
    columns = ['cohort', 'person_id', 'episode_id', 'age_at_event', 'sex', 'education', 'event_year',
               'preceding_primary_episode_count', 'event_date', 'pre1_date', 'event_wave', 'pre1_wave']
    if include_scores:
        columns += ['outcome_pre2_raw', 'outcome_pre1_raw', 'outcome_event_raw', 'outcome_year2_raw',
                    'event_self_report', 'pre1_self_report', 'year2_censored_by_later_primary']
    chunks = [pd.read_parquet(ROOT / f'data_processed/NMH_{cohort}_event_episodes.parquet', columns=columns)
              for cohort in ('UKHLS', 'HRS')]
    frame = pd.concat(chunks, ignore_index=True)
    assert not frame.duplicated(['cohort', 'episode_id']).any()
    if include_scores:
        for point in ('pre2', 'pre1', 'event', 'year2'):
            col = 'outcome_' + point + '_raw'
            legal = frame[col].between(0, 36) & ((frame.cohort == 'UKHLS') | frame[col].le(8))
            frame.loc[~legal, col] = np.nan
        frame.loc[~frame.pre1_self_report, 'outcome_pre1_raw'] = np.nan
        frame.loc[~frame.event_self_report, 'outcome_event_raw'] = np.nan
        frame.loc[frame.year2_censored_by_later_primary, 'outcome_year2_raw'] = np.nan
    return frame


def prepare():
    verify_sources(full=True)
    if (VLOCK / 'manifest.json').exists():
        log('Existing complete-case and validation lock verified; not replaced.')
        return
    VLOCK.mkdir(parents=True, exist_ok=True)
    targets = pd.read_parquet(LOCK / 'target_registry.parquet')
    histories = pd.read_parquet(LOCK / 'history_registry.parquet')
    folds = pd.read_parquet(LOCK / 'grouped_cv_assignments.parquet')
    # Only covariates, not any psychological scores, are read before case lock.
    sources = load_sources(False)
    cov = sources.rename(columns={'episode_id': 'target_episode_id'})
    tcov = targets[['sample_id', 'cohort', 'person_id', 'target_episode_id']].merge(
        cov, on=['cohort', 'person_id', 'target_episode_id'], how='left', validate='many_to_one')
    assert tcov.age_at_event.notna().all() and tcov.sex.isin(['male', 'female']).all()
    assert tcov.education.isin(EDUCATION).all()
    hcov = histories.merge(sources.rename(columns={'episode_id': 'history_episode_id'}),
        on=['cohort', 'person_id', 'history_episode_id'], how='left', validate='many_to_one')
    prior_cov_ok = (hcov.age_at_event.between(16, 110) & hcov.event_year.notna()
                    & hcov.preceding_primary_episode_count.notna())
    hcov['residualization_covariates_complete'] = prior_cov_ok
    all_history_ok = hcov.groupby(['sample_id', 'person_id']).residualization_covariates_complete.all()
    cases, members, runs = [], [], []
    for sample_id, sample in targets.groupby('sample_id', sort=True):
        base = sample.iloc[0]
        reps = ['same_response'] if base.analysis == 'recurrence' else ['mean', 'most_recent', 'residualized_mean', 'between_history_sd']
        combinations = [('main', rep) for rep in reps] + [('recency', reps[0])]
        for benchmark, representation in combinations:
            chosen = sample.copy()
            if representation == 'residualized_mean':
                chosen = chosen.loc[[all_history_ok.loc[(sample_id, pid)] for pid in chosen.person_id]]
            if benchmark == 'recency':
                chosen = chosen.loc[chosen.pre2_available & chosen.pre1_available &
                                    chosen.pre2_date.lt(chosen.pre1_date) & chosen.pre2_wave.lt(chosen.pre1_wave)]
            case_id = f'{sample_id}__{benchmark}__{representation}'
            chosen = chosen.sort_values(['event_date', 'person_id'])
            n = len(chosen)
            cutoff = chosen.event_date.iloc[math.ceil(0.70 * n) - 1] if n else pd.NaT
            train = set(chosen.loc[chosen.event_date.le(cutoff), 'person_id'])
            proposed_test = set(chosen.loc[chosen.event_date.gt(cutoff), 'person_id'])
            overlap = proposed_test & train
            test = proposed_test - train
            assert not train & test and len(train) + len(test) + len(overlap) == n
            role = 'primary' if base.role == 'primary' and benchmark == 'main' and representation in ('same_response', 'mean') else 'secondary'
            case = {'case_id': case_id, 'sample_id': sample_id, 'cohort': base.cohort, 'analysis': base.analysis,
                    'event_family': base.family if base.analysis == 'recurrence' else 'multiple', 'outcome': base.outcome,
                    'variant': base.variant, 'benchmark': benchmark, 'representation': representation,
                    'base_role': role, 'n_parent': len(sample), 'n_complete': n, 'missing_excluded': len(sample) - n,
                    'temporal_cutoff': cutoff, 'temporal_train_n': len(train), 'temporal_test_n': len(test),
                    'temporal_person_overlap_excluded': len(overlap)}
            cases.append(case)
            fmap = folds.loc[folds.sample_id.eq(sample_id)].set_index('person_id').outer_fold.to_dict()
            for r in chosen.itertuples(index=False):
                members.append({'case_id': case_id, 'sample_id': sample_id, 'cohort': r.cohort,
                    'person_id': r.person_id, 'target_episode_id': r.target_episode_id, 'event_date': r.event_date,
                    'outer_fold': int(fmap[r.person_id]), 'temporal_assignment': 'train' if r.person_id in train else 'test',
                    'temporal_cutoff': cutoff})
            for formulation in ('change_score', 'ancova_level'):
                for scale in ('standardized', 'raw'):
                    name = f'{case_id}__{formulation}__{scale}'
                    run_id = hashlib.sha256(name.encode()).hexdigest()[:16]
                    runs.append({**case, 'run_id': run_id, 'formulation': formulation, 'scale': scale,
                                 'analysis_role': role if formulation == 'change_score' and scale == 'standardized' else 'secondary'})
    case_frame, member_frame, run_frame = pd.DataFrame(cases), pd.DataFrame(members), pd.DataFrame(runs)
    assert len(case_frame) == 48 and len(run_frame) == 192
    files = [seal_parquet(case_frame, VLOCK / 'case_registry.parquet', ['case_id']),
             seal_parquet(member_frame, VLOCK / 'case_memberships.parquet', ['case_id', 'person_id']),
             seal_parquet(run_frame, VLOCK / 'run_registry.parquet', ['run_id'])]
    seal_json(VLOCK / 'manifest.json', {'locked_at_utc': utc(), 'protocol_sha256': PROTOCOL_SHA,
        'implementation_note_sha256': NOTE_SHA, 'engine_sha256_before_estimation': sha(Path(__file__)),
        'psychological_scores_read_before_case_lock': False, 'files': files,
        'case_rules': 'Fixed pre2 subset; all-history covariate completeness for residualized representation; no histories dropped.',
        'cutoff_rule': 'Observed date at 1-based ceil(0.70*N); all ties train; participant overlap excluded.',
        'cv_rules': 'Original locked person folds inherited by complete subsets, representations and nested comparisons.',
        'no_old_DRAFT_fallbacks_imported': True})
    case_frame.to_csv(WORK / 'validation_case_counts_intermediate.csv', index=False)
    log(case_frame[['cohort', 'analysis', 'benchmark', 'representation', 'outcome', 'n_complete',
                    'temporal_cutoff', 'temporal_train_n', 'temporal_test_n']].to_string(index=False))
    log('48 cases and 192 registered model-set runs locked, without reading psychological scores.')


def get_data():
    source = load_sources(True)
    targets = pd.read_parquet(LOCK / 'target_registry.parquet')
    histories = pd.read_parquet(LOCK / 'history_registry.parquet')
    source_lookup = {(r.cohort, r.episode_id): r._asdict() for r in source.itertuples(index=False)}
    # Count only primary episodes already ascertainable at the current pre1.
    # In particular, a persistent-health onset is not yet known at its index
    # interview if the required next-wave confirmation has not been observed.
    knowledge_dates = {}
    for cohort in ('UKHLS', 'HRS'):
        masks = pd.read_parquet(ROOT / f'outputs/nmh_extension_20260904/{cohort}_episode_eligibility_masks.parquet',
                                columns=['person_id', 'n_primary_labels', 'event_date', 'qualification_date'])
        for pid, group in masks.loc[masks.n_primary_labels.gt(0)].groupby('person_id'):
            dates = group[['event_date', 'qualification_date']].max(axis=1, skipna=False).dropna().sort_values()
            knowledge_dates[(cohort, int(pid))] = dates.to_numpy('datetime64[ns]')
    def observed_count(cohort, pid, cutoff):
        dates = knowledge_dates.get((cohort, int(pid)), np.array([], dtype='datetime64[ns]'))
        return int(np.searchsorted(dates, np.datetime64(cutoff), side='right'))
    trows, hrows, count_checks = [], [], []
    for r in targets.to_dict('records'):
        s = source_lookup[(r['cohort'], r['target_episode_id'])]
        assert s['person_id'] == r['person_id']
        past_count = observed_count(r['cohort'], r['person_id'], r['pre1_date'])
        count_checks.append({'sample_id': r['sample_id'], 'person_id': r['person_id'], 'record_role': 'target',
            'episode_id': r['target_episode_id'], 'n_observed_at_pre1': past_count,
            'source_preceding_index_count': s['preceding_primary_episode_count']})
        trows.append({**r, 'age': s['age_at_event'], 'sex': s['sex'], 'education': s['education'],
            'year': s['event_year'], 'n_preceding_primary': past_count,
            **{p: s['outcome_' + p + '_raw'] for p in ('pre2', 'pre1', 'event', 'year2')}})
    for r in histories.to_dict('records'):
        s = source_lookup[(r['cohort'], r['history_episode_id'])]
        assert s['person_id'] == r['person_id']
        past_count = observed_count(r['cohort'], r['person_id'], s['pre1_date'])
        count_checks.append({'sample_id': r['sample_id'], 'person_id': r['person_id'], 'record_role': 'history',
            'episode_id': r['history_episode_id'], 'n_observed_at_pre1': past_count,
            'source_preceding_index_count': s['preceding_primary_episode_count']})
        hrows.append({**r, 'family': r['history_family'], 'age': s['age_at_event'], 'year': s['event_year'],
            'n_preceding_primary': past_count,
            **{p: s['outcome_' + p + '_raw'] for p in ('pre1', 'event', 'year2')}})
    pd.DataFrame(count_checks).to_parquet(ARC / 'event_count_information_cutoff_QA.parquet', index=False)
    return pd.DataFrame(trows), pd.DataFrame(hrows)


def encode_context(frame, multi=False, historical=False):
    # Category definitions come from the existing data dictionary, not test outcomes.
    matrix = {'age': frame.age.to_numpy(float), 'calendar_year': frame.year.to_numpy(float),
              'n_preceding_primary': frame.n_preceding_primary.to_numpy(float)}
    for category in FAMILIES[1:]:
        matrix['family_' + category] = frame.family.eq(category).to_numpy(float)
    if not historical:
        matrix['sex_female'] = frame.sex.eq('female').to_numpy(float)
        for category in EDUCATION[1:]:
            matrix['education_' + category] = frame.education.eq(category).to_numpy(float)
        if multi:
            matrix['n_qualifying_prior_episodes'] = frame.prior_episode_count.to_numpy(float)
    return pd.DataFrame(matrix, index=frame.index)


def ols(train_x, train_y, test_x):
    """OLS with training-only predictor conditioning; no regularization or tuning.

    Zero-variance columns stay zero. A test row outside the training design row
    space is flagged non-estimable rather than assigned an arbitrary category effect.
    """
    a, b, y = np.asarray(train_x, float), np.asarray(test_x, float), np.asarray(train_y, float)
    if len(y) == 0 or not np.isfinite(a).all() or not np.isfinite(y).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite_or_empty_training_design')
    center = a.mean(axis=0)
    spread = a.std(axis=0, ddof=0)
    spread[spread == 0] = 1.0
    a = np.column_stack([np.ones(len(a)), (a - center) / spread])
    b = np.column_stack([np.ones(len(b)), (b - center) / spread])
    coef, _, rank, singular = np.linalg.lstsq(a, y, rcond=None)
    fitted = a @ coef
    prediction = b @ coef
    rss = float(np.sum((y - fitted) ** 2))
    sigma2 = rss / len(y)
    # Mathematical estimability check, not a data-dependent model-selection rule.
    null_component = b - b @ (np.linalg.pinv(a) @ a)
    unsupported = np.linalg.norm(null_component, axis=1) > 1e-8 * (1 + np.linalg.norm(b, axis=1))
    prediction[unsupported] = np.nan
    return {'prediction': prediction, 'sigma2': sigma2, 'rss_train': rss, 'n_train': len(y),
            'rank': int(rank), 'n_design_columns': a.shape[1], 'unsupported_rows': int(unsupported.sum()),
            'coefficient': coef, 'conditioning_mean': center, 'conditioning_sd': spread}


def history_feature(d, h, train_ids, mu, sd, representation, outcome):
    h = h.copy()
    h['pre1_z'] = (h.pre1 - mu) / sd
    endpoint = h.event if outcome == 'acute' else h.year2
    h['response'] = (endpoint - mu) / sd - h.pre1_z
    assert np.isfinite(h.response).all()
    residual_meta = None
    if representation == 'residualized_mean':
        train_history = h.person_id.isin(train_ids)
        x = encode_context(h, historical=True)
        x['pre1_z'] = h.pre1_z.to_numpy()
        fit = ols(x.loc[train_history], h.loc[train_history, 'response'], x)
        h['response'] = h.response.to_numpy() - fit['prediction']
        residual_meta = {k: fit[k] for k in ('rss_train', 'n_train', 'rank', 'n_design_columns', 'unsupported_rows')}
    grouped = h.sort_values(['person_id', 'history_order']).groupby('person_id', sort=False)
    values = {}
    for pid, group in grouped:
        response = group.response.to_numpy(float)
        if not np.isfinite(response).all():
            values[pid] = np.nan
        elif representation == 'most_recent':
            values[pid] = float(response[-1])
        elif representation == 'between_history_sd':
            values[pid] = float(np.std(response, ddof=1)) if len(response) >= 2 else np.nan
        else:
            values[pid] = float(np.mean(response))
    return d.person_id.map(values).to_numpy(float), residual_meta


def specifications(run):
    if run['benchmark'] == 'recency':
        features = {'C0': ['pre2_z'], 'C1': ['pre1_z'], 'C2': ['pre2_z', 'pre1_z'],
                    'C3': ['pre2_z', 'pre1_z', 'history']}
        contrasts = [('C1', 'C0'), ('C2', 'C1'), ('C3', 'C2')]
    else:
        prefix = 'R' if run['analysis'] == 'recurrence' else 'M'
        features = {prefix + '0': [], prefix + '1': ['history'], prefix + '2': ['pre1_z'], prefix + '3': ['pre1_z', 'history']}
        contrasts = [(prefix + '1', prefix + '0'), (prefix + '3', prefix + '2')]
    if run['formulation'] == 'ancova_level':
        features = {model: list(dict.fromkeys(cols + ['pre1_z'])) for model, cols in features.items()}
    return features, contrasts


def evaluate_split(run, d, h, train, test, split_name):
    models, _ = specifications(run)
    train_ids, test_ids = set(d.loc[train, 'person_id']), set(d.loc[test, 'person_id'])
    assert not train_ids & test_ids
    n_train, n_test = int(train.sum()), int(test.sum())
    base = d.loc[test, ['cohort', 'person_id', 'target_episode_id']].copy()
    base['split'] = split_name
    base['n_train'] = n_train
    base['n_test'] = n_test
    if n_train < 2:
        raise ValueError('fewer_than_two_training_persons')
    mu = float(d.loc[train, 'pre1'].mean())
    sd = float(d.loc[train, 'pre1'].std(ddof=1))
    if run['scale'] == 'raw':
        mu, sd = 0.0, 1.0
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError('nonpositive_training_pre1_sd')
    # Transform endpoints before response construction, using one common pair of parameters.
    dz = d.copy()
    for point in ('pre2', 'pre1', 'event', 'year2'):
        dz[point + '_z'] = (d[point] - mu) / sd
    endpoint = 'event' if run['outcome'] == 'acute' else 'year2'
    y = dz[endpoint + '_z'].to_numpy(float)
    y_raw = d[endpoint].to_numpy(float)
    if run['formulation'] == 'change_score':
        y = y - dz.pre1_z.to_numpy(float)
        y_raw = y_raw - d.pre1.to_numpy(float)
    assert np.isfinite(y).all()
    dz['history'], residual_meta = history_feature(dz, h, train_ids, mu, sd, run['representation'], run['outcome'])
    context = encode_context(dz, multi=run['analysis'] == 'multi_event')
    base['y_observed'] = y[test]
    base['y_raw'] = y_raw[test]
    base['target_pre1_raw'] = d.loc[test, 'pre1'].to_numpy()
    base['training_pre1_mean'] = mu
    base['training_pre1_sd_ddof1'] = sd
    base['history_feature'] = dz.loc[test, 'history'].to_numpy()
    fit_rows = []
    for model, extras in models.items():
        x = context.copy()
        for name in extras:
            x[name] = dz[name].to_numpy(float)
        record = {'run_id': run['run_id'], 'split': split_name, 'model': model, 'n_train': n_train,
                  'n_test': n_test, 'training_pre1_mean': mu, 'training_pre1_sd_ddof1': sd,
                  'training_person_ids_sha256': hashlib.sha256(','.join(map(str, sorted(train_ids))).encode()).hexdigest(),
                  'test_person_ids_sha256': hashlib.sha256(','.join(map(str, sorted(test_ids))).encode()).hexdigest(),
                  'residual_history_fit': json.dumps(residual_meta, allow_nan=False) if residual_meta else ''}
        try:
            fitted = ols(x.loc[train], y[train], x.loc[test])
            pred, sigma2 = fitted['prediction'], fitted['sigma2']
            record.update({k: fitted[k] for k in ('rss_train', 'rank', 'n_design_columns', 'unsupported_rows')})
            record.update(sigma2_train=fitted['sigma2'], status='ok' if np.isfinite(pred).all() else 'nonestimable_test_design',
                          coefficient_json=json.dumps(fitted['coefficient'].tolist()),
                          predictor_columns=json.dumps(['intercept'] + list(x.columns)),
                          conditioning_mean_json=json.dumps(fitted['conditioning_mean'].tolist()),
                          conditioning_sd_json=json.dumps(fitted['conditioning_sd'].tolist()))
        except (ValueError, np.linalg.LinAlgError) as exc:
            pred, sigma2 = np.full(n_test, np.nan), np.nan
            record.update(sigma2_train=np.nan, status=str(exc), rss_train=np.nan, rank=0,
                          n_design_columns=len(x.columns) + 1, unsupported_rows=n_test,
                          coefficient_json='', predictor_columns=json.dumps(['intercept'] + list(x.columns)),
                          conditioning_mean_json='', conditioning_sd_json='')
        base['prediction_' + model] = pred
        base['prediction_raw_' + model] = pred * sd + (mu if run['formulation'] == 'ancova_level' else 0)
        base['sigma2_' + model] = sigma2
        if np.isfinite(sigma2) and sigma2 > 0:
            base['log_density_' + model] = -0.5 * (np.log(2 * np.pi * sigma2) + (y[test] - pred) ** 2 / sigma2)
        else:
            base['log_density_' + model] = np.nan
        fit_rows.append(record)
    return base, fit_rows


def metric_values(y, pred, log_density):
    result = dict.fromkeys(METRICS, np.nan)
    if len(y) == 0 or not np.isfinite(y).all() or not np.isfinite(pred).all():
        return result
    error = y - pred
    sst = float(np.sum((y - y.mean()) ** 2))
    result['predictive_r2'] = 1 - float(np.sum(error ** 2)) / sst if sst > 0 else np.nan
    result['rmse'] = float(np.sqrt(np.mean(error ** 2)))
    result['mae'] = float(np.mean(np.abs(error)))
    if np.isfinite(log_density).all():
        result['mean_log_predictive_density'] = float(np.mean(log_density))
    centered_pred = pred - pred.mean()
    denominator = float(np.sum(centered_pred ** 2))
    if len(y) >= 2 and denominator > 0:
        slope = float(np.sum(centered_pred * (y - y.mean())) / denominator)
        result['calibration_slope'] = slope
        result['calibration_intercept'] = float(y.mean() - slope * pred.mean())
    return result


def metric_bundle(y, predictions, densities, models, contrasts, ix=None):
    if ix is not None:
        y, predictions, densities = y[ix], predictions[ix], densities[ix]
    bundle = {}
    for i, model in enumerate(models):
        for metric, value in metric_values(y, predictions[:, i], densities[:, i]).items():
            bundle[model + '|' + metric] = value
    for augmented, baseline in contrasts:
        for metric in METRICS:
            bundle[augmented + '-' + baseline + '|' + metric] = bundle[augmented + '|' + metric] - bundle[baseline + '|' + metric]
    return bundle


def summarize_predictions(run, predictions, validation):
    model_map, contrasts = specifications(run)
    models = list(model_map)
    n = len(predictions)
    assert predictions.person_id.is_unique
    y = predictions.y_observed.to_numpy(float)
    pred = predictions[['prediction_' + m for m in models]].to_numpy(float)
    lp = predictions[['log_density_' + m for m in models]].to_numpy(float)
    point = metric_bundle(y, pred, lp, models, contrasts)
    keys = list(point)
    finite_point = np.isfinite(list(point.values()))
    seed = int(hashlib.sha256(f'{SEED}|{run["run_id"]}|{validation}'.encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    replicates = []
    attempts = 0
    # A fixed computational guard prevents infinite loops if valid bootstrap draws
    # are mathematically impossible. It does not change any sample or model.
    max_attempts = 500000
    if n >= 2 and finite_point.any():
        while len(replicates) < BOOTSTRAPS and attempts < max_attempts:
            attempts += 1
            ix = rng.randint(0, n, n)
            bundle = metric_bundle(y, pred, lp, models, contrasts, ix)
            values = np.asarray([bundle[k] for k in keys], float)
            if not np.isfinite(values[finite_point]).all():
                continue
            replicates.append(values)
    rep_array = np.asarray(replicates) if replicates else np.empty((0, len(keys)))
    rows = []
    for j, key in enumerate(keys):
        term, metric = key.split('|')
        value = point[key]
        estimated = bool(np.isfinite(value))
        lo = hi = np.nan
        if estimated and len(replicates) == BOOTSTRAPS:
            lo, hi = np.quantile(rep_array[:, j], [0.025, 0.975])
        rows.append({**run, 'validation': validation, 'row_type': 'increment' if '-' in term else 'model',
                     'term': term, 'metric': metric, 'estimate': value, 'ci_lower': lo, 'ci_upper': hi,
                     'n_evaluated': n, 'n_finite_model_predictions': int(np.isfinite(pred).all(axis=1).sum()),
                     'bootstrap_successful': len(replicates), 'bootstrap_attempted': attempts,
                     'bootstrap_seed': seed, 'status': 'estimated' if estimated else 'not_estimable',
                     'interval_status': '500_paired_person_resamples' if estimated and len(replicates) == BOOTSTRAPS else 'not_estimable',
                     'increment_definition': 'augmented_minus_baseline',
                     'source_prediction_file': f'archive/predictions/{run["run_id"]}_{validation}.parquet'})
    reps = pd.DataFrame(rep_array, columns=keys)
    reps.insert(0, 'bootstrap_id', np.arange(1, len(reps) + 1))
    reps.insert(0, 'validation', validation)
    reps.insert(0, 'run_id', run['run_id'])
    return pd.DataFrame(rows), reps


def fit_all():
    verify_sources(full=True)
    assert (VLOCK / 'manifest.json').exists(), 'Validation cases must be locked before fitting.'
    start = ARC / 'estimation_started.json'
    if not start.exists():
        seal_json(start, {'first_fit_start_utc': utc(), 'protocol_sha256': PROTOCOL_SHA, 'note_sha256': NOTE_SHA,
            'implementation_note_hash_record_sha256': sha(ARC / 'pre_estimation_implementation_hash.json'),
            'validation_manifest_sha256': sha(VLOCK / 'manifest.json'), 'engine_sha256': sha(Path(__file__)),
            'bootstrap_target_successful': BOOTSTRAPS, 'bootstrap_computational_attempt_guard': 500000,
            'bootstrap_type': 'paired person resampling of fixed held-out prediction bundles, not refitting models',
            'ols_solver': 'numpy.linalg.lstsq; training-only numerical conditioning; no regularization',
            'zero_or_nonfinite_density_variance': 'not estimable, no alternative method',
            'nonestimable_design': 'test rows outside training-design row space flagged; not silently dropped',
            'old_DRAFT_rules_used': False})
    targets, histories = get_data()
    runs = pd.read_parquet(VLOCK / 'run_registry.parquet').to_dict('records')
    members = pd.read_parquet(VLOCK / 'case_memberships.parquet')
    for sub in ('predictions', 'bootstrap', 'run_results', 'fit_records'):
        (ARC / sub).mkdir(exist_ok=True)
    completed = 0
    for run in runs:
        run_id = run['run_id']
        done = ARC / 'run_results' / f'{run_id}_complete.json'
        if done.exists():
            completed += 1
            continue
        verify_sources()
        case_members = members.loc[members.case_id.eq(run['case_id'])]
        d = targets.loc[targets.sample_id.eq(run['sample_id'])].merge(
            case_members[['person_id', 'outer_fold', 'temporal_assignment']], on='person_id', how='inner', validate='one_to_one')
        d = d.sort_values('person_id').reset_index(drop=True)
        h = histories.loc[histories.sample_id.eq(run['sample_id']) & histories.person_id.isin(d.person_id)].copy()
        assert len(d) == run['n_complete'] and d.person_id.is_unique
        fit_records, results = [], []
        cv_predictions = []
        for fold in range(10):
            train, test = d.outer_fold.ne(fold).to_numpy(), d.outer_fold.eq(fold).to_numpy()
            if not test.any():
                continue
            prediction, records = evaluate_split(run, d, h, train, test, f'fold_{fold}')
            cv_predictions.append(prediction)
            fit_records += records
        if cv_predictions:
            oof = pd.concat(cv_predictions, ignore_index=True).sort_values('person_id').reset_index(drop=True)
            assert len(oof) == len(d) and set(oof.person_id) == set(d.person_id)
        else:
            raise ValueError('No held-out cross-validation observations.')
        train, test = d.temporal_assignment.eq('train').to_numpy(), d.temporal_assignment.eq('test').to_numpy()
        temporal, records = evaluate_split(run, d, h, train, test, 'temporal')
        fit_records += records
        for validation, prediction in [('grouped_cv', oof), ('temporal', temporal)]:
            prediction.insert(0, 'run_id', run_id)
            prediction.insert(1, 'case_id', run['case_id'])
            path = ARC / 'predictions' / f'{run_id}_{validation}.parquet'
            prediction.to_parquet(path, index=False)
            result, boot = summarize_predictions(run, prediction, validation)
            results.append(result)
            boot.to_parquet(ARC / 'bootstrap' / f'{run_id}_{validation}.parquet', index=False)
        pd.concat(results, ignore_index=True).to_parquet(ARC / 'run_results' / f'{run_id}.parquet', index=False)
        pd.DataFrame(fit_records).to_parquet(ARC / 'fit_records' / f'{run_id}.parquet', index=False)
        seal_json(done, {'run_id': run_id, 'completed_at_utc': utc(), 'engine_sha256': sha(Path(__file__)),
                         'protocol_sha256': PROTOCOL_SHA, 'note_sha256': NOTE_SHA})
        completed += 1
        log(f'Completed {completed}/{len(runs)}: {run["sample_id"]}, {run["benchmark"]}, {run["representation"]}, {run["formulation"]}, {run["scale"]}')
    verify_sources(full=True)
    summary = pd.concat([pd.read_parquet(ARC / 'run_results' / f'{r["run_id"]}.parquet') for r in runs], ignore_index=True)
    summary.to_parquet(ARC / 'all_performance_results.parquet', index=False)
    fit = pd.concat([pd.read_parquet(ARC / 'fit_records' / f'{r["run_id"]}.parquet') for r in runs], ignore_index=True)
    fit.to_parquet(ARC / 'all_fit_records.parquet', index=False)
    software = {'completed_at_utc': utc(), 'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__,
                'platform': platform.platform(), 'engine_sha256': sha(Path(__file__)),
                'protocol_sha256': PROTOCOL_SHA, 'note_sha256': NOTE_SHA,
                'model_set_runs': len(runs), 'performance_rows': len(summary), 'fit_records': len(fit),
                'parent_inputs_and_results_unchanged': True}
    (ARC / 'software_environment.json').write_text(json.dumps(software, indent=2) + '\n')
    log('All registered runs completed. Performance files saved; reporting/independent checks remain.')


def synthetic_tests():
    # Engineering test data only; no cohort data is loaded.
    x = pd.DataFrame({'x': [-2., -1., 0., 1., 2.], 'constant': [1.] * 5})
    y = np.array([-2., 0., 0., 4., 3.])
    fit = ols(x, y, x)
    assert np.allclose(fit['prediction'], np.column_stack([np.ones(5), x.x]) @ np.linalg.lstsq(np.column_stack([np.ones(5), x.x]), y, rcond=None)[0])
    assert np.isclose(fit['sigma2'], np.sum((y - fit['prediction']) ** 2) / 5)
    metrics = metric_values(y, fit['prediction'], np.zeros(5))
    assert np.isfinite(list(metrics.values())).all()
    assert np.isclose(metrics['mean_log_predictive_density'], 0)
    log('Synthetic OLS, training-variance and metric tests passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'fit', 'synthetic_tests'])
    args = parser.parse_args()
    {'prepare': prepare, 'fit': fit_all, 'synthetic_tests': synthetic_tests}[args.phase]()
