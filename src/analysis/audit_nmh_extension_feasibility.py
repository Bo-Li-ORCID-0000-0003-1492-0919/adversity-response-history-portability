#!/usr/bin/env python3
"""Coefficient-blind extension eligibility audit: masks, IDs, dates and counts only.

Never imports the prior analysis modules; never computes response values.
Output CSV authoring is delegated to the companion artifact-tool exporter.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import hashlib
import json
import platform

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/NMH_extension_feasibility'
WORK = ROOT / 'outputs/nmh_extension_20260904'
FREEZE = OUT / '04_external_preregistration_freeze_DRAFT.md'
FREEZE_HASH = '10f81292d089158777381efb7b558aec9108fecd291fe705054965768b89e32e'
PRIMARY = ('widowhood', 'unemployment', 'caregiving', 'health')
REPEAT_PRIMARY = ('unemployment', 'caregiving')
COHORTS = ('UKHLS', 'HRS')
MONTH_DAYS = 30.4375
EPFILES = {c: ROOT / f'data_processed/NMH_{c}_event_episodes.parquet' for c in COHORTS}
UK_PANEL = ROOT / 'data_processed/ukhls_adult_core_panel.csv.gz'
HRS_CORE = ROOT / 'raw_data/HRS/randhrs1992_2022v1_STATA/randhrs1992_2022v1.dta'
HRS_FAMILY = ROOT / 'raw_data/HRS/randhrsfam1992_2022v1_STATA/randhrsfamr1992_2022v1.dta'
TIME_BOUNDS = (0, 8, 12, 18, 24, 30, 36, 48, 60, 120)
FIELDS = ['section', 'cohort', 'sample_scope', 'outcome', 'event_family', 'statistic', 'category', 'n', 'denominator', 'notes']
COUNTS = {1: [], 2: [], 3: []}
META = {}


def log(s):
    print(s, flush=True)


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def add(file, section, cohort, scope, outcome, family, statistic, n, category='', denominator=None, notes=''):
    COUNTS[file].append(dict(zip(FIELDS, [section, cohort, scope, outcome, family, statistic,
                                         str(category), int(n), None if denominator is None else int(denominator), notes])))


def valid(s, lo, hi):
    return pd.to_numeric(s, errors='coerce').between(lo, hi)


def before(a, b):
    return pd.notna(a) and pd.notna(b) and a < b


def months(a, b):
    return (b-a).days / MONTH_DAYS if pd.notna(a) and pd.notna(b) else None


def time_bin(v):
    if v is None or pd.isna(v):
        return 'missing_actual_dates'
    if v < 0:
        return 'negative_invalid'
    for lo, hi in zip(TIME_BOUNDS[:-1], TIME_BOUNDS[1:]):
        if lo <= v < hi:
            return f'[{lo},{hi})_months'
    return '[120,infinity)_months'


def date_from_stata(s):
    x = pd.to_numeric(s, errors='coerce')
    ok = x.between(0, 50000)
    out = pd.Series(pd.NaT, index=x.index, dtype='datetime64[ns]')
    # Convert only valid integer Stata days; avoid overflow on float missing codes.
    out.loc[ok] = pd.Timestamp('1960-01-01') + pd.to_timedelta(x.loc[ok].astype('int64'), unit='D')
    return out


def load_episode_masks(cohort):
    columns = ['cohort', 'person_id', 'episode_id', 'event_wave', 'event_date', 'event_year',
               'primary_event_type', 'primary_event_labels', 'secondary_event_labels', 'all_event_labels',
               'n_primary_labels', 'n_secondary_labels', 'compound_primary', 'isolated_primary',
               'pre2_wave', 'pre1_wave', 'year2_wave', 'pre2_date', 'pre1_date', 'year2_date',
               'time_selection_method', 'event_self_report', 'pre1_self_report',
               'age_at_event', 'sex', 'education', 'education_source_wave', 'baseline_socioeconomic',
               'year2_censored_by_later_primary', 'preceding_primary_episode_count',
               'outcome_pre2_raw', 'outcome_pre1_raw', 'outcome_event_raw', 'outcome_year2_raw']
    e = pd.read_parquet(EPFILES[cohort], columns=columns)
    assert e.episode_id.is_unique and not e.duplicated(['person_id', 'event_wave']).any()
    for point in ('pre2', 'pre1', 'event', 'year2'):
        source = f'outcome_{point}_raw'
        e[f'{point}_available'] = valid(e[source], 0, 36 if cohort == 'UKHLS' else 8)
        del e[source]  # No raw mental-health values survive this input boundary.
    e['pre1_available'] &= e.pre1_self_report
    e['event_available'] &= e.event_self_report
    e['year2_available'] &= ~e.year2_censored_by_later_primary
    e['acute_complete'] = e.pre1_available & e.event_available
    e['persistence_complete'] = e.pre1_available & e.year2_available
    e['joint_complete'] = e.acute_complete & e.persistence_complete
    e['age_known'] = valid(e.age_at_event, 16, 110)
    e['sex_known'] = e.sex.isin(['male', 'female'])
    e['education_known'] = e.education.isin(['degree', 'other_qualification', 'no_qualification'])
    e['socio_known'] = e.baseline_socioeconomic.notna() & e.baseline_socioeconomic.ne('unknown')
    e['year_known'] = e.event_year.notna()
    e['baseline_usable'] = e.age_known & e.sex_known & e.year_known
    if cohort == 'UKHLS':
        future_edu = e.education_known & e.education_source_wave.gt(e.pre1_wave)
        assert not future_edu.any(), 'Existing education timing is not pre-event.'
    # Replace other participant characteristics by availability flags as well.
    e = e.drop(columns=['age_at_event', 'sex', 'education', 'baseline_socioeconomic'])
    for c in ('event_date', 'pre1_date', 'pre2_date', 'year2_date'):
        e[c] = pd.to_datetime(e[c], errors='coerce')
    e['index_valid'] = (e.event_self_report & e.pre1_self_report & e.event_wave.sub(e.pre1_wave).eq(1)
                        & e.event_date.gt(e.pre1_date))
    if cohort == 'UKHLS':
        gap = (e.event_date-e.pre1_date).dt.days / MONTH_DAYS
        e['index_valid'] &= gap.between(8, 18)
    e['family'] = e.primary_event_type
    financial = (e.n_primary_labels.eq(0) & e.n_secondary_labels.eq(1)
                 & e.secondary_event_labels.eq('financial_strain'))
    e.loc[financial, 'family'] = 'financial_strain'
    e['allowed_label'] = ((e.isolated_primary & e.family.isin(PRIMARY)) | financial) & ~e.compound_primary
    e['recurrence_valid'] = True
    e['recurrence_status'] = 'first_observed_onset'
    e['qualification_date'] = e.event_date
    e['confirmation_valid'] = True
    META[cohort] = {'source_episode_rows': len(e), 'source_episode_persons': int(e.person_id.nunique())}
    return e.sort_values(['person_id', 'event_wave', 'episode_id']).reset_index(drop=True)


def load_states(cohort, persons):
    if cohort == 'UKHLS':
        cols = ['pidp', 'wave', 'interview_date', 'ivfio', 'jbstat', 'caregiver', 'health', 'finnow']
        p = pd.read_csv(UK_PANEL, usecols=cols, low_memory=False).rename(columns={'pidp': 'person_id'})
        p = p.loc[p.person_id.isin(persons)].copy()
        p['interview_date'] = pd.to_datetime(p.interview_date, errors='coerce')
        p['self_report'] = p.ivfio.isin([1, 3])
    else:
        requested = ['hhidpn']
        for w in range(1, 17):
            requested += [f'r{w}{s}' for s in ('iwmid', 'iwstat', 'proxy', 'lbrf', 'adl5a')]
        fam_requested = ['hhidpn'] + [f'r{w}ppcr' for w in range(1, 17)]
        with pd.io.stata.StataReader(HRS_CORE, convert_categoricals=False) as f:
            available = set(f.variable_labels())
        with pd.io.stata.StataReader(HRS_FAMILY, convert_categoricals=False) as f:
            fam_available = set(f.variable_labels())
        cols = [x for x in requested if x in available]
        fcols = [x for x in fam_requested if x in fam_available]
        META['HRS']['targeted_core_columns'] = cols
        META['HRS']['targeted_family_columns'] = fcols
        META['HRS']['unavailable_requested_columns'] = sorted(set(requested+fam_requested)-set(cols+fcols))
        wide = pd.read_stata(HRS_CORE, columns=cols, convert_categoricals=False, convert_dates=False)
        family = pd.read_stata(HRS_FAMILY, columns=fcols, convert_categoricals=False, convert_dates=False)
        wide = wide.loc[wide.hhidpn.isin(persons)].merge(family, on='hhidpn', how='left', validate='one_to_one')
        chunks = []
        for w in range(1, 17):
            get = lambda stem: pd.to_numeric(wide.get(f'r{w}{stem}', pd.Series(float('nan'), index=wide.index)), errors='coerce')
            chunks.append(pd.DataFrame({'person_id': wide.hhidpn.astype('int64'), 'wave': w,
                'interview_date': date_from_stata(get('iwmid')),
                'self_report': get('iwstat').isin([1, 2]) & get('proxy').eq(0),
                'lbrf': get('lbrf'), 'adl': get('adl5a'), 'ppcr': get('ppcr')}))
        p = pd.concat(chunks, ignore_index=True)
    assert not p.duplicated(['person_id', 'wave']).any()
    META[cohort]['state_rows_loaded'] = len(p)
    # This is the only targeted state extraction; no psychological values read.
    return {(int(r['person_id']), int(r['wave'])): r for r in p.to_dict('records')}


def reset_between(cohort, family, earlier, later, states):
    n = 2 if family == 'health' else 1
    prior = [states.get((int(later['person_id']), int(later['event_wave'])-k)) for k in range(n, 0, -1)]
    if any(p is None or not p['self_report'] or pd.isna(p['interview_date']) for p in prior):
        return False, 'missing_or_proxy_washout_wave'
    if any(p['wave'] <= earlier['event_wave'] or not before(earlier['event_date'], p['interview_date'])
           or not before(p['interview_date'], later['event_date']) for p in prior):
        return False, 'no_completed_washout_between_onsets'
    if n == 2:
        if not before(prior[0]['interview_date'], prior[1]['interview_date']):
            return False, 'unordered_washout_dates'
        if cohort == 'UKHLS' and not 8 <= months(prior[0]['interview_date'], prior[1]['interview_date']) <= 18:
            return False, 'washout_adjacent_gap_outside_8_18'
    if cohort == 'UKHLS':
        col, good = {'unemployment': ('jbstat', {1, 2}), 'caregiving': ('caregiver', {0}),
                     'health': ('health', {2}), 'financial_strain': ('finnow', {1, 2, 3})}[family]
    else:
        col, good = {'unemployment': ('lbrf', {1, 2, 4, 5, 6, 7}),
                     'caregiving': ('ppcr', {0}), 'health': ('adl', {0})}[family]
    if not all(p[col] in good for p in prior):
        return False, 'normal_state_not_observed'
    return True, 'verified_recurrence'


def add_recurrence_flags(cohort, e, states):
    histories = {}
    audit = Counter()
    date_mismatch = 0
    for i, r in e.iterrows():
        check = states.get((int(r.person_id), int(r.event_wave)))
        if check and pd.notna(r.event_date) and pd.notna(check['interview_date']) and r.event_date != check['interview_date']:
            date_mismatch += 1
        labels = str(r.all_event_labels).split('|')
        for family in labels:
            if family not in (*PRIMARY, 'financial_strain'):
                continue
            key = (int(r.person_id), family)
            prev = histories.get(key)
            status, ok = 'first_observed_onset', True
            if prev is not None:
                if family == 'widowhood':
                    ok, status = False, 'widowhood_recurrence_excluded'
                else:
                    ok, status = reset_between(cohort, family, prev, r, states)
            if r.family == family:
                e.at[i, 'recurrence_valid'] = ok
                e.at[i, 'recurrence_status'] = status
            audit[(family, status)] += 1
            histories[key] = r
        if r.family == 'health':
            nxt = states.get((int(r.person_id), int(r.event_wave)+1))
            okay = bool(nxt and nxt['self_report'] and before(r.event_date, nxt['interview_date']))
            if okay:
                okay = (nxt['health'] == 1 and 8 <= months(r.event_date, nxt['interview_date']) <= 18) if cohort == 'UKHLS' else nxt['adl'] in {1, 2, 3, 4, 5}
            e.at[i, 'confirmation_valid'] = okay
            e.at[i, 'qualification_date'] = nxt['interview_date'] if okay else pd.NaT
    assert date_mismatch == 0, f'{cohort}: extracted dates disagree with frozen episode dates.'
    for (family, reason), n in audit.items():
        add(1, 'onset_recurrence_QA', cohort, 'all_source_labels', 'availability_only', family, 'onsets', n, reason)
    META[cohort]['event_date_mismatches'] = date_mismatch
    e['structural_eligible'] = e.allowed_label & e.index_valid & e.recurrence_valid & e.confirmation_valid
    for flag in ('allowed_label', 'index_valid', 'recurrence_valid', 'confirmation_valid', 'structural_eligible',
                 'compound_primary', 'year2_censored_by_later_primary'):
        add(1, 'source_QA', cohort, 'source_episode_rows', 'availability_only', 'all', flag, int(e[flag].sum()), denominator=len(e))
    financial_with_primary = e.all_event_labels.str.split('|').map(lambda x: 'financial_strain' in x) & e.n_primary_labels.gt(0)
    if cohort == 'UKHLS':
        add(1, 'source_QA', cohort, 'financial_secondary', 'availability_only', 'financial_strain',
            'financial_onset_with_primary_excluded', int(financial_with_primary.sum()))
    return e


def ordered(e):
    return e.sort_values(['event_date', 'event_wave', 'episode_id']).to_dict('records')


def pair_sort(p):
    return (p['earlier_event_date'], p['earlier_event_wave'], p['earlier_episode_id'],
            p['event_date'], p['event_wave'], p['target_episode_id'])


def target_record(r):
    names = ['cohort', 'person_id', 'family', 'event_wave', 'event_date', 'event_year', 'pre1_wave', 'pre2_wave',
             'year2_wave', 'pre1_date', 'pre2_date', 'year2_date', 'pre1_available', 'pre2_available',
             'event_available', 'year2_available', 'acute_complete', 'persistence_complete', 'joint_complete',
             'age_known', 'sex_known', 'education_known', 'socio_known', 'baseline_usable', 'time_selection_method']
    return {**{k: r[k] for k in names}, 'target_episode_id': r['episode_id']}


def pair_record(a, b):
    return {**target_record(b), 'earlier_episode_id': a['episode_id'], 'earlier_family': a['family'],
            'earlier_event_date': a['event_date'], 'earlier_event_wave': a['event_wave'],
            'earlier_year2_date': a['year2_date'], 'earlier_year2_wave': a['year2_wave'],
            'earlier_joint_complete': a['joint_complete'],
            'event_interval_months': months(a['event_date'], b['event_date']),
            'completion_to_target_pre_months': months(a['year2_date'], b['pre1_date'])}


def summarize_targets(file, section, cohort, scope, outcome, rows, family='all'):
    n = len(rows)
    persons = len({r['person_id'] for r in rows})
    add(file, section, cohort, scope, outcome, family, 'rows', n)
    add(file, section, cohort, scope, outcome, family, 'unique_persons', persons)
    for flag in ('pre1_available', 'pre2_available', 'acute_complete', 'persistence_complete',
                 'joint_complete', 'age_known', 'sex_known', 'education_known', 'socio_known', 'baseline_usable'):
        k = sum(bool(r[flag]) for r in rows)
        add(file, section, cohort, scope, outcome, family, flag+'_n', k, denominator=n)
        add(file, section, cohort, scope, outcome, family, flag+'_missing_n', n-k, denominator=n)
    for category, k in sorted(Counter(str(int(r['event_year'])) if pd.notna(r['event_year']) else 'missing' for r in rows).items()):
        add(file, section+'_calendar', cohort, scope, outcome, family, 'target_year_n', k, category, n)
    for category, k in sorted(Counter(r['family'] for r in rows).items()):
        add(file, section+'_target_family', cohort, scope, outcome, family, 'target_family_n', k, category, n)
    for gap in ('event_interval_months', 'completion_to_target_pre_months'):
        if rows and gap in rows[0]:
            for category, k in sorted(Counter(time_bin(r[gap]) for r in rows).items()):
                add(file, section+'_interval', cohort, scope, outcome, family, gap+'_n', k, category, n)


def one_per_person(rows, earlier_first=False):
    rows = sorted(rows, key=pair_sort if earlier_first else lambda r: (r['event_date'], r['event_wave'], r['target_episode_id']))
    keep = {}
    for r in rows:
        keep.setdefault(int(r['person_id']), r)
    return list(keep.values())


def same_domain(cohort, e, states):
    d = e.loc[e.structural_eligible & e.family.isin(['unemployment', 'caregiving', 'health', 'financial_strain'])]
    candidates = []
    funnel = Counter()
    for _, g in d.groupby('person_id', sort=False):
        for a, b in combinations(ordered(g), 2):
            if a['family'] != b['family']:
                continue
            family = a['family']
            funnel[(family, 'same_family_distinct_onset_pairs')] += 1
            ok, _ = reset_between(cohort, family, a, b, states)
            if not ok:
                continue
            funnel[(family, 'verified_reset_pairs')] += 1
            if not before(a['year2_date'], b['event_date']):
                continue
            funnel[(family, 'later_event_after_earlier_year2_pairs')] += 1
            if not before(a['year2_date'], b['pre1_date']) or not a['year2_wave'] < b['pre1_wave']:
                continue
            funnel[(family, 'nonoverlapping_onset_pairs')] += 1
            p = pair_record(a, b)
            candidates.append(p)
            if a['joint_complete']:
                funnel[(family, 'earlier_acute_and_year2_complete_pairs')] += 1
                if b['acute_complete']:
                    funnel[(family, 'both_acute_complete_pairs')] += 1
                if b['joint_complete']:
                    funnel[(family, 'both_joint_complete_pairs')] += 1
    stages = ('same_family_distinct_onset_pairs', 'verified_reset_pairs',
              'later_event_after_earlier_year2_pairs', 'nonoverlapping_onset_pairs',
              'earlier_acute_and_year2_complete_pairs', 'both_acute_complete_pairs',
              'both_joint_complete_pairs')
    audited_families = ('unemployment', 'caregiving', 'health') + (('financial_strain',) if cohort == 'UKHLS' else ())
    for family in audited_families:
        values = [funnel[(family, stage)] for stage in stages]
        assert all(a >= b for a, b in zip(values[:-1], values[1:])), (cohort, family, values)
        for stage, n in zip(stages, values):
            add(1, 'all_candidate_pair_funnel', cohort, 'before_one_person_selection', 'availability_only', family, stage, n)
    scopes = {'primary_pool': REPEAT_PRIMARY, 'unemployment': ('unemployment',), 'caregiving': ('caregiving',), 'health_secondary': ('health',)}
    if cohort == 'UKHLS':
        scopes['financial_secondary'] = ('financial_strain',)
    selections = []
    for scope, families in scopes.items():
        options = [p for p in candidates if p['family'] in families]
        for outcome in ('onset_timing', 'acute', 'persistence'):
            q = options
            if outcome != 'onset_timing':
                q = [p for p in q if p['earlier_joint_complete'] and p['baseline_usable']
                     and (p['acute_complete'] if outcome == 'acute' else p['joint_complete'])]
            chosen = one_per_person(q, earlier_first=True)
            summarize_targets(1, 'same_domain', cohort, scope, outcome, chosen)
            for p in chosen:
                selections.append({**p, 'sample_scope': scope, 'outcome': outcome})
    return selections


def comparison_targets(cohort, e, states):
    d = e.loc[e.structural_eligible]
    registry = []
    selected_dual = []
    for _, g in d.groupby('person_id', sort=False):
        records = ordered(g)
        for j, target in enumerate(records):
            if not target['acute_complete'] or not target['baseline_usable']:
                continue
            hist = [h for h in records[:j] if h['joint_complete']
                    and before(h['year2_date'], target['pre1_date'])
                    and h['year2_wave'] < target['pre1_wave']
                    and before(h['qualification_date'], target['pre1_date'])]
            same = [h for h in hist if h['family'] == target['family'] and target['family'] != 'widowhood'
                    and reset_between(cohort, target['family'], h, target, states)[0]]
            cross = [h for h in hist if h['family'] in PRIMARY and h['family'] != target['family']]
            if not same and not cross:
                continue
            for outcome in ('acute', 'persistence'):
                if outcome == 'persistence' and not target['joint_complete']:
                    continue
                registry.append({**target_record(target), 'outcome': outcome,
                                 'same_earlier_episode_id': same[0]['episode_id'] if same else '',
                                 'cross_earlier_episode_id': cross[0]['episode_id'] if cross else '',
                                 'same_available': bool(same), 'cross_available': bool(cross)})
    families = (*PRIMARY, 'financial_strain') if cohort == 'UKHLS' else PRIMARY
    for outcome in ('acute', 'persistence'):
        z = [r for r in registry if r['outcome'] == outcome]
        for family in families:
            f = [r for r in z if r['family'] == family]
            arms = {}
            for arm in ('same', 'cross', 'both'):
                q = ([r for r in f if r['same_available'] and r['cross_available']] if arm == 'both'
                     else [r for r in f if r[arm+'_available']])
                arms[arm] = q
                summarize_targets(1, 'comparable_target_registry', cohort, arm, outcome, q, family)
                one = one_per_person(q)
                summarize_targets(1, 'comparable_target_one_person_per_family', cohort, arm, outcome, one, family)
            years_same = {int(r['event_year']) for r in arms['same']}
            years_cross = {int(r['event_year']) for r in arms['cross']}
            common = sorted(years_same & years_cross)
            add(1, 'calendar_support', cohort, 'same_cross_common_years', outcome, family, 'n_distinct_common_years', len(common))
            for arm in ('same', 'cross'):
                years = sorted({int(r['event_year']) for r in arms[arm]})
                if years:
                    add(1, 'calendar_support', cohort, arm, outcome, family, 'year_range', len(arms[arm]), f'{years[0]}..{years[-1]}')
                for year in common:
                    q = [r for r in arms[arm] if int(r['event_year']) == year]
                    add(1, 'calendar_support', cohort, arm, outcome, family, 'common_year_target_events', len(q), year)
                    add(1, 'calendar_support', cohort, arm, outcome, family, 'common_year_unique_persons', len({r['person_id'] for r in q}), year)
        both_primary = one_per_person([r for r in z if r['family'] in REPEAT_PRIMARY and r['same_available'] and r['cross_available']])
        summarize_targets(1, 'direct_paired_target_sample', cohort, 'primary_common_targets', outcome, both_primary)
        selected_dual += [{**r, 'sample_scope': 'primary_common_targets'} for r in both_primary]
    return registry, selected_dual


def multi_event(cohort, e):
    d = e.loc[e.structural_eligible & e.family.isin(PRIMARY)]
    ready = defaultdict(list)
    memberships = {}
    for _, g in d.groupby('person_id', sort=False):
        records = ordered(g)
        for j, target in enumerate(records):
            for outcome in ('acute', 'persistence', 'joint'):
                complete = 'event_date' if outcome == 'acute' else 'year2_date'
                hist = [h for h in records[:j] if h[outcome+'_complete']
                        and before(h[complete], target['pre1_date'])
                        and before(h['qualification_date'], target['pre1_date'])
                        and (h['event_wave'] if outcome == 'acute' else h['year2_wave']) < target['pre1_wave']]
                if len(hist) < 2 or not any(h['family'] != target['family'] for h in hist):
                    continue
                overlap_n = sum(not before(a[complete], b['pre1_date']) for a, b in combinations(hist, 2))
                row = {**target_record(target), 'outcome': outcome, 'prior_episode_count': len(hist),
                       'prior_family_count': len({h['family'] for h in hist}),
                       'different_family_prior_count': sum(h['family'] != target['family'] for h in hist),
                       'overlapping_history_window_pairs': overlap_n,
                       'prior_age_all_known': all(h['age_known'] for h in hist),
                       'prior_education_all_known': all(h['education_known'] for h in hist),
                       'target_metric_complete': bool(target[outcome+'_complete'])}
                ready[outcome].append(row)
                memberships[(outcome, target['episode_id'])] = hist
    selected, links = [], []
    for outcome in ('acute', 'persistence', 'joint'):
        for scope in ('history_ready', 'eligible'):
            options = ready[outcome]
            if scope == 'eligible':
                options = [r for r in options if r['target_metric_complete'] and r['baseline_usable']]
            chosen = one_per_person(options)
            summarize_targets(2, 'multi_event', cohort, scope, outcome, chosen)
            for variable in ('prior_episode_count', 'prior_family_count', 'different_family_prior_count'):
                for k, n in sorted(Counter(r[variable] for r in chosen).items()):
                    add(2, 'history_distribution', cohort, scope, outcome, 'all', variable, n, k, len(chosen))
            for variable in ('prior_age_all_known', 'prior_education_all_known'):
                add(2, 'history_covariate_availability', cohort, scope, outcome, 'all', variable, sum(r[variable] for r in chosen), denominator=len(chosen))
            add(2, 'history_window_QA', cohort, scope, outcome, 'all', 'persons_with_overlapping_history_windows',
                sum(r['overlapping_history_window_pairs'] > 0 for r in chosen), denominator=len(chosen))
            add(2, 'history_window_QA', cohort, scope, outcome, 'all', 'history_window_overlap_pairs',
                sum(r['overlapping_history_window_pairs'] for r in chosen))
            for r in chosen:
                selected.append({**r, 'sample_scope': scope})
                for order, h in enumerate(memberships[(outcome, r['target_episode_id'])], 1):
                    completion = h['event_date'] if outcome == 'acute' else h['year2_date']
                    assert before(completion, r['pre1_date']) and before(h['qualification_date'], r['pre1_date'])
                    links.append({'cohort': cohort, 'person_id': r['person_id'], 'sample_scope': scope, 'outcome': outcome,
                        'target_episode_id': r['target_episode_id'], 'history_order': order,
                        'history_episode_id': h['episode_id'], 'history_family': h['family'],
                        'history_completion_date': completion, 'history_qualification_date': h['qualification_date'],
                        'information_cutoff_date': r['pre1_date']})
    return selected, links


def recency(same, multi, direct):
    output = []
    groups = defaultdict(list)
    for kind, rows in [('same_domain', same), ('multi_event', multi), ('direct_same_cross', direct)]:
        for r in rows:
            groups[(kind, r['cohort'], r['sample_scope'], r['outcome'])].append(r)
    for (kind, cohort, scope, outcome), rows in groups.items():
        for family in ('all', *sorted({r['family'] for r in rows})):
            q = rows if family == 'all' else [r for r in rows if r['family'] == family]
            counts = Counter()
            for r in q:
                gap = months(r['pre2_date'], r['pre1_date'])
                to_event = months(r['pre1_date'], r['event_date'])
                known = gap is not None and gap > 0 and r['pre2_wave'] < r['pre1_wave']
                both = bool(r['pre1_available'] and r['pre2_available'])
                complete = both and known
                for flag, yes in [('latest_pre1_available', r['pre1_available']), ('additional_pre2_available', r['pre2_available']),
                                  ('both_measures_available', both), ('actual_gap_known_ordered', known),
                                  ('complete_measures_and_actual_gap', complete)]:
                    counts[flag] += bool(yes)
                counts['gap:'+time_bin(gap)] += 1
                counts['event_gap:'+time_bin(to_event)] += 1
                if complete:
                    counts['complete_gap:'+time_bin(gap)] += 1
                if family == 'all':
                    output.append({**r, 'sample_kind': kind, 'pre2_to_pre1_months': gap,
                                   'pre1_to_event_months': to_event, 'recency_complete': complete})
            add(3, kind, cohort, scope, outcome, family, 'target_rows', len(q))
            add(3, kind, cohort, scope, outcome, family, 'unique_persons', len({r['person_id'] for r in q}))
            for name, n in counts.items():
                if ':' in name:
                    statistic, category = name.split(':', 1)
                    add(3, kind, cohort, scope, outcome, family, statistic, n, category, len(q))
                else:
                    add(3, kind, cohort, scope, outcome, family, name, n, denominator=len(q))
                    add(3, kind, cohort, scope, outcome, family, name+'_missing', len(q)-n, denominator=len(q))
    return output


def threshold_counts(same, multi, direct, recent):
    for outcome in ('acute', 'persistence'):
        main = [r for r in same if r['sample_scope'] == 'primary_pool' and r['outcome'] == outcome]
        n = len(main)
        add(1, 'fixed_sample_size_gates', 'UKHLS+HRS_count_only', 'primary_pool', outcome, 'all', 'n_for_pooled_500_gate', n,
            'met' if n >= 500 else 'not_met', notes='No outcome pooling; source cohort plus ID is the person key.')
        for family in REPEAT_PRIMARY:
            add(1, 'pooled_family_count_only', 'UKHLS+HRS_count_only', 'primary_pool', outcome, family,
                'unique_persons', sum(r['family'] == family for r in main),
                notes='Descriptive total only; the frozen family gate is applied within each cohort.')
        for cohort in COHORTS:
            c = [r for r in main if r['cohort'] == cohort]
            add(1, 'fixed_sample_size_gates', cohort, 'primary_pool', outcome, 'all', 'n_for_cohort_300_gate', len(c), 'met' if len(c) >= 300 else 'secondary_only')
            for family in REPEAT_PRIMARY:
                k = sum(r['family'] == family for r in c)
                add(1, 'fixed_sample_size_gates', cohort, 'primary_pool', outcome, family, 'n_for_family_200_gate', k, 'met' if k >= 200 else 'secondary_only')
            k = sum(r['cohort'] == cohort and r['outcome'] == outcome for r in direct)
            add(1, 'fixed_sample_size_gates', cohort, 'primary_common_targets', outcome, 'all', 'n_for_cohort_300_gate', k, 'met' if k >= 300 else 'secondary_only')
    for outcome in ('acute', 'persistence', 'joint'):
        passed = 0
        for cohort in COHORTS:
            k = sum(r['cohort'] == cohort and r['outcome'] == outcome and r['sample_scope'] == 'eligible' for r in multi)
            passed += int(k >= 300)
            add(2, 'fixed_sample_size_gates', cohort, 'eligible', outcome, 'all', 'n_for_cohort_300_gate', k, 'met' if k >= 300 else 'secondary_only')
        add(2, 'fixed_sample_size_gates', 'both_cohorts', 'eligible', outcome, 'all', 'cohorts_meeting_300', passed,
            'no_multi_event_primary_analysis' if passed == 0 else 'cohort_specific_roles_as_listed')
    for kind in ('same_domain', 'multi_event', 'direct_same_cross'):
        relevant = [r for r in recent if r['sample_kind'] == kind and r['recency_complete'] and
                    r['sample_scope'] in ('primary_pool', 'eligible', 'primary_common_targets') and r['outcome'] != 'onset_timing']
        for cohort in COHORTS:
            for outcome in ('acute', 'persistence', 'joint'):
                q = [r for r in relevant if r['cohort'] == cohort and r['outcome'] == outcome]
                if kind != 'multi_event' and outcome == 'joint':
                    continue
                add(3, 'fixed_recency_sample_gates', cohort, kind, outcome, 'all', 'complete_n_for_cohort_300_gate', len(q), 'met' if len(q) >= 300 else 'secondary_only')


def table(rows, cols):
    return '| '+' | '.join(cols)+' |\n| '+' | '.join('---' for _ in cols)+' |\n' + '\n'.join(
        '| '+' | '.join(str(r.get(c, '')).replace('|', '/') for c in cols)+' |' for r in rows)


def write_reports(same, multi, direct, recent, source_manifest):
    common = [
        '本轮仅构建资格、统计计数与缺失；未查看新系数，未计算响应分数、均值、相关、R²、AIC、BIC或p值，未运行任何预测模型。',
        f'计数前冻结草案SHA-256：`{FREEZE_HASH}`。原有确认性episode与结果文件在运行前后哈希一致。',
        '分母以各表标注为准。人数为cohort内unique person；两cohort合计使用cohort+person_id，绝不混合心理量表原分。',
        '心理评分只被转换成合法/缺失标志；所有保存的资格清单均无心理评分、响应值或效应系数。',
    ]
    summary = []
    for cohort in COHORTS:
        scopes = ['primary_pool', 'unemployment', 'caregiving', 'health_secondary'] + (['financial_secondary'] if cohort == 'UKHLS' else [])
        for scope in scopes:
            z = [r for r in same if r['cohort'] == cohort and r['sample_scope'] == scope]
            summary.append({'cohort': cohort, 'scope': scope,
                'onset/timing persons=pairs': sum(r['outcome'] == 'onset_timing' for r in z),
                'acute complete persons=pairs': sum(r['outcome'] == 'acute' for r in z),
                'persistence complete persons=pairs': sum(r['outcome'] == 'persistence' for r in z)})
    gates = [r for r in COUNTS[1] if r['section'] == 'fixed_sample_size_gates']
    comparable = []
    for r in COUNTS[1]:
        if r['section'] == 'comparable_target_registry' and r['statistic'] in ('rows', 'unique_persons'):
            comparable.append({k:r[k] for k in ('cohort','outcome','event_family','sample_scope','statistic','n')})
    lines = ['# Same-domain positive-control feasibility audit', '', *common, '',
        '## 数据和资格规则', '',
        '- Source：`data_processed/NMH_UKHLS_event_episodes.parquet`和`data_processed/NMH_HRS_event_episodes.parquet`；没有从原始数据添加新episode。',
        '- 复发状态来自既有UKHLS panel及定向HRS状态提取；没有读取原始GHQ/CES-D题项或再次扫描全套原始文件。',
        '- unemployment/caregiving为同域primary候选；health及UKHLS financial单独secondary。排除重复widowhood和retirement。详见冻结草案逐family编码。',
        '- 仅isolated primary；financial secondary须其transition没有任何primary或其他secondary标签。primary+secondary标签在primary样本中仍沿用原先isolated定义，不擅自删除。',
        '- 每一scope/outcome每人一对：最早有合格后续配对的较早episode，加其第一后续同类episode。按日期、wave、ID固定排序。',
        '- 两次窗口不共享访谈：earlier year2 < later pre1 < later event，且波次有序；较早事件的acute及未censor year2必须完成。',
        '- onset/timing栏是完整性筛选前的纯事件分母，不是可建模样本；acute和persistence栏各自按最早有效配对规则选取。不能把不同栏当成完全相同的episode配对。',
        '- 完整样本要求目标age/sex/year可用；education unknown沿用既有独立类别，同时单独报告known/missing。', '',
        '## 一人一对计数', '', table(summary, list(summary[0])), '',
        'primary_pool已跨unemployment/caregiving去重；单family行各自去重，可能共享persons，不可求和代替pool。两年完整样本同时具有两次acute和两次persistence结果。', '',
        '## 固定门槛核对（不是效应判断）', '', table(gates, ['cohort','sample_scope','outcome','event_family','statistic','n','category']), '',
        'family 200门槛按primary_pool中实际保留配对的target family、在每cohort内执行；未达门槛不通过改选另一对或增加health/financial补足。', '',
        '## 同域/跨域可比较目标 registry', '',
        '下表分别给目标event条数和unique persons。同一个目标可同时有same/cross历史；both是该交集。并非把两组不同人的结果作差。family内每人最早目标计数另见CSV的comparable_target_one_person_per_family。', '',
        table(comparable, ['cohort','outcome','event_family','sample_scope','statistic','n']), '',
        '## 时间、日历与缺失', '',
        '详见01_same_domain_counts.csv：same_domain_calendar按目标整数年；same_domain_interval按固定月份区间；calendar_support逐年记录两臂实际共同年份及每臂N。n_distinct_common_years=0表示没有共同年份，不用范围相交替代。',
        '同/跨域两臂使用同一eligibility函数与baseline规则；age、sex、education、current pre1缺失分别报告。完整目标pre1率为100%是纳入条件导致，不是原始数据无缺失。onset/timing分母保留实际pre1缺失。', '',
        '## QA与限制', '',
        '- all_candidate_pair_funnel为去重前pair漏斗；其数量不能与每人一对混淆。每一步非递增已核验。',
        '- 恢复状态只表明访谈时观察到normal state，不证明连续整个间隔无逆境；health两个normal waves仍可能包含报告波动。',
        '- 未知或代理washout不算正常wave。日期不足或窗口共享者不以名义间隔补入主资格。',
        '- 既有后续primary-event censoring始终保留；financial未额外成为primary censoring事件。',
        '- HRS父母照护不同于UKHLS任意照护；health定义分别为持续ADL/持续长期疾病残疾。',
        '- 事件真实开始/结束日期未知；当前pre1可能已经位于真实暴露期；严格时间筛选并不建立因果前后关系。',
        '- health持续确认可能需要未来一波；任何历史只在该确认已完成时进入目标历史。',
        '- 所有family计数和pair漏斗阶段均报告，包括0；onset复发失败原因列出实际观察到的类别。不会据N调整复发定义或窗口。', '',
        '## 可复现记录', '',
        '资格清单、source/规则哈希及运行环境保存在outputs/nmh_extension_20260904。主计数表：01_same_domain_counts.csv。冻结草案未外部注册，下一轮须人工审核并外部登记后才执行估计。', '']
    (OUT/'01_same_domain_feasibility_audit.md').write_text('\n'.join(lines))
    msummary = []
    for cohort in COHORTS:
        for outcome in ('acute','persistence','joint'):
            z = [r for r in multi if r['cohort']==cohort and r['outcome']==outcome]
            ready = [r for r in z if r['sample_scope']=='history_ready']
            eligible = [r for r in z if r['sample_scope']=='eligible']
            msummary.append({'cohort':cohort,'outcome':outcome,'history-ready persons':len(ready),
                             'history-ready current-pre1 available':sum(r['pre1_available'] for r in ready),
                             'eligible persons':len(eligible),
                             'eligible with overlapping history windows':sum(r['overlapping_history_window_pairs']>0 for r in eligible)})
    mgates = [r for r in COUNTS[2] if r['section']=='fixed_sample_size_gates']
    lines = ['# Multi-event-history coefficient-blind feasibility audit','',*common,'',
        '## 固定构建规则','',
        '- 每人每种结局只保留最早eligible target。至少两个已完成历史，至少一个历史family与target不同；不要求三个不同family。',
        '- 仅既有isolated四primary事件；复发按新冻结reset规则核验；重复widowhood排除；无financial或retirement。health作为多事件既有primary领域保留，其同域positive-control仍仅secondary。',
        '- 历史截止于target pre1之前：acute需历史event已测量；persistence需历史未censor year2已测量；joint需同一历史episode两者完整。health额外需其下一波持续确认已完成。',
        '- 对历史不要求target的以后信息。目标结局仅检查可用性，绝不参与历史选择、排序或数值表征。',
        '- history-ready集合在目标结局筛选之前选择最早target，用于审计当前状态缺失。eligible集合则在相应目标结局及目标baseline资格通过之后选择最早target；两集合可能为同一人选择不同target。',
        '- 所有符合条件的历史均保留；本轮没有计算其均值、SD、残差或任何响应表征数值。',
        '- 不额外要求每两个历史window分离；明确计数共享观测window（例如一次year2与下一次pre1重合），而不是悄悄删除。它们都不能与目标pre1共享观测。', '',
        '## 主计数','',table(msummary,list(msummary[0])),'',
        '## 固定人数门槛','',table(mgates,['cohort','outcome','statistic','n','category']),'',
        '## 历史和目标分布','',
        '02_multi_event_counts.csv逐cohort、acute/persistence/joint及history-ready/eligible给出：prior_episode_count完整频数、prior_family_count完整频数、至少一个跨域历史的数量、target family和target year频数、current pre1/extra pre2及baseline缺失。',
        '每个频数表均以相应person-target集合为分母；不同结局样本不能相加。joint不自动等于两个样本的person交集，因为同一历史episode必须同时完整，且目标仍按最早合格规则选择。','',
        '## 已确认的数据限制与QA','',
        '- 旧三事件队列要求三个不同family及逐次两年分离，本轮不等同于旧队列，旧样本或确认性结果没有被修改。',
        '- 后期target、至少两次已完成历史、self访谈和censoring共同造成选择；计数不能代表人群发生率或预测功效。',
        '- acute和persistence历史截止时间不同；health资格需要持续确认日期。不能仅按event wave早于target就认定历史已知。',
        '- 当前pre1在完整样本中必然可用；history-ready分母才能显示其实际缺失。',
        '- HRS静态education的实际获得时间不能由该字段保证；报告缺失而不虚构时间。',
        '- 全部history links核验：ID一致、完成日期和资格确认均早于target pre1、至少两个历史、至少一个不同family、唯一target/person/outcome。',
        '- 心理评分、delta或任何效应系数未进入输出资格清单。','']
    (OUT/'02_multi_event_history_feasibility_audit.md').write_text('\n'.join(lines))
    rsummary = []
    group = defaultdict(list)
    for r in recent:
        if r['sample_scope'] in ('primary_pool','health_secondary','financial_secondary','eligible','history_ready','primary_common_targets'):
            group[(r['sample_kind'],r['cohort'],r['sample_scope'],r['outcome'])].append(r)
    for key,q in sorted(group.items()):
        rsummary.append(dict(zip(['sample','cohort','scope','outcome'],key)) | {
            'N':len(q),'current pre1 N':sum(r['pre1_available'] for r in q),
            'additional pre2 N':sum(r['pre2_available'] for r in q),
            'both measures + actual gap N':sum(r['recency_complete'] for r in q)})
    lines = ['# Recency measure availability — counts only','',*common,'',
        '## 观测及时间定义','',
        '- latest pre-event measure固定为既有target pre1；additional earlier measure为既有target pre2，不向更早或未来波次替补缺失。',
        '- UKHLS pre2在pre1前8–18个月内选最近12个月访谈；HRS为t−2。测量值不输出，仅保留合法范围/self-report可用标志。',
        '- 实际gap=(pre1_date−pre2_date).days/30.4375，另记录pre1→event。日期缺失不填名义月份；complete N要求两次心理评分有效、日期有序且wave顺序正确。',
        '- CSV gap列为全部目标的固定区间频数；complete_gap只包含两次测量和实际gap均完整者；event_gap是当前状态距事件访谈的实际间隔频数。', '',
        '## 可用人数','',table(rsummary,list(rsummary[0]) if rsummary else ['sample','cohort','scope','outcome','N']), '',
        '单family子样本及固定门槛计数在03_recency_counts.csv。完整性筛选前后不是同一分母；同一person可出现在不同分析集合中，不能把这些行相加。', '',
        '## 数据限制','',
        'pre2与pre1的访谈间隔不是逆境真实开始时间；HRS两年wave不直接等同UKHLS一年wave。pre2可能属于此前逆境经历的一部分；这里只记录可用性，没有假定它是未受影响的基线。',
        '缺失的额外测量不从更远访谈替补；没有估计recency影响、状态变化、可靠性或预测差异。外部预注册草案给出下一轮同一complete子集的比较规则。','']
    (OUT/'03_recency_measure_availability.md').write_text('\n'.join(lines))
    log(json.dumps({'same_domain':summary,'multi_event':msummary},ensure_ascii=False,indent=2))


def main():
    assert sha(FREEZE) == FREEZE_HASH, 'Draft changed after pre-count freeze.'
    WORK.mkdir(parents=True, exist_ok=True)
    protected = list(EPFILES.values()) + [ROOT/'data_processed/NMH_cross_event_prediction_samples.parquet']
    for folder in ('NMH_confirmatory','NMH_measurement_adjudication'):
        protected += [p for p in (ROOT/'results'/folder).iterdir() if p.is_file()]
    source_manifest = {str(p.relative_to(ROOT)):sha(p) for p in protected}
    source_manifest[str(UK_PANEL.relative_to(ROOT))] = sha(UK_PANEL)
    same, registry, direct, multi, links = [], [], [], [], []
    for cohort in COHORTS:
        log(f'{cohort}: loading episode availability masks and selected state columns')
        e = load_episode_masks(cohort)
        states = load_states(cohort, set(e.person_id))
        e = add_recurrence_flags(cohort,e,states)
        e.to_parquet(WORK/f'{cohort}_episode_eligibility_masks.parquet',index=False)
        log(f'{cohort}: selecting same-domain, common-target and multi-history eligibility; no response values')
        same += same_domain(cohort,e,states)
        reg, both = comparison_targets(cohort,e,states)
        registry += reg
        direct += both
        m, h = multi_event(cohort,e)
        multi += m
        links += h
    recent = recency(same,multi,direct)
    threshold_counts(same,multi,direct,recent)
    for name, rows, key in [
        ('same_domain_selected',same,['cohort','person_id','sample_scope','outcome']),
        ('comparable_target_registry',registry,['cohort','target_episode_id','outcome']),
        ('common_target_selected',direct,['cohort','person_id','outcome']),
        ('multi_event_selected',multi,['cohort','person_id','sample_scope','outcome']),
        ('multi_event_history_links',links,['cohort','target_episode_id','sample_scope','outcome','history_episode_id']),
        ('recency_availability',recent,['sample_kind','cohort','person_id','sample_scope','outcome'])]:
        f = pd.DataFrame(rows)
        if not f.empty:
            assert not f.duplicated(key).any(), name
            assert not any(x.startswith('outcome_') or x.endswith('_z') or x.endswith('_raw') for x in f.columns)
        f.to_parquet(WORK/f'{name}.parquet',index=False)
    for r in same:
        assert before(r['earlier_year2_date'],r['pre1_date']) and before(r['pre1_date'],r['event_date'])
        assert r['earlier_family']==r['family'] and r['earlier_episode_id']!=r['target_episode_id']
    for r in multi:
        assert r['prior_episode_count']>=2 and r['different_family_prior_count']>=1
    for p in protected:
        assert sha(p)==source_manifest[str(p.relative_to(ROOT))], f'Protected input modified: {p}'
    assert sha(FREEZE)==FREEZE_HASH
    META['provenance'] = source_manifest
    META['freeze_sha256'] = FREEZE_HASH
    META['python_version'] = platform.python_version()
    META['pandas_version'] = pd.__version__
    META['audit_script_sha256'] = sha(Path(__file__).resolve())
    META['targeted_HRS_source_metadata'] = {
        str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
        for p in (HRS_CORE, HRS_FAMILY)}
    META['coefficient_blind_input_projection'] = True
    META['protected_inputs_unchanged'] = True
    (WORK/'audit_metadata.json').write_text(json.dumps(META,ensure_ascii=False,indent=2))
    for i in (1,2,3):
        frame = pd.DataFrame(COUNTS[i],columns=FIELDS)
        assert (frame.n>=0).all()
        frame.to_csv(WORK/f'{i:02d}_counts_intermediate.csv',index=False)
    write_reports(same,multi,direct,recent,source_manifest)
    log('Eligibility audit complete. CSV intermediates await artifact-tool export/validation; no models were run.')


if __name__ == '__main__':
    main()
