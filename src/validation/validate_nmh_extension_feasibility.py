#!/usr/bin/env python3
"""Independent count/eligibility QA. Reads saved masks, not mental-health values."""
from pathlib import Path
from itertools import combinations
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'outputs/nmh_extension_20260904'
OUT = ROOT / 'results/NMH_extension_feasibility'
PRIMARY = {'widowhood', 'unemployment', 'caregiving', 'health'}


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def precedes(a, b):
    return pd.notna(a) and pd.notna(b) and a < b


def main():
    metadata = json.loads((WORK / 'audit_metadata.json').read_text())
    assert digest(OUT / '04_external_preregistration_freeze_DRAFT.md') == metadata['freeze_sha256']
    for name, original in metadata['provenance'].items():
        assert digest(ROOT / name) == original, name
    e = pd.concat([pd.read_parquet(WORK / f'{cohort}_episode_eligibility_masks.parquet') for cohort in ('UKHLS', 'HRS')])
    assert not e.duplicated(['cohort', 'episode_id']).any()
    assert not any(c.endswith(('_raw', '_z')) or c.startswith('outcome_') for c in e.columns)
    lookup = {(r.cohort, r.episode_id): r._asdict() for r in e.itertuples(index=False)}
    same = pd.read_parquet(WORK / 'same_domain_selected.parquet')
    registry = pd.read_parquet(WORK / 'comparable_target_registry.parquet')
    direct = pd.read_parquet(WORK / 'common_target_selected.parquet')
    multi = pd.read_parquet(WORK / 'multi_event_selected.parquet')
    links = pd.read_parquet(WORK / 'multi_event_history_links.parquet')
    recent = pd.read_parquet(WORK / 'recency_availability.parquet')
    for frame, key in [(same, ['cohort', 'person_id', 'sample_scope', 'outcome']),
                       (registry, ['cohort', 'target_episode_id', 'outcome']),
                       (direct, ['cohort', 'person_id', 'outcome']),
                       (multi, ['cohort', 'person_id', 'sample_scope', 'outcome']),
                       (links, ['cohort', 'target_episode_id', 'sample_scope', 'outcome', 'history_episode_id']),
                       (recent, ['sample_kind', 'cohort', 'person_id', 'sample_scope', 'outcome'])]:
        assert not frame.duplicated(key).any()
        assert not any(c.endswith(('_raw', '_z')) or c.startswith('outcome_') for c in frame.columns)

    # Reconstruct one-person selections from saved binary masks using an
    # independent explicit loop. No scores or response variables are loaded.
    expected_same = {}
    expected_multi = {}
    expected_links = {}
    for (cohort, pid), g in e.loc[e.structural_eligible].groupby(['cohort', 'person_id']):
        records = g.sort_values(['event_date', 'event_wave', 'episode_id']).to_dict('records')
        scopes = {'primary_pool': {'unemployment', 'caregiving'}, 'unemployment': {'unemployment'},
                  'caregiving': {'caregiving'}, 'health_secondary': {'health'}}
        if cohort == 'UKHLS':
            scopes['financial_secondary'] = {'financial_strain'}
        for a, b in combinations(records, 2):
            if a['family'] != b['family'] or a['family'] == 'widowhood':
                continue
            if not (precedes(a['year2_date'], b['pre1_date']) and a['year2_wave'] < b['pre1_wave']):
                continue
            # The recurrence mask already checks the immediate previous onset
            # and a normal pre-onset wave after that onset.
            for scope, families in scopes.items():
                if b['family'] not in families:
                    continue
                for outcome in ('onset_timing', 'acute', 'persistence'):
                    okay = (outcome == 'onset_timing' or (a['joint_complete'] and b['baseline_usable']
                            and b['acute_complete' if outcome == 'acute' else 'joint_complete']))
                    if okay:
                        expected_same.setdefault((cohort, int(pid), scope, outcome), (a['episode_id'], b['episode_id']))
        histories = [r for r in records if r['family'] in PRIMARY]
        for j, target in enumerate(histories):
            for outcome in ('acute', 'persistence', 'joint'):
                hist = []
                for earlier in histories[:j]:
                    end = 'event' if outcome == 'acute' else 'year2'
                    if (earlier[outcome + '_complete'] and precedes(earlier[end + '_date'], target['pre1_date'])
                            and earlier[end + '_wave'] < target['pre1_wave']
                            and precedes(earlier['qualification_date'], target['pre1_date'])):
                        hist.append(earlier)
                if len(hist) < 2 or not any(h['family'] != target['family'] for h in hist):
                    continue
                for scope in ('history_ready', 'eligible'):
                    if scope == 'eligible' and not (target[outcome + '_complete'] and target['baseline_usable']):
                        continue
                    key = (cohort, int(pid), scope, outcome)
                    if key not in expected_multi:
                        expected_multi[key] = target['episode_id']
                        expected_links[key] = {h['episode_id'] for h in hist}
    actual_same = {(r.cohort, int(r.person_id), r.sample_scope, r.outcome): (r.earlier_episode_id, r.target_episode_id)
                   for r in same.itertuples(index=False)}
    actual_multi = {(r.cohort, int(r.person_id), r.sample_scope, r.outcome): r.target_episode_id
                   for r in multi.itertuples(index=False)}
    assert actual_same == expected_same, 'Same-domain earliest pair reconstruction differs.'
    assert actual_multi == expected_multi, 'Multi-event earliest target reconstruction differs.'
    for key, group in links.groupby(['cohort', 'person_id', 'sample_scope', 'outcome']):
        assert set(group.history_episode_id) == expected_links[key]
        assert group.history_completion_date.lt(group.information_cutoff_date).all()
        assert group.history_qualification_date.lt(group.information_cutoff_date).all()
        target = lookup[(key[0], actual_multi[key])]
        for h in group.itertuples(index=False):
            assert lookup[(h.cohort, h.history_episode_id)]['person_id'] == target['person_id']
    for r in same.itertuples(index=False):
        a, b = lookup[(r.cohort, r.earlier_episode_id)], lookup[(r.cohort, r.target_episode_id)]
        assert a['person_id'] == b['person_id'] and a['family'] == b['family']
        if r.outcome != 'onset_timing':
            assert a['joint_complete'] and not a['year2_censored_by_later_primary']
            assert precedes(a['qualification_date'], b['pre1_date'])
        if r.outcome == 'persistence':
            assert b['joint_complete'] and not b['year2_censored_by_later_primary']
    for r in registry.itertuples(index=False):
        target = lookup[(r.cohort, r.target_episode_id)]
        for arm in ('same', 'cross'):
            if not getattr(r, arm + '_available'):
                continue
            earlier = lookup[(r.cohort, getattr(r, arm + '_earlier_episode_id'))]
            assert earlier['person_id'] == target['person_id'] and earlier['joint_complete']
            assert precedes(earlier['year2_date'], target['pre1_date'])
            assert precedes(earlier['qualification_date'], target['pre1_date'])
            assert (earlier['family'] == target['family']) == (arm == 'same')
        assert target['acute_complete'] and target['baseline_usable']
        if r.outcome == 'persistence':
            assert target['joint_complete']
    expected_direct = registry.loc[registry.family.isin(['unemployment', 'caregiving']) & registry.same_available & registry.cross_available]
    expected_direct = expected_direct.sort_values(['event_date', 'event_wave', 'target_episode_id']).drop_duplicates(['cohort', 'person_id', 'outcome'])
    assert set(zip(expected_direct.cohort, expected_direct.person_id, expected_direct.outcome, expected_direct.target_episode_id)) == set(zip(direct.cohort, direct.person_id, direct.outcome, direct.target_episode_id))

    for r in recent.itertuples(index=False):
        complete = bool(r.pre1_available and r.pre2_available and precedes(r.pre2_date, r.pre1_date) and r.pre2_wave < r.pre1_wave)
        assert r.recency_complete == complete
        if pd.notna(r.pre2_date) and pd.notna(r.pre1_date):
            assert r.pre2_to_pre1_months == (r.pre1_date - r.pre2_date).days / 30.4375

    csv_rows = {}
    for i, filename in enumerate(('01_same_domain_counts.csv', '02_multi_event_counts.csv', '03_recency_counts.csv'), 1):
        path = OUT / filename
        if not path.exists():
            path = WORK / f'{i:02d}_counts_intermediate.csv'
        counts = pd.read_csv(path, keep_default_na=False)
        assert not counts.duplicated(['section', 'cohort', 'sample_scope', 'outcome', 'event_family', 'statistic', 'category']).any()
        assert counts.n.ge(0).all()
        for r in counts.itertuples(index=False):
            if r.denominator != '':
                assert 0 <= r.n <= float(r.denominator)
            if i in (1, 2) and r.section in ('same_domain', 'multi_event') and r.statistic == 'unique_persons':
                frame = same if i == 1 else multi
                group = frame.loc[frame.cohort.eq(r.cohort) & frame.sample_scope.eq(r.sample_scope) & frame.outcome.eq(r.outcome)]
                assert r.n == group.person_id.nunique()
        distributions = counts.loc[counts.section.str.endswith(('_calendar', '_target_family', '_interval')) | counts.section.eq('history_distribution')]
        for _, g in distributions.groupby(['section', 'cohort', 'sample_scope', 'outcome', 'event_family', 'statistic']):
            assert g.n.sum() == float(g.denominator.iloc[0]), 'Frequency totals disagree.'
        csv_rows[filename] = len(counts)
    result = {'passed': True, 'original_sources_unchanged': True, 'pre_count_draft_unchanged': True,
              'same_domain_earliest_pair_reconstructed': True, 'multi_event_earliest_target_reconstructed': True,
              'all_history_memberships_reconstructed': True, 'same_cross_target_timing_verified': True,
              'person_keys_and_count_denominators_verified': True, 'no_psychological_scores_read': True,
              'csv_rows': csv_rows}
    (WORK / 'independent_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
