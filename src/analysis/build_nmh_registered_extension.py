#!/usr/bin/env python3
"""OSF zgeb3/v2 registration record and score-blind sample lock.

Does not import old analysis scripts or old draft protocols.
Only existing episode availability masks, identifiers and dates are used here.
"""
from pathlib import Path
from itertools import combinations
from datetime import datetime, timezone
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/NMH_registered_extension'
ARCHIVE = OUT / 'archive'
LOCK = ARCHIVE / 'sample_lock'
WORK = ROOT / 'outputs/nmh_registered_extension_20260904'
AUDIT = ROOT / 'outputs/nmh_extension_20260904'
PROTOCOL = ROOT / 'OSF_registration_bundle_v2/UPLOAD_TO_OSF/preregistration.pdf'
PROTOCOL_HASH = '8077d0105f3867533ed57ab4f1075f51712ff3cad3e23a8556c53c0d27f7b8c0'
SEED = 20260904
PRIMARY = {'widowhood', 'unemployment', 'caregiving', 'health'}
COHORTS = ('UKHLS', 'HRS')
EXPECTED_REC = {
    ('UKHLS', 'caregiving'): (1990, 1033), ('UKHLS', 'unemployment'): (90, 46),
    ('UKHLS', 'health'): (624, 396), ('UKHLS', 'financial_strain'): (1179, 548),
    ('HRS', 'caregiving'): (164, 126), ('HRS', 'unemployment'): (57, 45), ('HRS', 'health'): (112, 85)}
EXPECTED_MULTI = {('UKHLS', 'nonoverlap'): 1089, ('UKHLS', 'overlap_allowed'): 1603,
                  ('HRS', 'nonoverlap'): 188, ('HRS', 'overlap_allowed'): 270}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_once(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        f.write(text)
    path.chmod(0o444)


def immutable_parquet(frame, filename, key):
    assert not frame.duplicated(key).any(), filename
    path = LOCK / filename
    assert not path.exists(), f'Refusing to replace locked registry: {path}'
    frame.to_parquet(path, index=False)
    check = pd.read_parquet(path)
    pd.testing.assert_frame_equal(frame, check)
    path.chmod(0o444)
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'rows': len(frame), 'unique_key': key}


def table(frame):
    cols = list(frame.columns)
    return '\n'.join(['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---'] * len(cols)) + ' |'] +
                     ['| ' + ' | '.join(str(v).replace('|', '/') for v in row) + ' |' for row in frame.itertuples(index=False, name=None)])


def registration_record():
    assert sha(PROTOCOL) == PROTOCOL_HASH
    # Formal PDF is controlling. Read all pages before any sample operations.
    pages = PdfReader(PROTOCOL).pages
    protocol_text = '\n'.join(p.extract_text() for p in pages)
    assert len(pages) == 5 and len(protocol_text) > 10000
    record_path = ARCHIVE / 'registration_input_hashes.json'
    if record_path.exists():
        record = json.loads(record_path.read_text())
        for name, digest in record['inputs'].items():
            assert sha(ROOT / name) == digest, name
        return record
    OUT.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    source_paths = [PROTOCOL, ROOT / 'OSF_registration_bundle_v2/UPLOAD_TO_OSF/feasibility_summary.pdf']
    for filename in ('01_same_domain_counts.csv', '02_multi_event_counts.csv', '03_recency_counts.csv'):
        registered = ROOT / 'OSF_registration_bundle_v2/UPLOAD_TO_OSF' / filename
        local = ROOT / 'results/NMH_extension_feasibility' / filename
        assert sha(registered) == sha(local), filename
        source_paths += [registered, local]
    for filename in ('01_same_domain_feasibility_audit.md', '02_multi_event_history_feasibility_audit.md', '03_recency_measure_availability.md'):
        source_paths.append(ROOT / 'results/NMH_extension_feasibility' / filename)
    source_paths += [AUDIT / name for name in ('UKHLS_episode_eligibility_masks.parquet', 'HRS_episode_eligibility_masks.parquet',
                    'same_domain_selected.parquet', 'multi_event_selected.parquet', 'multi_event_history_links.parquet',
                    'independent_validation.json', 'audit_metadata.json')]
    source_paths += [ROOT / f'data_processed/NMH_{c}_event_episodes.parquet' for c in COHORTS]
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    # Reading hashes verifies preservation without inspecting any old estimates.
    protected = {str(p.relative_to(ROOT)): sha(p) for folder in ('NMH_confirmatory', 'NMH_measurement_adjudication')
                 for p in (ROOT / 'results' / folder).iterdir() if p.is_file()}
    record = {'registration_url': 'https://osf.io/zgeb3/', 'registration_id': 'zgeb3',
              'registration_timestamp_as_supplied': '2026-09-04 11:41 AM (OSF display; timezone not supplied)',
              'submission_and_exact_v2_file_confirmed_by_user': True,
              'remote_registration_independently_fetched': False,
              'recorded_at_utc': now(), 'controlling_protocol_sha256': PROTOCOL_HASH,
              'inputs': inputs, 'protected_parent_results': protected,
              'v1_and_old_DRAFT_not_used_for_decisions': True}
    write_once(record_path, json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    rows = pd.DataFrame([{'file': name, 'SHA-256': digest} for name, digest in inputs.items()])
    write_once(OUT / '00_registration_and_hash_record.md', '\n'.join([
        '# Registration and input hashes', '',
        '- 唯一正式协议：OSF zgeb3 的 v2 `preregistration.pdf`。',
        '- Registration URL: https://osf.io/zgeb3/',
        '- 注册时间：2026-09-04 11:41 AM，按用户提供的 OSF 显示值原样记录；时区未提供，不擅自换算。',
        '- 用户已确认注册已提交，且该 PDF 即最终冻结版本；本轮未进行网络搜索，也未独立抓取远端页面。',
        f'- 正式 PDF SHA-256：`{PROTOCOL_HASH}`。',
        f'- 本地登记时间（UTC）：{record["recorded_at_utc"]}。',
        '- 已完整阅读正式 PDF 的5页；v1和旧DRAFT均不作为样本、模型、门槛或fallback的依据。',
        '- 上传包内三份 coefficient-blind CSV 与项目原审计CSV逐字节哈希一致。',
        '- 先完成登记与只读资格重建，再进行估计；不得使用已有全样本标准化值或旧响应预测列。',
        '- 资格清单是已注册可行性数据的复用，不把生成该清单时的旧方案作为当前决策依据。', '',
        '## Input SHA-256', '', table(rows), '',
        '逐文件机器可读记录及父项目只读结果保护哈希见 archive/registration_input_hashes.json。', '']))
    return record


def before(a, b):
    return pd.notna(a) and pd.notna(b) and a < b


def target_row(sample_id, role, analysis, variant, outcome, r):
    fields = ['cohort', 'person_id', 'family', 'event_wave', 'event_date', 'event_year',
              'pre1_wave', 'pre2_wave', 'year2_wave', 'pre1_date', 'pre2_date', 'year2_date',
              'pre1_available', 'pre2_available', 'event_available', 'year2_available',
              'age_known', 'sex_known', 'education_known', 'baseline_usable']
    return dict(sample_id=sample_id, role=role, analysis=analysis, variant=variant, outcome=outcome,
                target_episode_id=r['episode_id'], **{k: r[k] for k in fields})


def history_row(target, history, order):
    return {'sample_id': target['sample_id'], 'cohort': target['cohort'], 'person_id': target['person_id'],
            'target_episode_id': target['target_episode_id'], 'history_episode_id': history['episode_id'],
            'history_order': order, 'history_family': history['family'],
            'history_event_date': history['event_date'], 'history_pre1_wave': history['pre1_wave'],
            'history_event_wave': history['event_wave'], 'history_year2_date': history['year2_date'],
            'history_qualification_date': history['qualification_date'],
            'history_age_known': history['age_known'], 'information_cutoff_date': target['pre1_date']}


def rebuild():
    record = registration_record()
    if (LOCK / 'manifest.json').exists():
        manifest = json.loads((LOCK / 'manifest.json').read_text())
        for f in manifest['files']:
            assert sha(ROOT / f['path']) == f['sha256']
        print('Existing locked registry verified; not overwritten.', flush=True)
        return
    LOCK.mkdir(parents=True, exist_ok=True)
    registries, histories = [], []
    old_same = pd.read_parquet(AUDIT / 'same_domain_selected.parquet')
    old_multi = pd.read_parquet(AUDIT / 'multi_event_selected.parquet')
    old_links = pd.read_parquet(AUDIT / 'multi_event_history_links.parquet')
    for cohort in COHORTS:
        e = pd.read_parquet(AUDIT / f'{cohort}_episode_eligibility_masks.parquet')
        assert not e.duplicated(['person_id', 'episode_id']).any()
        assert not any(c.endswith(('_raw', '_z')) or c.startswith('outcome_') for c in e.columns)
        assert (e.structural_eligible == (e.allowed_label & e.index_valid & e.recurrence_valid & e.confirmation_valid)).all()
        eligible = e.loc[e.structural_eligible].copy()
        assert not eligible.compound_primary.any()
        rec = {}
        multi = {}
        for person, group in eligible.groupby('person_id', sort=True):
            rows = group.sort_values(['event_date', 'event_wave', 'episode_id']).to_dict('records')
            for a, b in combinations(rows, 2):
                family = b['family']
                if a['family'] != family or (cohort, family) not in EXPECTED_REC:
                    continue
                if not (a['joint_complete'] and b['baseline_usable'] and
                        before(a['year2_date'], b['pre1_date']) and a['year2_wave'] < b['pre1_wave'] and
                        before(a['qualification_date'], b['pre1_date'])):
                    continue
                for outcome, flag in [('acute', 'acute_complete'), ('persistence', 'joint_complete')]:
                    if b[flag]:
                        rec.setdefault((int(person), family, outcome), (a, b))
            primary_rows = [r for r in rows if r['family'] in PRIMARY]
            for i, target in enumerate(primary_rows):
                if not target['acute_complete'] or not target['baseline_usable']:
                    continue
                prior = [h for h in primary_rows[:i] if h['acute_complete'] and
                         before(h['event_date'], target['pre1_date']) and h['event_wave'] < target['pre1_wave'] and
                         before(h['qualification_date'], target['pre1_date'])]
                if len(prior) >= 2 and any(h['family'] != target['family'] for h in prior):
                    multi.setdefault(int(person), (target, prior))
        for (person, family, outcome), (a, b) in rec.items():
            scope = {'health': 'health_secondary', 'financial_strain': 'financial_secondary'}.get(family, family)
            old = old_same.loc[(old_same.cohort == cohort) & (old_same.person_id == person) &
                               (old_same.sample_scope == scope) & (old_same.outcome == outcome)]
            assert len(old) == 1 and old.iloc[0].earlier_episode_id == a['episode_id'] and old.iloc[0].target_episode_id == b['episode_id']
            sid = f'recurrence_{cohort}_{family}_{outcome}'
            role = 'primary' if cohort == 'UKHLS' and family == 'caregiving' else 'secondary'
            target = target_row(sid, role, 'recurrence', 'same_domain', outcome, b)
            target.update(prior_episode_count=1, prior_family_count=1, historical_observation_overlap_pairs=0)
            registries.append(target)
            histories.append(history_row(target, a, 1))
        for person, (target, prior) in multi.items():
            old = old_multi.loc[(old_multi.cohort == cohort) & (old_multi.person_id == person) &
                                (old_multi.sample_scope == 'eligible') & (old_multi.outcome == 'acute')]
            assert len(old) == 1 and old.iloc[0].target_episode_id == target['episode_id']
            old_members = old_links.loc[(old_links.cohort == cohort) & (old_links.person_id == person) &
                                       (old_links.sample_scope == 'eligible') & (old_links.outcome == 'acute')]
            assert set(old_members.history_episode_id) == {h['episode_id'] for h in prior}
            observation_overlap = sum(bool({a['pre1_wave'], a['event_wave']} & {b['pre1_wave'], b['event_wave']})
                                      for a, b in combinations(prior, 2))
            assert observation_overlap == old.iloc[0].overlapping_history_window_pairs
            for variant in ('nonoverlap', 'overlap_allowed'):
                if variant == 'nonoverlap' and observation_overlap > 0:
                    continue
                sid = f'multi_{cohort}_acute_{variant}'
                role = 'primary' if cohort == 'UKHLS' and variant == 'nonoverlap' else 'secondary'
                t = target_row(sid, role, 'multi_event', variant, 'acute', target)
                t.update(prior_episode_count=len(prior), prior_family_count=len({h['family'] for h in prior}),
                         historical_observation_overlap_pairs=observation_overlap)
                registries.append(t)
                histories += [history_row(t, h, order) for order, h in enumerate(prior, 1)]
        print(f'{cohort}: v2 recurrence and multi-event selections rebuilt from masks.', flush=True)

    targets = pd.DataFrame(registries).sort_values(['sample_id', 'person_id']).reset_index(drop=True)
    links = pd.DataFrame(histories).sort_values(['sample_id', 'person_id', 'history_order']).reset_index(drop=True)
    counts = []
    fold_rows = []
    for sid, group in targets.groupby('sample_id', sort=True):
        first = group.iloc[0]
        if first.analysis == 'recurrence':
            expected = EXPECTED_REC[(first.cohort, first.family)][0 if first.outcome == 'acute' else 1]
        else:
            expected = EXPECTED_MULTI[(first.cohort, first.variant)]
        assert len(group) == expected and group.person_id.nunique() == expected, sid
        # Ten folds are formed independently for each registered analysis sample.
        # All representations and nested models inherit this fixed assignment.
        people = np.sort(group.person_id.to_numpy())
        shuffled = np.random.RandomState(SEED).permutation(people)
        for fold, ids in enumerate(np.array_split(shuffled, 10)):
            for pid in ids:
                fold_rows.append({'sample_id': sid, 'cohort': first.cohort, 'person_id': int(pid), 'outer_fold': fold,
                                  'seed': SEED, 'algorithm': 'sorted_persons_RandomState_permutation_array_split_10'})
        current = dict(sample_id=sid, cohort=first.cohort, analysis=first.analysis, role=first.role,
                       variant=first.variant, outcome=first.outcome,
                       event_family=first.family if first.analysis == 'recurrence' else 'multiple',
                       persons=len(group), target_rows=len(group), history_rows=int(links.sample_id.eq(sid).sum()),
                       preregistration_expected_persons=expected, rows_match_registered_count=True,
                       current_pre1_complete=int(group.pre1_available.sum()),
                       pre2_complete=int(group.pre2_available.sum()),
                       age_known=int(group.age_known.sum()), sex_known=int(group.sex_known.sum()),
                       education_known=int(group.education_known.sum()),
                       participants_with_shared_history_observations=int(group.historical_observation_overlap_pairs.gt(0).sum()))
        counts.append(current)
    folds = pd.DataFrame(fold_rows).sort_values(['sample_id', 'person_id']).reset_index(drop=True)
    count_frame = pd.DataFrame(counts)
    files = [immutable_parquet(targets, 'target_registry.parquet', ['sample_id', 'person_id']),
             immutable_parquet(links, 'history_registry.parquet', ['sample_id', 'person_id', 'history_episode_id']),
             immutable_parquet(folds, 'grouped_cv_assignments.parquet', ['sample_id', 'person_id']),
             immutable_parquet(count_frame, 'sample_counts.parquet', ['sample_id'])]
    manifest = {'locked_at_utc': now(), 'controlling_protocol_sha256': PROTOCOL_HASH,
                'builder_sha256': sha(Path(__file__)), 'input_hash_record_sha256': sha(ARCHIVE / 'registration_input_hashes.json'),
                'all_registries_constructed_without_response_values': True,
                'historical_overlap_verified_by_shared_person_wave_observation_ids': True,
                'no_reselection_after_historical_overlap_exclusion': True,
                'cross_validation_seed': SEED, 'files': files}
    write_once(LOCK / 'manifest.json', json.dumps(manifest, indent=2) + '\n')
    count_frame.to_csv(WORK / '01_sample_counts_intermediate.csv', index=False)
    report_cols = ['cohort', 'event_family', 'outcome', 'variant', 'role', 'persons', 'history_rows', 'pre2_complete']
    write_once(OUT / '01_sample_rebuild_and_QA.md', '\n'.join([
        '# Registered sample rebuild and QA', '',
        'Controlling protocol: OSF zgeb3, confirmed v2 PDF. No v1/DRAFT rules or fallback thresholds were used.', '',
        '## Locked eligibility samples', '', table(count_frame[report_cols]), '',
        '## Selection and preservation checks', '',
        '- Recurrence: one earliest valid earlier episode and first later same-family recurrence per person and outcome; earlier acute and uncensored two-year observations complete; later required outcome complete.',
        '- Earlier year2 strictly precedes later pre1 by date and wave. Distinct episodes; compound primary transitions excluded. Registered recurrence/reset masks retained.',
        '- Caregiving primary uses the family-specific sample (UKHLS 1,990 / 1,033), not the unemployment-plus-caregiving pool.',
        '- Multi-event: earliest eligible target, all completed prior acute histories, at least one prior family different from target. Qualification and outcome observations precede target pre1.',
        '- Non-overlap filter applied after earliest target selection. Shared mental-health person-wave IDs independently reproduce the audited overlap flags. No alternative target is substituted.',
        '- UKHLS 1,603 minus 514 = 1,089 non-overlap participants; HRS 270 minus 82 = 188. Broader samples are sensitivity samples only.',
        '- Every rebuilt target and history membership matches the registered coefficient-blind source list; all expected sample counts match.',
        '- Each sample has a unique (sample_id, cohort, person_id) target key. All histories and target records inherit their person fold; comparisons within a sample share folds.',
        '- Pre2 is the existing fixed additional observation. Its missingness is reported and no earlier replacement is searched.',
        '- Unknown education remains an explicit existing category. Education-known counts are not misreported as complete observed education.',
        '- No psychological score, response coefficient or predictive performance was read to build these registries.',
        '- Lock is write-once at the script level, filesystem read-only, and SHA-256 checked before fitting. It is not an external archival/WORM guarantee.',
        '- Temporal assignments and model-specific complete-case registries will be locked before estimation after the unspecified implementation details are resolved.', '',
        '## Archives', '',
        'archive/sample_lock contains target/history registries, deterministic grouped-CV assignments, sample counts and their SHA-256 manifest.',
        'Raw person identifiers and person-level outputs remain local. They are not authorized for OSF upload by this task.', '']))
    environment = {'recorded_at_utc': now(), 'python': sys.version, 'platform': platform.platform(),
                   'numpy': np.__version__, 'pandas': pd.__version__,
                   'builder_sha256': sha(Path(__file__)), 'models_run': False}
    write_once(ARCHIVE / 'sample_build_environment.json', json.dumps(environment, indent=2) + '\n')
    for name, original in {**record['inputs'], **record['protected_parent_results']}.items():
        assert sha(ROOT / name) == original, name
    print(count_frame[report_cols].to_string(index=False), flush=True)
    print('Score-blind sample lock complete. No models run.', flush=True)


if __name__ == '__main__':
    rebuild()
