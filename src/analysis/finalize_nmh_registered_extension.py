from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import sys
import numpy as np
import pandas as pd
import pyarrow
import pypdf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/NMH_registered_extension'
ARC = OUT / 'archive'
WORK = ROOT / 'outputs/nmh_registered_extension_20260904'


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    manifest_path = ARC / 'final_artifact_manifest.json'
    assert not manifest_path.exists(), 'Final manifest already exists; do not overwrite it.'
    required = ['00_registration_and_hash_record.md', '00_pre_estimation_implementation_note.md',
                '01_sample_rebuild_and_QA.md', '01_sample_counts.csv',
                '02_within_domain_recurrence_results.md', '02_within_domain_recurrence_results.csv',
                '03_multi_event_acute_history_results.md', '03_multi_event_acute_history_results.csv',
                '04_recency_benchmark_results.md', '04_recency_benchmark_results.csv',
                '05_sensitivity_results.md', '05_sensitivity_results.csv', '06_registered_extension_final_summary.md']
    assert all((OUT / f).exists() and (OUT / f).stat().st_size for f in required)
    qa = json.loads((ARC / 'independent_analysis_validation.json').read_text())
    assert qa['passed']
    source = json.loads((ARC / 'registration_input_hashes.json').read_text())
    for f, h in {**source['inputs'], **source['protected_parent_results']}.items():
        assert digest(ROOT / f) == h, f
    note = json.loads((ARC / 'pre_estimation_implementation_hash.json').read_text())
    assert digest(ROOT / note['note_path']) == note['note_sha256']
    results = pd.read_parquet(ARC / 'all_performance_results.parquet')
    export = json.loads((WORK / 'result_csv_export_validation.json').read_text())
    for record in export:
        name = record['file']
        frame = pd.read_csv(OUT / name)
        if name.startswith('02'):
            reference = results.loc[results.analysis.eq('recurrence') & results.benchmark.eq('main')]
        elif name.startswith('03'):
            reference = results.loc[results.analysis.eq('multi_event') & results.benchmark.eq('main')]
        elif name.startswith('04'):
            reference = results.loc[results.benchmark.eq('recency')]
        else:
            reference = results.loc[results.analysis_role.eq('secondary')]
        key = ['run_id', 'validation', 'row_type', 'term', 'metric']
        a = frame.sort_values(key).reset_index(drop=True)
        b = reference.sort_values(key).reset_index(drop=True)
        assert len(a) == len(b) == record['rows']
        assert not a.duplicated(key).any()
        for col in ('estimate', 'ci_lower', 'ci_upper'):
            assert np.allclose(a[col], b[col], rtol=1e-13, atol=1e-13)
        for col in ('n_evaluated', 'bootstrap_successful', 'analysis_role', 'status'):
            assert a[col].equals(b[col])
        assert record['roundtrip_identical']
    shutil.copy2(WORK / 'result_csv_export_validation.json', ARC / 'csv_export_validation.json')
    software_path = ARC / 'software_environment.json'
    software = json.loads(software_path.read_text())
    software.update({'python_executable': sys.executable, 'pyarrow_version': pyarrow.__version__,
                     'pypdf_version': pypdf.__version__, 'runtime_bundle': '26.903.11726',
                     'csv_authoring': '@oai/artifact-tool with verified CSV serialization from authored range values',
                     'output_csv_values_match_analysis_parquet': True})
    software_path.write_text(json.dumps(software, indent=2) + '\n')
    code = ARC / 'code'
    code.mkdir(exist_ok=False)
    files = [ROOT / 'scripts' / name for name in ('build_nmh_registered_extension.py', 'freeze_nmh_registered_implementation.py',
        'run_nmh_registered_extension.py', 'validate_nmh_registered_extension.py', 'report_nmh_registered_extension.py',
        'finalize_nmh_registered_extension.py')]
    files.append(WORK / 'export_registered_csvs.mjs')
    code_hashes = {}
    for f in files:
        destination = code / f.name
        shutil.copy2(f, destination)
        destination.chmod(0o444)
        code_hashes[str(f.relative_to(ROOT))] = digest(f)
        assert digest(destination) == digest(f)
    data_files = sorted(p for p in OUT.rglob('*') if p.is_file() and p != manifest_path)
    manifest = {'finalized_at_utc': datetime.now(timezone.utc).isoformat(),
                'registered_protocol_sha256': source['controlling_protocol_sha256'],
                'pre_estimation_note_sha256': note['note_sha256'], 'code_hashes': code_hashes,
                'independent_validation_passed': True, 'csv_export_validation_passed': True,
                'original_sources_and_parent_results_unchanged': True,
                'file_count_excluding_this_manifest': len(data_files),
                'files': [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in data_files]}
    with manifest_path.open('x') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')
    manifest_path.chmod(0o444)
    print(json.dumps({'completed': True, 'required_main_files': len(required), 'registered_result_rows': len(results),
                      'manifest_files': len(data_files), 'note_sha256': note['note_sha256'],
                      'finalized_at_utc': manifest['finalized_at_utc']}, indent=2))


if __name__ == '__main__':
    main()
