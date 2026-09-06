#!/usr/bin/env python3
"""Format all locked registered results; no new model fitting or result selection."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/NMH_registered_extension'
ARC = OUT / 'archive'
WORK = ROOT / 'outputs/nmh_registered_extension_20260904'


def md(frame):
    def fmt(x):
        if isinstance(x, (float, np.floating)):
            return 'not estimable' if not np.isfinite(x) else f'{x:.6f}'
        if pd.isna(x):
            return ''
        if isinstance(x, pd.Timestamp):
            return x.strftime('%Y-%m-%d')
        return str(x).replace('|', '/')
    return '\n'.join(['| ' + ' | '.join(frame.columns) + ' |', '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'] +
                     ['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)])


def increments(frame):
    q = frame.loc[frame.row_type.eq('increment') & frame.metric.eq('predictive_r2')].copy()
    cols = ['cohort', 'event_family', 'outcome', 'variant', 'representation', 'formulation', 'validation',
            'term', 'n_evaluated', 'estimate', 'ci_lower', 'ci_upper']
    return q[cols].rename(columns={'estimate': 'delta_predictive_R2', 'ci_lower': 'CI_lower', 'ci_upper': 'CI_upper'})


def performance(frame):
    q = frame.loc[frame.row_type.eq('model')]
    index = ['cohort', 'outcome', 'formulation', 'validation', 'term', 'n_evaluated']
    return q.pivot(index=index, columns='metric', values='estimate').reset_index().rename_axis(None, axis=1)


def main():
    validation = json.loads((ARC / 'independent_analysis_validation.json').read_text())
    assert validation['passed']
    r = pd.read_parquet(ARC / 'all_performance_results.parquet')
    cases = pd.read_parquet(ARC / 'validation_lock/case_registry.parquet')
    runs = pd.read_parquet(ARC / 'validation_lock/run_registry.parquet')
    counts = pd.read_parquet(ARC / 'sample_lock/sample_counts.parquet')
    fit = pd.read_parquet(ARC / 'all_fit_records.parquet')
    note = json.loads((ARC / 'pre_estimation_implementation_hash.json').read_text())
    starting = json.loads((ARC / 'estimation_started.json').read_text())
    for name, frame in {
        '02_within_domain_recurrence_results': r.loc[r.analysis.eq('recurrence') & r.benchmark.eq('main')],
        '03_multi_event_acute_history_results': r.loc[r.analysis.eq('multi_event') & r.benchmark.eq('main')],
        '04_recency_benchmark_results': r.loc[r.benchmark.eq('recency')],
        '05_sensitivity_results': r.loc[r.analysis_role.eq('secondary')],
    }.items():
        frame = frame.sort_values(['cohort', 'sample_id', 'benchmark', 'representation', 'scale', 'formulation',
                                   'validation', 'row_type', 'term', 'metric']).reset_index(drop=True)
        frame.to_csv(WORK / f'{name}_intermediate.csv', index=False)
    preamble = [
        '唯一正式依据：OSF zgeb3 的v2协议；另执行用户批准、估计前封存的三项技术性操作化。v1和旧DRAFT未用于分析决策。',
        f'实施说明封存时间（UTC）：{note["sealed_at_utc"]}；SHA-256：`{note["note_sha256"]}`。',
        f'本轮估计开始时间（UTC）：{starting["first_fit_start_utc"]}。',
        '所有CI为500次成功的person-cluster paired bootstrap百分位95%区间。样本内每人唯一target；同次重采样在全部比较模型中一致。',
        'CSV保留所有注册结果的完整数值和角色。MD按预定分析层级展示，不按结果方向筛选。MD小数显示六位，精确值见CSV。',
    ]
    methods = [
        '## 共同计算规则', '',
        '- OLS，无正则化、变量搜索、交互扩展或调参。context：target family、pre1年龄、sex、education、target year、当时已观察到的primary-event数量；多事件模型另含合格历史episode数量。',
        '- “已观察到”按相应pre1信息截止时间计算。持续health事件须其后续确认已完成才计入已知历史事件数量，避免把后来才确认的onset标签带入历史context。',
        '- 每个完整病例集合沿用锁定的person 10-fold分组（seed 20260904）；各嵌套模型共用训练和测试人员。subset不为某个模型重新抽样。',
        '- 每个split只用训练target pre1的均值和样本SD（ddof=1）；同一参数应用于target、历史及pre2。先标准化量表值，再构造响应。raw-scale分析使用原分。',
        '- 时间cutoff为排序后ceil(0.70×N)位置的实际日期；同日全部归训练集，不插值、不移动切点。各case人数和日期见01样本QA补充表。',
        '- Predictive R²=1−SSE/SST_test。交叉验证指标汇总全部OOF人，而不是简单平均10个fold的R²；OOF标准化值使用各自训练fold的量尺。',
        '- RMSE/MAE/calibration intercept使用所报告的量尺。校准以held-out y对point prediction拟合截距与斜率，只作评价，不回填预测。',
        '- Mean log predictive density为plug-in Gaussian：每模型自身RSS_train/n_train，不使用测试残差方差，不加参数不确定性修正。',
        '- 增量统一为“增强模型−基线模型”：R²/MLPD为正对应增加；RMSE/MAE为负对应误差减少。校准截距/斜率的差值本身没有单一优劣方向。',
        '- Bootstrap重采样已保存的held-out person prediction bundles，不重拟合训练模型。区间不涵盖重新分折和全部训练不确定性。',
        '- Bootstrap随机流按估计前代码中的seed、run_id和validation确定。同一run内模型严格配对；不同scale/formulation的run未共用随机抽样索引，因此即使点指标相同，有限500次重采样的CI也可略有差异。没有在看到结果后重抽样来对齐区间。',
        '- 不插补心理结局，不winsorize，不改变事件或删失规则；education unknown作为既有显式类别保留。数学不可估值应标记而非换模型；本次所有拟合与指标均可计算。',
        '- 线性代数仅作训练内数值条件化，rank及零/非有限预测密度状态被记录。未引用旧DRAFT中的人数fallback。', '',
    ]
    rec = r.loc[r.analysis.eq('recurrence') & r.benchmark.eq('main')]
    primary_rec = rec.loc[rec.cohort.eq('UKHLS') & rec.event_family.eq('caregiving') & rec.scale.eq('standardized')]
    (OUT / '02_within_domain_recurrence_results.md').write_text('\n'.join([
        '# Within-domain recurrence results', '', *preamble, '',
        '## 注册模型', '',
        '- R0=context；R1=R0+earlier same-domain response；R2=R0+current pre1；R3=R2+earlier response。',
        '- 分别预测later acute与two-year persistence。前者历史变量为earlier acute；后者为earlier persistence。',
        '- UKHLS caregiving为primary。HRS parental caregiving、两cohort unemployment/health、UKHLS financial strain均为secondary。',
        '- ANCOVA的目标是对应later event量表水平（acute）或year2量表水平（persistence），所有模型强制包含current pre1。因此ANCOVA中R0=R2、R1=R3，两个历史增量数值重复，不视为独立证据。', '',
        '## UKHLS caregiving：标准化结果与ANCOVA配套结果', '', md(increments(primary_rec)), '',
        '## UKHLS caregiving：模型层面全部指标', '', md(performance(primary_rec)), '',
        '## 所有注册复发family：标准化增量', '', md(increments(rec.loc[rec.scale.eq('standardized')])), '',
        'Raw-scale结果、全部RMSE/MAE/MLPD/校准增量和区间完整保存在02 CSV，并收录到05次分析CSV；未删除负增量。', '',
        *methods,
    ]))
    multi = r.loc[r.analysis.eq('multi_event') & r.benchmark.eq('main')]
    main_multi = multi.loc[multi.variant.eq('nonoverlap') & multi.representation.eq('mean') & multi.scale.eq('standardized')]
    (OUT / '03_multi_event_acute_history_results.md').write_text('\n'.join([
        '# Multi-event acute-history results', '', *preamble, '',
        '## 注册模型与样本地位', '',
        '- M0=context；M1=M0+mean prior acute response；M2=M0+current pre1；M3=M2+mean prior acute response。',
        '- 均值包含该target之前全部已完成的合格历史，不挑选反应最大的两次。至少两个历史、至少一个与target不同family。',
        '- UKHLS non-overlap N=1,089为primary；HRS non-overlap N=188为secondary。UKHLS N=1,603/HRS N=270的overlap-allowed集合仅作注册敏感性。',
        '- 在最早合格target确定后排除历史共享观测者，没有另选target。所有历史及持续性确认在target pre1前已观察。',
        '- Secondary representations：most recent、training-fold residualized mean、between-history SD（ddof=1）。均替代history特征，不新增其他特征组合。',
        '- 残差化训练模型只使用外层训练persons的合格历史，调整prior family、prior pre1、age、year和当时已观察到的先前primary-event数量；未使用held-out target或held-out history拟合残差化参数。',
        '- ANCOVA始终包含pre1，M0=M2、M1=M3。没有运行multi-event persistence/joint目标或共同target的same-versus-cross头对头分析。', '',
        '## Non-overlap mean history：标准化主表示与ANCOVA', '', md(increments(main_multi)), '',
        '## Non-overlap mean history：全部模型指标', '', md(performance(main_multi)), '',
        '## 预定表示与重叠敏感性：标准化change-score增量', '',
        md(increments(multi.loc[multi.scale.eq('standardized') & multi.formulation.eq('change_score')])), '',
        '上述次分析保持secondary角色；完整raw-scale、ANCOVA及全部评价指标保存在03/05 CSV。', '', *methods,
    ]))
    recency = r.loc[r.benchmark.eq('recency')]
    (OUT / '04_recency_benchmark_results.md').write_text('\n'.join([
        '# Recency benchmark results', '', *preamble, '',
        '## 完全相同的complete-case比较', '',
        '- C0=context+pre2；C1=context+pre1；C2=context+pre2+pre1；C3=context+pre2+pre1+registered history。',
        '- 复发history为对应earlier response；多事件history为全部合格prior acute responses的均值。',
        '- pre2固定使用既有记录，不搜索更远替代。每个case内四模型完全相同人员、fold和时间切点。所有recency结果均为secondary。',
        '- C1−C0是当前状态与较早状态的非嵌套比较；C2−C1是加入pre2；C3−C2是同时控制两个状态后加入历史。',
        '- ANCOVA要求所有模型有pre1，因此C0与C2相同；该形式下C1−C0不再是单独pre1对单独pre2的比较。仍按注册要求并列报告，不修改模型。', '',
        '## Recency complete-case人数和时间切点', '',
        md(cases.loc[cases.benchmark.eq('recency'), ['sample_id', 'n_parent', 'n_complete', 'missing_excluded',
           'temporal_cutoff', 'temporal_train_n', 'temporal_test_n', 'temporal_person_overlap_excluded']]), '',
        '## 标准化change-score比较', '',
        md(increments(recency.loc[recency.scale.eq('standardized') & recency.formulation.eq('change_score')])), '',
        'ANCOVA、raw-scale、各模型与各增量的六项评价指标及其CI全部见04 CSV。这里不对recency作因果解释。', '', *methods,
    ]))
    secondary = r.loc[r.analysis_role.eq('secondary')]
    sensitivity_inventory = runs.groupby(['analysis', 'benchmark', 'representation', 'formulation', 'scale', 'analysis_role']).size().reset_index(name='model_set_runs')
    (OUT / '05_sensitivity_results.md').write_text('\n'.join([
        '# Registered secondary and sensitivity results', '', *preamble, '',
        '本文件按注册角色汇总次分析，不将次分析替代主分析。05 CSV是02/03/04中secondary记录的可检索汇集，同一结果的重复收录不代表新增独立分析。', '',
        '## 完整运行清单', '', md(sensitivity_inventory), '',
        '## 预定敏感性', '',
        '- HRS recurrence与multi-event为secondary，无论结果方向如何。',
        '- 两cohort unemployment/health和UKHLS financial-strain recurrence为secondary。',
        '- Raw-scale：所有模型使用原始GHQ/CES-D量尺；点预测逆变换后已与标准化版本逐人核对。',
        '- ANCOVA：对应follow-up水平为目标，mandatory pre1；与change-score配套报告，不选择较有利的形式。',
        '- Multi-event most-recent、fold-residualized mean与between-history SD均保持secondary。',
        '- Overlap-allowed历史集合与recency complete-case集合均保持secondary。', '',
        '## UKHLS caregiving raw-scale增量（固定展示层）', '',
        md(increments(rec.loc[rec.cohort.eq('UKHLS') & rec.event_family.eq('caregiving') & rec.scale.eq('raw')])), '',
        '## Multi-event overlap-allowed标准化change-score增量（固定展示层）', '',
        md(increments(multi.loc[multi.variant.eq('overlap_allowed') & multi.scale.eq('standardized') & multi.formulation.eq('change_score')])), '',
        f'完整05 CSV包含{len(secondary):,}条次分析指标/增量记录。全部方向、CI和状态均保留。', '', *methods,
    ]))
    primary = r.loc[r.analysis_role.eq('primary')]
    core = increments(primary)
    hrs = increments(main_multi.loc[main_multi.cohort.eq('HRS') & main_multi.formulation.eq('change_score')])
    numerical_checks = []
    for validation_name in ('grouped_cv', 'temporal'):
        uk = primary.loc[primary.analysis.eq('multi_event') & primary.validation.eq(validation_name) &
                         primary.term.eq('M3-M2') & primary.metric.eq('predictive_r2')].iloc[0]
        h = main_multi.loc[main_multi.cohort.eq('HRS') & main_multi.formulation.eq('change_score') &
                           main_multi.validation.eq(validation_name) & main_multi.term.eq('M3-M2') &
                           main_multi.metric.eq('predictive_r2')].iloc[0]
        numerical_checks.append({'validation': validation_name, 'UKHLS_delta_R2': uk.estimate,
                                'UKHLS_CI_lower': uk.ci_lower, 'UKHLS_CI_upper': uk.ci_upper,
                                'HRS_secondary_delta_R2': h.estimate,
                                'UKHLS_delta_gt_0.0025': bool(uk.estimate > 0.0025),
                                'UKHLS_CI_excludes_zero': bool(uk.ci_lower > 0 or uk.ci_upper < 0),
                                'same_direction': bool(np.sign(uk.estimate) == np.sign(h.estimate))})
    (OUT / '06_registered_extension_final_summary.md').write_text('\n'.join([
        '# Registered extension: final numerical summary', '', *preamble, '',
        '## 完成范围', '',
        '- 18组资格样本，48组固定完整病例/验证集合，192个注册模型组合运行（包含队列、表示、量尺、formulation和敏感性组合）。',
        '- 每组合4个模型、10-fold与一次时间外推，共8,448份OLS训练记录。384组评价各完成500次成功person-cluster paired bootstrap，共192,000份bootstrap重复摘要。',
        '- 保存146,924条person-level held-out记录（跨分析集合的记录数，不是unique participants）和14,688条指标/增量记录。',
        '- 所有拟合与指标可计算。原始来源、OSF PDF、实施说明及样本/验证清单哈希保持不变。', '',
        '## 主分析：standardized change-score', '', md(core), '',
        '## HRS multi-event：secondary conceptual replication', '', md(hrs), '',
        '## 注册参照值的数值核对', '', md(pd.DataFrame(numerical_checks)), '',
        '该表只核对v2预设的数值条件，不作理论、因果或发表判断。0.005与0.010亦保留为原协议的描述参照，不据此选择模型或样本。', '',
        '## 样本与时间验证', '',
        md(cases.loc[cases.benchmark.eq('main') & cases.representation.isin(['same_response', 'mean']),
                     ['sample_id', 'n_complete', 'temporal_cutoff', 'temporal_train_n', 'temporal_test_n',
                      'temporal_person_overlap_excluded']]), '',
        '## 已确认限制', '',
        '- HRS及其他注册次分析维持secondary层级。时间holdout人数小于cohort资格人数，不能混用分母。',
        '- HRS parental caregiving与UKHLS caregiving、HRS persistent ADL与UKHLS long-standing illness/disability并非相同测量。',
        '- 事件以访谈状态transition对齐，真实发生日期不完整；当前pre1不是保证未受事件影响的因果基线。',
        '- 持续health确认的可知时间与index date不同；本轮明确限制历史及context事件计数的信息截止时间。',
        '- 有complete-case、复发资格、长期结果删失和重叠窗口筛选；这些结果不代表人群发生率。',
        '- HRS静态education的获得时间不能从该字段确认。Unknown category被保留，但不当作完整实测教育信息。',
        '- OOF指标在fold-specific训练量尺上汇总；raw-scale敏感性另报，不能混合两cohort原分。',
        '- ANCOVA存在因mandatory pre1造成的重复模型；已经标注，不当作独立重复证据。',
        '- Gaussian density为固定训练残差方差plug-in评价，不包括训练参数不确定性。Bootstrap固定已有预测，不覆盖全部训练和分折不确定性。', '',
        '- 不同scale/formulation使用估计前代码固定的不同bootstrap随机流；相同点指标的CI可能有Monte Carlo差异。同一run内嵌套模型始终配对，未按结果重抽样。', '',
        '## 文件与复现', '',
        '- 00_registration_and_hash_record.md：注册与输入来源。',
        '- 00_pre_estimation_implementation_note.md：用户批准的三项技术操作化；archive/pre_estimation_implementation_hash.json：封存时间与SHA-256。',
        '- 01_sample_rebuild_and_QA.md / 01_sample_counts.csv：资格与验证计数。',
        '- 02、03、04、05的MD与CSV：复发、多事件、recency、全部次分析。',
        '- archive/sample_lock与archive/validation_lock：不可覆写清单、固定fold、时间assignment、run registry。',
        '- archive/predictions：逐run的OOF/temporal person predictions；archive/bootstrap：500次paired resample摘要。',
        '- archive/all_fit_records.parquet：各训练折的标准化、OLS训练方差、rank、系数及预测列。',
        '- archive/data_dictionary.md、software_environment.json、independent_analysis_validation.json、final_artifact_manifest.json：字典、环境、独立QA与文件哈希。',
        '- 个人级UKHLS/HRS数据只保存在本地，不上传OSF。未进行网络搜索，未写Introduction/Discussion，未修改既有确认性结果。', '']))
    (ARC / 'validation_split_QA.md').write_text('\n'.join([
        '# Fixed complete-case sets and temporal splits', '',
        '本表在任何本轮结果查看前由ID、日期和缺失标志固定。每行四个嵌套模型使用完全相同的person集合。', '',
        md(cases[['sample_id', 'benchmark', 'representation', 'n_parent', 'n_complete', 'missing_excluded',
                  'temporal_cutoff', 'temporal_train_n', 'temporal_test_n', 'temporal_person_overlap_excluded']]), '',
        '所有48个case均保留10个非空person folds。cutoff为实际第ceil(0.70×N)个观察日期，同日全部训练。人员重叠排除数均为0，因为每个样本已经每人唯一target。', '']))
    (ARC / 'data_dictionary.md').write_text('''# Registered extension archive dictionary

## Identity and roles

- `cohort`: UKHLS or HRS. Never combine raw scores across cohorts.
- `person_id`: exact within-cohort longitudinal ID; the cross-cohort key is cohort + person_id.
- `sample_id`: registered recurrence family/outcome or multi-event overlap variant.
- `case_id`: sample + benchmark + historical representation, after identical complete-case selection.
- `run_id`: deterministic SHA-256 prefix of case + formulation + scale. Links all archives.
- `target_episode_id`, `history_episode_id`: immutable existing episode identifiers.
- `analysis_role`: primary only for standardized change-score UKHLS caregiving acute/persistence and UKHLS non-overlap mean acute history; all other runs secondary.
- `base_role`: role before raw-scale or ANCOVA sensitivity designation.
- `variant`: same_domain, nonoverlap, or overlap_allowed; never relabel overlap sensitivity as primary.

## Sample and timing fields

- `event_date`, `pre1_date`, `pre2_date`, `year2_date`: existing actual interview dates, not asserted true onset dates.
- Corresponding `*_wave` fields: original aligned survey-wave IDs.
- `prior_episode_count`: qualifying completed histories; `prior_family_count`: distinct historical families.
- `historical_observation_overlap_pairs`: historical response pairs sharing a person-wave mental-health observation.
- `history_order`: chronological eligible historical order; no target or future outcome enters this ordering.
- `information_cutoff_date`: target pre1 date; historical response and qualification must already be observed.
- `history_qualification_date`: date the event definition becomes ascertainable, including persistent-health confirmation.
- `*_available`, `*_known`, `baseline_usable`: score/covariate availability masks, not score values.
- `n_complete`, `n_parent`, `missing_excluded`: model-case denominator, original eligible denominator, excluded for required missing inputs.
- `outer_fold`: locked participant fold 0–9. All nested models retain identical persons.
- `temporal_cutoff`: observed target date at 1-based ceil(0.70*N); ties train.
- `temporal_assignment`: train/test; `temporal_person_overlap_excluded`: overlap removals after temporal split.

## Psychological fields and historical representations

- UKHLS raw outcome is GHQ-12 Likert 0–36; HRS raw outcome is CES-D 0–8; higher values indicate more symptoms/distress.
- Acute = event − pre1; persistence = year2 − pre1. Year2 retains existing subsequent-primary-event censoring.
- `formulation=change_score`: corresponding acute/persistence change target.
- `formulation=ancova_level`: event level for acute, year2 level for persistence; mandatory pre1 in every model.
- `representation=same_response`: earlier same-family corresponding response.
- `mean`: arithmetic mean of all qualifying prior acute responses.
- `most_recent`: last eligible prior acute response by locked historical order.
- `residualized_mean`: mean residual from outer-training-history OLS on prior family, pre1, age, year, and observed preceding primary count.
- `between_history_sd`: sample SD, ddof=1, of at least two prior acute responses; a registered secondary history representation.

## Predictions and fitted models

- Each `archive/predictions/{run_id}_{validation}.parquet` row is one held-out person for that run; validation is grouped_cv or temporal.
- `y_observed`: target on that run's reported scale; `y_raw`: original-scale target.
- `target_pre1_raw`: current pre1 for inverse-scale and ANCOVA QA.
- `training_pre1_mean`, `training_pre1_sd_ddof1`: common training-target-pre1 location/scale in standardized runs; 0 and 1 denote identity transformation in raw-scale runs, not empirical means/SDs.
- `prediction_R0..R3`, `prediction_M0..M3`, `prediction_C0..C3`: held-out model point predictions.
- `prediction_raw_*`: point predictions expressed in original cohort units.
- `sigma2_*`: each model's own RSS_train/n_train, on its reported outcome scale.
- `log_density_*`: plug-in Gaussian log density using that training variance; no test variance or parameter-uncertainty correction.
- `history_feature`: held-out person's historical feature computed using only their past observations and the outer-training transforms.
- `all_fit_records.parquet`: one row per run/split/model; n_train, n_test, training parameters, RSS, sigma2, rank, design columns, numerical conditioning, coefficients and training/test ID hashes.
- `coefficient_json`: coefficients for intercept and numerically conditioned predictors listed in `predictor_columns`; use saved conditioning_mean_json/conditioning_sd_json for reproduction. These are not automatically raw-unit covariate effects.
- `event_count_information_cutoff_QA.parquet`: contextual preceding-primary count known by the corresponding pre1 versus the legacy preceding-index count. Persistent health confirmation after pre1 is not treated as already-known context.

## Results and bootstrap

- `row_type`: model performance or increment; `term`: model name or augmented−baseline contrast.
- `metric`: predictive_r2, rmse, mae, mean_log_predictive_density, calibration_intercept, calibration_slope.
- `estimate`, `ci_lower`, `ci_upper`: held-out point estimate and paired-person 95% percentile bootstrap interval.
- All increments are augmented minus baseline; negative error increments mean lower RMSE/MAE; calibration increments have no universal preferred sign.
- Predictive R² = 1−SSE/SST_test. OOF metrics use all held-out persons, not averaged fold-specific R² values.
- Calibration is held-out y regressed on predictions with intercept; it is an evaluation only, never used to recalibrate predictions.
- `bootstrap_successful`, `bootstrap_attempted`, `bootstrap_seed`: 500 accepted paired draws, attempted draw count, deterministic seed.
- Bootstrap seeds are derived from the fixed seed, run_id and validation. Nested models share draws within a run; different scale/formulation runs have different fixed random streams, so Monte Carlo CI differences are possible even for identical point metrics.
- Each bootstrap Parquet contains run_id, validation, bootstrap_id and columns `{model_or_contrast}|{metric}`; all models in a row share resampled person bundles. Training models are not refitted.
- `status`, `interval_status`: mathematical estimability and interval completion, not theory/publication decisions.
- 02/03/04 CSVs partition all registered runs; 05 repeats their secondary rows for retrieval, not additional analyses.

## Boundaries

Only OSF v2 and the user-approved pre-estimation technical note control the analysis. No v1/DRAFT rules, new event families, subgroup searches, causal claims, or unregistered target outcomes were used. Individual-level archives are local restricted-data artifacts, not public-upload files.
''')
    print(json.dumps({'result_rows': len(r), 'secondary_rows': len(secondary), 'reports_written': 5,
                      'csv_intermediates_ready': 4, 'independent_validation_passed': True}, indent=2))


if __name__ == '__main__':
    main()
