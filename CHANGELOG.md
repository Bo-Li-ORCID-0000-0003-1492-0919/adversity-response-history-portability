# Changelog

## 1.0.2 - 2026-09-06

Closed cross-file complementary disclosure paths by linking participant-count fields across public files by `sample_id`. Auxiliary pre2-completeness, model-evaluation and related counts are blanked where retained counts could reveal 1 through 10 observations; release notes identify the affected rows. The validation suite now fails on within-sample count differences of 1 through 10 while excluding positions, waves, proportions, tuning parameters and model estimates. No model result or manuscript display was changed.

## 1.0.1 - 2026-09-06

Strengthened public-release disclosure control without changing any model, coefficient, confidence interval or analysis sample. Small `missing_excluded` cells and their complementary `n_parent` cells are suppressed in the public benchmark files, additional recurrence-distribution cells are suppressed to prevent recovery by subtraction, and the validator now checks these cases.

## 1.0.0 - 2026-09-05

Initial public release accompanying the manuscript. It includes the analysis scripts, aggregate source data, figure-reproduction tools, table workbooks and registration materials. The HRS role label in the Figure 1 source workbook was corrected to “independent conceptual replication”; no numerical result was changed.
