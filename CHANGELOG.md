# Changelog

## 1.0.3

Packages the changes prepared and checked locally on 7 September 2026, with release metadata prepared on 12 September 2026. The publication date is recorded by the GitHub release and its corresponding archive. Version 1.0.2 is retained unchanged as the preceding archive.

Integrated the eight revised Extended Data figure PDFs, their exact display script and label tests from the corrected submission package. Aggregate source values and the four main figure designs were not changed. Updated only the local-plan and implementation-note wording in Supplementary Table 4; its registration row and numeric values were retained. Corrected optional figure commands to include their required output directory.

Subsequent visual QA identified automatic system-Arial substitution when reproducing the approved Liberation Sans Extended Data PDFs on macOS. This version pins unmodified Liberation Sans regular/bold assets (SIL Open Font License included) as the Extended Data default. Main figure fonts and all approved reference PDFs remain unchanged.

Reproduced the earlier analysis branches in the licensed local environment and saved previously missing fold, scale, fit, prediction and bootstrap records as explicitly post-result artifacts. All 3,493 checked aggregate rows reproduced within the declared numerical tolerance, with no four-decimal display changes. The replays did not invoke the evaluation-MSE variance fallback. Historical records were not overwritten or backdated.

Removed the invalid-training-variance evaluation-MSE fallback from the two earlier metric routines in this version: such calls now raise an explicit error. Finite positive variance calculations are unchanged. The registered extension's existing not-estimable rule is unchanged. Bounded the descriptive bootstrap retry loop so repeated singular solves cannot bypass the attempt limit. These are prospective safety guards, not changes to the reported estimates. Added regression tests and method/reproduction documentation. The 7 September preparation did not modify any remote release or archive.

## 1.0.2 - 2026-09-06

Closed cross-file complementary disclosure paths by linking participant-count fields across public files by `sample_id`. Auxiliary pre2-completeness, model-evaluation and related counts are blanked where retained counts could reveal 1 through 10 observations; release notes identify the affected rows. The validation suite now fails on within-sample count differences of 1 through 10 while excluding positions, waves, proportions, tuning parameters and model estimates. No model result or manuscript display was changed.

## 1.0.1 - 2026-09-06

Strengthened public-release disclosure control without changing any model, coefficient, confidence interval or analysis sample. Small `missing_excluded` cells and their complementary `n_parent` cells are suppressed in the public benchmark files, additional recurrence-distribution cells are suppressed to prevent recovery by subtraction, and the validator now checks these cases.

## 1.0.0 - 2026-09-05

Initial public release accompanying the manuscript. It includes the analysis scripts, aggregate source data, figure-reproduction tools, table workbooks and registration materials. The HRS role label in the Figure 1 source workbook was corrected to “independent conceptual replication”; no numerical result was changed.
