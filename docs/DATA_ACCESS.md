# Data access and disclosure control

UKHLS participant-level data are not provided here. Eligible researchers must apply through the UK Data Service for study 6614 and follow the applicable licence. HRS participant-level data are also not provided; researchers must obtain the RAND HRS Longitudinal and Family products through the HRS data portal and follow its terms.

Local participant-level derivatives—including episode rows, IDs, eligibility masks, validation splits, sample locks and individual predictions—must not be redistributed or placed under version control. UKHLS and HRS remain separate cohort-specific inputs; participant rows are never pooled across providers.

The CSV and XLSX files included in `source_data/`, `extended_data_tables/` and `supplementary_tables/` contain non-identifiable aggregate display or model-level results. Do not attempt to re-identify study participants. Never commit authorized raw data or local derived data to this repository.
