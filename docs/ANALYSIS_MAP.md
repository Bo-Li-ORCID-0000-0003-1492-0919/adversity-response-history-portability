# Analysis and display map

This table links each paper display to the analysis script and aggregate file used to prepare it. The public figure scripts read aggregate source data only. A full analysis requires licensed UKHLS and HRS data.

| Display | What it shows | Analysis script(s) | Aggregate input | Public output | Checks |
|---|---|---|---|---|---|
| Figure 1 | Cohorts and the three response-history tests | `run_nmh_confirmatory_rebuild.py`; `run_nmh_registered_extension.py` | sample-flow and extension count files | `reproduce_main_figures.py` → `Figure_1.pdf` | displayed counts and role labels |
| Figure 2 | Cross-adversity prediction results | `run_nmh_measurement_adjudication.py` | baseline information decomposition | `reproduce_main_figures.py` → `Figure_2.pdf` | estimates, intervals and labels |
| Figure 3 | Recurrent caregiving and multiple prior responses | `run_nmh_registered_extension.py` | recurrence and multiple-history results | `reproduce_main_figures.py` → `Figure_3.pdf` | estimates, intervals and four-decimal display |
| Figure 4 | Recency benchmark | `run_nmh_registered_extension.py` | recency results | `reproduce_main_figures.py` → `Figure_4.pdf` | panel selection, estimates and labels |
| Table 1 | Cohorts and analysis samples | cohort construction and extension reporting | sample-flow and count files | `source_data/tables/Table_1.xlsx` | workbook readback and manuscript comparison |
| Table 2 | Cross-adversity information decomposition | `run_nmh_measurement_adjudication.py` | baseline information decomposition | `source_data/tables/Table_2.xlsx` | workbook readback and numeric comparison |
| Table 3 | Registered extension and secondary results | `report_nmh_registered_extension.py` | recurrence, multiple-history and recency results | `source_data/tables/Table_3.xlsx` | workbook readback and numeric comparison |
| Extended Data Figure 1 | Event-centred trajectories | `run_nmh_confirmatory_rebuild.py` | trajectory estimates | `reproduce_extended_data_figures.py` | all rows plotted and labels checked |
| Extended Data Figure 2 | Episode construction and availability | cohort construction | harmonization and sample-flow results | `reproduce_extended_data_figures.py` | all rows plotted and counts checked |
| Extended Data Figure 3 | Trajectory-model prediction | confirmatory analysis | model-comparison results | `reproduce_extended_data_figures.py` | estimates and intervals checked |
| Extended Data Figure 4 | Cross-adversity robustness analyses | confirmatory and measurement analyses | cross-cohort and alternative-outcome results | `reproduce_extended_data_figures.py` | estimates, intervals and labels checked |
| Extended Data Figure 5 | Item completeness and reliability | `run_nmh_measurement_adjudication.py` | item and reliability results | `reproduce_extended_data_figures.py` | values, denominators and units checked |
| Extended Data Figure 6 | Recurrence across event families | `run_nmh_registered_extension.py` | recurrence results | `reproduce_extended_data_figures.py` | all rows and displayed rounding checked |
| Extended Data Figure 7 | Alternative multiple-history summaries | registered extension analysis | multiple-history and overlap-sensitivity results | `reproduce_extended_data_figures.py` | all rows and representation labels checked |
| Extended Data Figure 8 | Weighting and complete-case analyses | confirmatory and extension sensitivity analyses | weighting and sensitivity results | `reproduce_extended_data_figures.py` | descriptive estimates checked |
| Extended Data Table 1 | Event definitions and sample flow | cohort construction | event and count results | `extended_data_tables/Extended_Data_Table_1.xlsx` | workbook structure and counts checked |
| Extended Data Table 2 | Model and validation metrics | confirmatory and extension analyses | model-level results | `extended_data_tables/Extended_Data_Table_2.xlsx` | workbook readback and metric labels checked |
| Supplementary Table 1 | Item mapping and reliability | measurement analysis | item and reliability results | `supplementary_tables/Supplementary_Table_1.xlsx` | workbook and error scan |
| Supplementary Table 2 | Registered predictive increments | extension analysis | recurrence, multiple-history and recency results | `supplementary_tables/Supplementary_Table_2.xlsx` | workbook and numeric comparison |
| Supplementary Table 3 | Weighting and attrition checks | sensitivity analyses | weighting and missingness results | `supplementary_tables/Supplementary_Table_3.xlsx` | workbook and error scan |
| Supplementary Table 4 | Registration and analysis chronology | analysis records | dates and analysis roles | `supplementary_tables/Supplementary_Table_4.xlsx` | dates and labels checked |

The table workbooks are supplied as aggregate outputs. The public workflow validates and copies them; it does not recreate them from participant-level data.
