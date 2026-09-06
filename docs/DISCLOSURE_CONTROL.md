# Disclosure control

Before a commit, run `make check` and inspect the staged files.

The validator blocks common licensed-data formats, participant identifier columns, spreadsheet errors, local data paths and files larger than 100 MB. CSV and XLSX files are checked by content rather than by extension alone. Aggregate files may contain sample counts and model results, but they must not contain participant IDs. As an added precaution, unpublished count cells from 1 through 10 are suppressed in the public analysis-output files. Additional cells are suppressed where necessary to prevent a suppressed value or small-count tail from being recovered by subtraction. The validator links explicitly defined participant-count fields by `sample_id` across the public analysis outputs and display-source CSVs; it fails if two retained counts for the same sample differ by 1 through 10. Fields such as item position, wave, proportions, tuning parameters and model estimates are excluded from this count check. Suppressed rows are marked by `count_suppressed` and a `public_release_note`. Counts reported in the manuscript and its displays are retained.

Do not add raw UKHLS or HRS data, participant-level derivatives, ethics applications, CVs, reference PDFs, email, credentials or old manuscripts.
