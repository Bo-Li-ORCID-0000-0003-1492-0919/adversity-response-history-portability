# Reproducibility

## Public aggregate workflow

The public workflow reads only the aggregate CSV and XLSX files included in this repository. It generates the four main figures and eight Extended Data figures. It also validates the table workbooks and copies them to a separate output folder.

```bash
make check
make public-artifacts
make figures-qa
```

The figure scripts do not fit models, resample participants or read licensed data. Output is written under `outputs/`, which is ignored by Git.

Arial regular and bold reproduce the manuscript typography. If Arial is unavailable, the scripts use Liberation Sans as a fallback. Set `NMH_ARIAL_FONT_DIR` to a directory containing `Arial.ttf` and `Arial Bold.ttf` when exact typography is required. Poppler is required for `make figures-qa`. The QA script checks page dimensions, extracted text and displayed numbers. It also reports raster differences, which can vary with font and PDF-rendering libraries.

Optional review PNGs can be created with:

```bash
python src/figures/reproduce_main_figures.py --render-review-pngs
python src/figures/reproduce_extended_data_figures.py --render-review-pngs
```

## Full analysis with licensed data

A full rerun requires:

- UKHLS study 6614 Stata files;
- RAND HRS Longitudinal File 2022;
- RAND HRS Family Data 2022;
- intermediate project files produced by the earlier cohort-construction stages.

The analysis scripts retain the original project structure. They do not yet provide a single clean-room command, and some local path settings must be adapted before use. The main stages were run in this order:

1. `02_build_linkage_panels.py`
2. `03_constructs_events.py`
3. `build_person_event_response_dataset.py`
4. `07_cross_stressor_reactivity_feasibility.py`
5. `run_nmh_confirmatory_rebuild.py --stage build`
6. `run_nmh_confirmatory_rebuild.py --stage analysis`
7. `run_nmh_measurement_adjudication.py`
8. `audit_nmh_extension_feasibility.py` and `validate_nmh_extension_feasibility.py`
9. `build_nmh_registered_extension.py`
10. `run_nmh_registered_extension.py prepare`
11. `run_nmh_registered_extension.py fit`
12. `report_nmh_registered_extension.py`
13. `finalize_nmh_registered_extension.py`
14. Export aggregate results and run the public display workflow.

The licensed data and participant-level intermediate files are not distributed. Runtime and peak memory were not recorded consistently.

## Recorded software versions

The registered extension recorded Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, PyArrow 25.0.1 and pypdf 6.10.0. The public display workflow also uses reportlab 4.4.9, openpyxl 3.1.5 and Pillow 12.3.0.

Earlier analysis records listed Python 3.12.13, NumPy 2.5.2, pandas 2.2.3 and SciPy 1.18.1. These versions are listed in `requirements-analysis.txt`; they were not re-tested while preparing the public aggregate workflow.
