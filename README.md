# Adversity-response history portability

This repository accompanies the manuscript **“Limited predictive portability of resilience inferred from adversity-response histories.”**

The study asks whether a person’s response to an earlier adversity helps predict their response to a later adversity. It examines three settings: different adversity types, recurrent caregiving, and summaries of several prior events. It also compares response history with mental health measured immediately before the later event.

## Contents

- `src/analysis/`: analysis scripts used for the UKHLS and HRS work.
- `src/figures/`: scripts that reproduce the main and Extended Data figures from aggregate source data.
- `scripts/`: public validation and table-preparation tools.
- `source_data/`: aggregate results and source-data workbooks. No participant-level rows are included.
- `extended_data_tables/` and `supplementary_tables/`: tables supplied with the manuscript.
- `expected_outputs/`: final figure PDFs used as references for reproduction checks.
- `registration/`: the registered extension protocol and a short note on its scope.

## Data access

UKHLS and HRS participant-level data are not included. Researchers must obtain the two datasets separately from their official providers and follow the relevant terms of use. Participant IDs, episode-level files, fold assignments and individual predictions must not be added to this repository. See `docs/DATA_ACCESS.md`.

## Reproduce the public figures and tables

The public workflow uses aggregate files only. It does not fit models or read UKHLS or HRS participant-level data.

Use Python 3.12 and install the packages in `requirements.txt`. Poppler is also required so that `pdftoppm` is available. The manuscript figures use Arial. On systems where Arial is not available, the scripts use Liberation Sans as a fallback. Set `NMH_ARIAL_FONT_DIR` to a folder containing `Arial.ttf` and `Arial Bold.ttf` when exact manuscript typography is required.

```bash
python -m pip install -r requirements.txt
make check
make public-artifacts
make figures-qa
```

Generated files are written to `outputs/figure_reproduction/` and `outputs/table_reproduction/`. The table step validates the supplied workbooks and copies them to a separate output folder; it does not rebuild tables from participant-level data.

`make figures-qa` compares the generated figures with the reference PDFs. It checks page size, all extracted text and all displayed numbers, and reports raster differences. PDF bytes and rasterization can vary across systems because of fonts and graphics libraries. Optional review PNGs can be created by running either figure script with `--render-review-pngs`.

## Full analysis

A full rerun requires licensed UKHLS study 6614 data, RAND HRS Longitudinal File 2022, RAND HRS Family Data 2022, and intermediate project files produced during cohort construction. The analysis scripts are provided for transparency, but this repository does not offer a one-command clean-room rerun. The required layout and stage order are described in `docs/REPRODUCIBILITY.md`.

## Registration

The recurrent-caregiving and multiple-history extensions were registered on [OSF](https://osf.io/zgeb3/) before their coefficients were estimated. The original cross-adversity analyses were completed earlier and are not described as preregistered. See `docs/PREREGISTRATION.md` and `registration/REGISTRATION_SCOPE.md`.

## Citation

Citation metadata are provided in `CITATION.cff`. The manuscript DOI will be added after publication.

## License

Author-written code is released under the MIT License. UKHLS and HRS participant-level data are not included and remain subject to the providers' terms. See `docs/LICENSE_SCOPE.md` for the scope of the repository licence.

## Contact

Bo Li: bo-li@sjtu.edu.cn
