PYTHON ?= python

.PHONY: check figures tables public-artifacts figures-qa

check:
	$(PYTHON) -m compileall -q src scripts tests
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) scripts/validate_public_release.py

figures:
	$(PYTHON) src/figures/reproduce_main_figures.py --output-dir outputs/figure_reproduction/main
	$(PYTHON) src/figures/reproduce_extended_data_figures.py --output-dir outputs/figure_reproduction/extended_data

tables:
	$(PYTHON) scripts/reproduce_tables_from_source_data.py --output-dir outputs/table_reproduction

public-artifacts: figures tables

figures-qa:
	$(PYTHON) scripts/check_figure_reproduction.py --generated outputs/figure_reproduction
