# Reproducible pipeline. Run `make help` for targets.
PYTHON ?= .venv/bin/python
FOUNDATION_PYTHON ?= .venv-foundation/bin/python
export PYTHONPATH := src
export MPLCONFIGDIR := $(CURDIR)/.cache/mpl
export MPLBACKEND := Agg

.PHONY: help install test lint prediction study foundation foundation-install memorization mechanisms figures paper report \
        presentation notebook docs reproduce clean

help:  ## Show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-13s %s\n", $$1, $$2}'

install:  ## Create .venv and install the package with dev and docs extras
	python3 -m venv .venv
	.venv/bin/pip install -e '.[dev,docs]'

test:  ## Run the test suite
	$(PYTHON) -m pytest -q

lint:  ## Static checks
	$(PYTHON) -m ruff check src scripts tests

prediction:  ## Validation audit: row-level vs mouse-level cross-validation
	$(PYTHON) scripts/run_analysis.py

study:  ## Factor decomposition and factorial protein contrasts
	$(PYTHON) scripts/study.py
	$(PYTHON) scripts/factorial_analysis.py

foundation:  ## Foundation-model benchmark (needs TabPFN/Kumo runtimes; see README)
	$(PYTHON) scripts/study.py --foundation

foundation-install:  ## Separate env with TabPFN and Kumo (large download: PyTorch)
	python3 -m venv .venv-foundation
	.venv-foundation/bin/pip install -e '.[foundation]'

memorization:  ## Contamination test: original vs obfuscated data (logistic, TabPFN, Kumo)
	$(PYTHON) scripts/memorization_test.py logistic
	PYTHONPATH=src:scripts $(FOUNDATION_PYTHON) scripts/memorization_test.py tabpfn
	PYTHONPATH=src:scripts $(FOUNDATION_PYTHON) scripts/memorization_test.py kumo

mechanisms:  ## Dosage, learning signature, rescue, pathways, network, power
	$(PYTHON) scripts/mechanism_analysis.py

figures:  ## All figures (prediction and mechanistic) from saved results
	$(PYTHON) scripts/make_figures.py
	$(PYTHON) scripts/make_mechanism_figures.py

paper:  ## Manuscript PDF (needs tectonic)
	$(PYTHON) scripts/make_paper.py

report:  ## Plain-language report, Markdown + PDF (needs pandoc and tectonic)
	$(PYTHON) scripts/make_report.py

presentation:  ## Slide deck, PPTX + PDF
	$(PYTHON) scripts/make_presentation.py

notebook:  ## Execute the walkthrough notebook in place (with this project's interpreter)
	mkdir -p .cache/jupyter/kernels/mice-protein
	printf '{"argv": ["%s", "-m", "ipykernel_launcher", "-f", "{connection_file}"], "display_name": "mice-protein", "language": "python"}' \
		"$(abspath $(PYTHON))" > .cache/jupyter/kernels/mice-protein/kernel.json
	JUPYTER_PATH=$(CURDIR)/.cache/jupyter $(PYTHON) -m jupyter nbconvert --to notebook --execute \
		--inplace --ExecutePreprocessor.kernel_name=mice-protein notebooks/walkthrough.ipynb

docs: paper report presentation  ## Paper, report and slides

reproduce: test lint prediction study mechanisms figures docs  ## Everything, from the raw workbook

clean:  ## Remove caches and build intermediates (keeps results)
	find src tests scripts -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .cache paper/*.log paper/*.aux
