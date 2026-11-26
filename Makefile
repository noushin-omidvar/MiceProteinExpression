# Reproducible pipeline. Run `make help` for targets.
PYTHON ?= .venv/bin/python
FOUNDATION_PYTHON ?= .venv-foundation/bin/python
export PYTHONPATH := src
export MPLCONFIGDIR := $(CURDIR)/.cache/mpl
export MPLBACKEND := Agg

.PHONY: help install test lint prediction study mechanisms figures paper report \
        presentation notebook docs reproduce lock clean fm-install fm-experiments fm-analyze \
        fm-figures fm-docs fm-reproduce fm-matched fm-matched-review fm-sensitivity \
        fm-tuned fm-factorized fm-svm-audit fm-runtime fm-paper fm-all

help:  ## Show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-13s %s\n", $$1, $$2}'

install:  ## Create .venv with the exact pinned versions (requirements/lock-core.txt)
	python3 -m venv .venv
	.venv/bin/pip install -c requirements/lock-core.txt -e '.[dev,docs]'

test:  ## Run the test suite
	$(PYTHON) -m pytest -q

lint:  ## Static checks
	$(PYTHON) -m ruff check src scripts tests foundation_models

prediction:  ## Validation audit: row-level vs mouse-level cross-validation
	$(PYTHON) scripts/run_analysis.py

study:  ## Factor decomposition and factorial protein contrasts
	$(PYTHON) scripts/study.py
	$(PYTHON) scripts/factorial_analysis.py


mechanisms:  ## Dosage, learning signature, rescue, pathways, network, power
	$(PYTHON) scripts/mechanism_analysis.py

figures:  ## All figures (prediction and mechanistic) from saved results
	$(PYTHON) scripts/make_figures.py
	$(PYTHON) scripts/make_mechanism_figures.py
	$(PYTHON) scripts/make_workflows.py
	$(PYTHON) scripts/make_primer_figures.py

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

reproduce: test lint prediction study mechanisms figures docs  ## Biology study, from the raw workbook

# ---- Foundation-model study (foundation_models/) ------------------------------------------
fm-install:  ## Environment with TabPFN and Kumo (.venv-foundation; large PyTorch download)
	python3 -m venv .venv-foundation
	.venv-foundation/bin/pip install -c requirements/lock-foundation.txt -e '.[foundation,docs]'

fm-experiments:  ## Run every pre-specified experiment (slow on CPU)
	$(FOUNDATION_PYTHON) foundation_models/scripts/run_experiments.py all

fm-matched:  ## Post hoc fixed-holdout learning curves (slow on CPU)
	$(FOUNDATION_PYTHON) foundation_models/scripts/matched_holdout.py

fm-matched-review:  ## Five-draw matched-holdout learning curves
	$(FOUNDATION_PYTHON) foundation_models/scripts/matched_holdout.py --config matched_holdout_review.yaml --output matched_holdout_review

fm-sensitivity:  ## Post hoc aggregation and missingness controls
	$(FOUNDATION_PYTHON) foundation_models/scripts/sensitivity.py

fm-tuned:  ## Post hoc nested tuning of locally available conventional models
	$(FOUNDATION_PYTHON) foundation_models/scripts/tuned_baselines.py

fm-factorized:  ## Design-informed genotype × conditioning × treatment baseline
	$(FOUNDATION_PYTHON) foundation_models/scripts/factorized_baseline.py

fm-svm-audit:  ## Classification decisions separate from SVM probability argmax
	$(FOUNDATION_PYTHON) foundation_models/scripts/svm_main_audit.py
	$(FOUNDATION_PYTHON) foundation_models/scripts/svm_decision_audit.py

fm-runtime:  ## Controlled single-split timing decomposition
	$(FOUNDATION_PYTHON) foundation_models/scripts/runtime_audit.py

fm-analyze:  ## Hypothesis tests and summary.json
	$(PYTHON) foundation_models/scripts/analyze.py
	$(PYTHON) foundation_models/scripts/analyze_extensions.py

fm-figures:  ## Figures for the foundation-model study
	$(PYTHON) foundation_models/scripts/make_figures.py
	$(PYTHON) foundation_models/scripts/make_extension_figures.py
	$(PYTHON) foundation_models/scripts/make_workflows.py

fm-paper:  ## Foundation-model study paper PDF (needs tectonic)
	$(PYTHON) foundation_models/scripts/make_paper.py

fm-docs:  ## Paper, plain-language report and slides for the foundation-model study
	$(PYTHON) foundation_models/scripts/make_paper.py
	$(PYTHON) foundation_models/scripts/make_report.py
	$(PYTHON) foundation_models/scripts/make_presentation.py

fm-reproduce: fm-analyze fm-figures fm-docs  ## Rebuild figures and documents from saved results

fm-all:  ## Full foundation-model benchmark and figures (requires installed checkpoints)
	$(MAKE) fm-experiments
	$(MAKE) fm-tuned
	$(MAKE) fm-factorized
	$(MAKE) fm-matched
	$(MAKE) fm-svm-audit
	$(MAKE) fm-matched-review
	$(MAKE) fm-sensitivity
	$(MAKE) fm-runtime
	$(MAKE) fm-reproduce

lock:  ## Refresh both lock files from the current environments
	.venv/bin/pip freeze --exclude-editable | grep -v mice-protein > requirements/lock-core.txt
	.venv-foundation/bin/pip freeze --exclude-editable | grep -v mice-protein > requirements/lock-foundation.txt

clean:  ## Remove caches and build intermediates (keeps results)
	find src tests scripts -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .cache paper/*.log paper/*.aux
