PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin

.PHONY: install dev eval deploy quality style setup ingest

# Cluster-machine setup: preflight checks + install the eval scheduler.
setup:
	./pipeline/setup.sh

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -e ".[dev]"

# Serve the dashboard locally at http://localhost:8000 with real results data.
dev: site/results.json
	$(PYTHON) -m http.server 8000 -d site

site/results.json: results.json
	cp results.json $@

# Regenerate site/data.json from checkpoint eval folders (+ results.json seed).
ingest:
	$(PYTHON) pipeline/ingest.py

# Run the full eval cycle for all pending models (requires cluster access;
# see CHEATSHEET.md).
eval:
	./pipeline/refresh.sh

# Publish updated results (commits and pushes results.json + results/).
deploy:
	git add results.json results/
	git commit -m "eval results $$(date -u +%F)"
	git push

# Lint + verify all pipeline modules import cleanly.
quality:
	$(BIN)/ruff check pipeline/
	cd pipeline && ../$(BIN)/python -c "import discover, deploy_agent, ingest, normalize, orchestrate"

# Auto-format the pipeline code.
style:
	$(BIN)/ruff format pipeline/
