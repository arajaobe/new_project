.PHONY: install run debug clean lint lint-strict

PROJECT_DIR := llm_sdk
PYTHON := $(PROJECT_DIR)/.venv/bin/python
MAIN := src/main.py

install:
	cd $(PROJECT_DIR) && uv sync

run:
	$(PYTHON) $(MAIN)

debug:
	$(PYTHON) -m pdb $(MAIN)

clean:
	find . -type d \( -name __pycache__ -o -name .mypy_cache -o -name .pytest_cache \) -prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 .
	mypy . --strict
