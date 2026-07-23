.PHONY: setup test lint serve

VENV := venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

test:
	$(VENV)/bin/pytest

lint:
	@if $(VENV)/bin/ruff --version >/dev/null 2>&1; then \
		$(VENV)/bin/ruff check .; \
	elif command -v ruff >/dev/null 2>&1; then \
		ruff check .; \
	else \
		$(VENV)/bin/flake8 . 2>/dev/null || flake8 .; \
	fi

serve:
	$(VENV)/bin/uvicorn app.serving.main:app --host 0.0.0.0 --port 8000 --reload
