.PHONY: install test lint typecheck check reproduce

install:
	pip install -e ".[dev]"

test:
	pytest --cov=eqdrisk --cov-report=term-missing

lint:
	ruff format --check .
	ruff check .

typecheck:
	mypy src/eqdrisk

check: lint typecheck test

reproduce:
	eqdrisk run --date $(DATE)
