.PHONY: install ruff ty	

install:
	uv sync --all-groups


ruff:
	uv run ruff check --fix --unsafe-fixes --extend-select I
	uv run ruff format

ty:
	uv run ty check

