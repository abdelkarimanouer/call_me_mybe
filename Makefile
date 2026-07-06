run:
	@uv run -m src $(ARG)

install:
	@uv sync

debug:
	@uv run -m pdb -m src $(ARG)

clean:
	@rm -rf __pycache__ .mypy_cache .pytest_cache src/__pycache__

lint:
	@uv run flake8 src
	@uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
