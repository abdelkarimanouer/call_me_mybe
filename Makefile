UV_CACHE_DIR = /goinfre/aanouer/.uv_cache
HF_HOME = /goinfre/aanouer/.hf_cache

export UV_CACHE_DIR HF_HOME

setup:
	@mkdir -p $(UV_CACHE_DIR) $(HF_HOME)
	@uv venv /goinfre/aanouer/callme
	@ln -sf /goinfre/aanouer/callme .venv

run:
	@uv run -m src $(ARG)

install:
	@uv sync
	@uv pip install -e llm_sdk

debug:
	@uv run -m pdb -m src $(ARG)

clean:
	@find . -type d \( -name "__pycache__" -o -name ".mypy_cache" \) -exec rm -rf {} +

lint:
	@uv run flake8 src
	@uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
