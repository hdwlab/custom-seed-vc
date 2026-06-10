#
# Makefile
#

help: ## Show this help
	@echo "Help"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "    \033[36m%-20s\033[93m %s\n", $$1, $$2}'

.PHONY: default
default: help

.venv:
	uv sync

install: .venv

.PHONY: lint
lint: .venv		## Run the linter
	uv run ruff check .

.PHONY: test
test: .venv		## Run pytest
	uv run pytest tests

.PHONY: format
format: .venv	## Run the formatter
	uv run ruff format .

.PHONY: clean
clean:		## Clean up the project
	rm -rf .venv
	rm -rf .ruff_cache
	rm -rf .pytest_cache
	rm -rf __pycache__
	rm -rf *.egg-info
	rm -rf dist
	rm -rf build
