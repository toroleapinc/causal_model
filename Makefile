.PHONY: install run test notebook clean

install:
	pip install -e ".[dev]"

run:
	python scripts/run_analysis.py

test:
	pytest tests/ -v

notebook:
	jupyter notebook notebooks/causal_inference_demo.ipynb

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache *.egg-info dist build
