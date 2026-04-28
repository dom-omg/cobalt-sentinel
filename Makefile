.PHONY: install test exp1 exp2 exp3 exp4 exp5 all clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=ide --cov-report=term-missing

exp1:
	TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m experiments.exp1_headline

exp2:
	TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m experiments.exp2_ablations

exp3:
	TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m experiments.exp3_adversarial

exp4:
	TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m experiments.exp4_real_traces

exp5:
	TRANSFORMERS_OFFLINE=1 python -m experiments.exp5_lower_bound

all: test exp1 exp2 exp3 exp4 exp5

clean:
	rm -rf results/ __pycache__ .pytest_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
