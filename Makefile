.PHONY: install test exp1 exp2 exp3 exp4 exp5 exp6 exp7 exp8 figures all clean

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

exp6:
	TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m experiments.exp6_significance --reuse_exp1

exp7:
	TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m experiments.exp7_case_study

exp8:
	TRANSFORMERS_OFFLINE=1 python -m experiments.exp8_industry_baselines

figures:
	MPLBACKEND=Agg python -m experiments.exp_figures
	MPLBACKEND=Agg python -m experiments.exp_architecture_fig

all: test exp1 exp2 exp3 exp4 exp5 exp6 exp7 exp8 figures

clean:
	rm -rf results/ __pycache__ .pytest_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
