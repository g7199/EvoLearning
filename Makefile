# SimPath Experiment Runner
# ─────────────────────────────────────────────────────────────
# make test                    단위 테스트 (~3초)
# make run-mock                mock 모드 (LLM 없이, ~2분)
# make run-openai              OpenAI GPT-5.4 ($OPENAI_API_KEY 필요)
# make run-anthropic           Anthropic Claude Sonnet 4.6 ($ANTHROPIC_API_KEY 필요)
# make run-both                GPT-5.4 + Claude Sonnet 4.6 동시 (논문용)
# ─────────────────────────────────────────────────────────────

SHELL := /bin/bash
CONDA_ACTIVATE := source ~/miniconda3/etc/profile.d/conda.sh && conda activate sim
PYTHON := $(CONDA_ACTIVATE) && python

.PHONY: test run-mock run-openai run-anthropic run-both clean

test:
	$(PYTHON) -m pytest tests/test_core.py -v

run-mock:
	$(PYTHON) scripts/run_experiment.py --provider mock

run-openai:
	$(PYTHON) scripts/run_experiment.py --provider openai

run-anthropic:
	$(PYTHON) scripts/run_experiment.py --provider anthropic

run-both:
	$(PYTHON) scripts/run_experiment.py --provider both

clean:
	rm -rf outputs/results/* outputs/logs/* __pycache__ .pytest_cache
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
