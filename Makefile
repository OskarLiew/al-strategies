.PHONY: test
test:
	black .
	pytest
	mypy --ignore-missing-imports .
	pylint al_strats/ tests/
