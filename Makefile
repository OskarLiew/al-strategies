.PHONY: test
test:
	pytest
	black .
	mypy .
	pylint al_strats/ tests/
