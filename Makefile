.PHONY: format

format:
	python -m isort src
	python -m black src
