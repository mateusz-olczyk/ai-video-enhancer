.PHONY: format

format:
	python3 -m isort src
	python3 -m black src
