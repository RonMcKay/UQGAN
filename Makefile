all: test

test:
	flake8 . --count --statistics
	black --check .
	isort --check --settings-path .isort.cfg .

format:
	black .
	isort --settings-path .isort.cfg .

clean:
	rm -rf build sdist __pycache__ __local__storage__ _cache_*
