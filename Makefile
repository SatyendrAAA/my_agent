.PHONY: run test lint demo clean

run:
	python3 -m agent

test:
	python3 -m unittest discover -s tests -t . -v

lint:
	python3 -m pyflakes agent tests || echo "install pyflakes: pip install pyflakes"

demo:
	python3 -m agent --offline --no-save "read README.md"
	python3 -m agent --offline --no-save "ls"
	python3 -m agent --offline --no-save "python: print(6 * 7)"

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf build dist *.egg-info .pytest_cache
