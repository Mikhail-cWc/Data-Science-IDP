PYTHON = python3
COVERAGE = coverage

SRC_FILES = score_function.py score_function_test.py

TARGET = score_function.py score_function_test.py

coverage:
	$(COVERAGE) run -m pytest $(SRC_FILES)
	$(COVERAGE) report
    
test:
	$(PYTHON) -m unittest $(TARGET)

clean:
	rm -rf __pycache__ .pytest_cache .coverage

