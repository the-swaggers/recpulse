help:
	@echo "help - display this message"
	@echo "format - formats all .py files and makes sure they are styled well"


test:
	@pytest tests/*.py


format:
	@black recpulse/*.py tests/*.py tests/*.py tests/MNIST/main.py


lint:
	@echo "run flake8"
	@flake8 recpulse/*.py tests/*.py tests/MNIST/main.py
	@echo "run mypy"
	@mypy recpulse/*.py tests/*.py tests/MNIST/main.py
	@echo "run isort"
	@isort recpulse/*.py tests/*.py tests/MNIST/main.py


docstyle:
	@echo "run pydocstyle"
	@pydocstyle recpulse/*.py tests/*.py tests/MNIST/main.py


test_mnist:
	@python -m tests.MNIST.main
