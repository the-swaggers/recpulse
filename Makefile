help:
	@echo "help - display this message"
	@echo "format - formats all .py files and makes sure they are styled well"


test:
	@pytest tests/*.py


format:
	@black classes/*.py tests/*.py


lint:
	@flake8 classes/*.py tests/*.py
	@mypy classes/*.py tests/*.py
	@isort --check --diff classes/*.py tests/*.py
	@pydocstyle classes/*.py tests/*.py
