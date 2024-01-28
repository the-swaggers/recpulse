help:
	@echo "help - display this message"
	@echo "format - formats all .py files and makes sure they are styled well"


test:
	@pytest tests/*.py


format:
	@black src/*.py
	@black classes/*.py
