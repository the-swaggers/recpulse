help:
	@echo "help - display this message"
	@echo "format - formats all .py files and makes sure they are styled well"


format:
	@black src/*.py