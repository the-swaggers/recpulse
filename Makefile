help:
	@echo "help - display this message"
	@echo "format - formats all .py files and makes sure they are styled well"
	@echo "lint - static type checker"
	@echo "docstyle - checks docstrings"
	@echo "test - run unit tests"
	@echo "test_mnist - run simple NN on MNIST dataset (this one is a way heavier task)"


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


test:
	@pytest tests/*.py


test_mnist:
	@python -m tests.MNIST.main


build_cuda:
	@nvcc -o cuda_utils/tensor.so --shared -Xcompiler -fPIC cuda_utils/tensor.cu
	@nvcc -o cuda_utils/libcuda_config.so --shared -Xcompiler -fPIC cuda_utils/cuda_config.cu

