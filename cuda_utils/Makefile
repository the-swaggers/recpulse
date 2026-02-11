# Python
PYTHON := python3
PYTHON_CONFIG := python3-config
PYTHON_INCLUDES := $(shell $(PYTHON_CONFIG) --includes)
PYTHON_LDFLAGS := $(shell $(PYTHON_CONFIG) --ldflags)

CC := gcc
NVCC := nvcc

CORE_DIR := core
FUNCTIONAL_DIR := functional
OPS_DIR := ops
OPTIM_DIR := optim
PYTHON_DIR := python
BUILD_DIR := build

TARGET := recpulse_cuda.so

C_SOURCES := $(CORE_DIR)/tensor.c $(CORE_DIR)/tensor_host.c \
             $(FUNCTIONAL_DIR)/functional.c $(FUNCTIONAL_DIR)/functional_host.c \
             $(OPS_DIR)/ops.c $(OPS_DIR)/grad_host.c \
             $(OPTIM_DIR)/optim.c $(OPTIM_DIR)/optim_host.c
CU_SOURCES := $(CORE_DIR)/tensor_device.cu $(CORE_DIR)/cuda_helpers.cu \
              $(FUNCTIONAL_DIR)/functional_device.cu \
              $(OPS_DIR)/grad_device.cu \
              $(OPTIM_DIR)/optim_device.cu
PY_SOURCES := $(PYTHON_DIR)/tensor_python.c

C_OBJECTS := $(patsubst $(CORE_DIR)/%.c,$(BUILD_DIR)/%.o,$(filter $(CORE_DIR)/%.c,$(C_SOURCES))) \
             $(patsubst $(FUNCTIONAL_DIR)/%.c,$(BUILD_DIR)/%.o,$(filter $(FUNCTIONAL_DIR)/%.c,$(C_SOURCES))) \
             $(patsubst $(OPS_DIR)/%.c,$(BUILD_DIR)/%.o,$(filter $(OPS_DIR)/%.c,$(C_SOURCES))) \
             $(patsubst $(OPTIM_DIR)/%.c,$(BUILD_DIR)/%.o,$(filter $(OPTIM_DIR)/%.c,$(C_SOURCES)))
CU_OBJECTS := $(patsubst $(CORE_DIR)/%.cu,$(BUILD_DIR)/%.o,$(filter $(CORE_DIR)/%.cu,$(CU_SOURCES))) \
              $(patsubst $(FUNCTIONAL_DIR)/%.cu,$(BUILD_DIR)/%.o,$(filter $(FUNCTIONAL_DIR)/%.cu,$(CU_SOURCES))) \
              $(patsubst $(OPS_DIR)/%.cu,$(BUILD_DIR)/%.o,$(filter $(OPS_DIR)/%.cu,$(CU_SOURCES))) \
              $(patsubst $(OPTIM_DIR)/%.cu,$(BUILD_DIR)/%.o,$(filter $(OPTIM_DIR)/%.cu,$(CU_SOURCES)))
PY_OBJECTS := $(patsubst $(PYTHON_DIR)/%.c,$(BUILD_DIR)/%.o,$(PY_SOURCES))

ALL_OBJECTS := $(C_OBJECTS) $(CU_OBJECTS) $(PY_OBJECTS)

CUDA_ARCH := -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_86,code=sm_86 \
             -gencode arch=compute_89,code=sm_89 \
             -gencode arch=compute_90,code=sm_90

CFLAGS := -O3 -fPIC -std=c11 -Wall -Wextra
CFLAGS += -I$(CORE_DIR) -I$(FUNCTIONAL_DIR) -I$(OPS_DIR) -I$(OPTIM_DIR) $(PYTHON_INCLUDES)
CFLAGS += -I/opt/cuda/include

NVCCFLAGS := -O3 --shared -Xcompiler -fPIC
NVCCFLAGS += -I$(CORE_DIR) -I$(FUNCTIONAL_DIR) -I$(OPS_DIR) -I$(OPTIM_DIR) $(PYTHON_INCLUDES)
NVCCFLAGS += $(CUDA_ARCH)

LDFLAGS := -shared
LDFLAGS += -L/opt/cuda/lib64 -lcudart -lcublas
LDFLAGS += $(PYTHON_LDFLAGS)

.PHONY: all clean help install config test test_functional test_sum_mean test_scalar test_split test_shape_ops test_backward test_optim test_all

all: $(TARGET)

$(TARGET): $(ALL_OBJECTS)
	@echo "Linking $@..."
	$(NVCC) $(LDFLAGS) -o $@ $^
	@echo ""
	@echo "Build complete: $@"
	@echo ""
	@echo "Usage:"
	@echo "  python3 -c 'import recpulse_cuda; print(recpulse_cuda.zeros([2, 3]))'"
	@echo ""

$(BUILD_DIR)/%.o: $(CORE_DIR)/%.c | $(BUILD_DIR)
	@echo "Compiling C: $<"
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(CORE_DIR)/%.cu | $(BUILD_DIR)
	@echo "Compiling CUDA: $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(FUNCTIONAL_DIR)/%.cu | $(BUILD_DIR)
	@echo "Compiling CUDA: $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(FUNCTIONAL_DIR)/%.c | $(BUILD_DIR)
	@echo "Compiling C: $<"
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(OPS_DIR)/%.c | $(BUILD_DIR)
	@echo "Compiling C: $<"
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(OPS_DIR)/%.cu | $(BUILD_DIR)
	@echo "Compiling CUDA: $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(OPTIM_DIR)/%.c | $(BUILD_DIR)
	@echo "Compiling C: $<"
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(OPTIM_DIR)/%.cu | $(BUILD_DIR)
	@echo "Compiling CUDA: $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(PYTHON_DIR)/%.c | $(BUILD_DIR)
	@echo "Compiling Python binding: $<"
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR) $(TARGET)
	@echo "Clean complete"

install: $(TARGET)
	@echo "Installing $(TARGET)..."
	cp $(TARGET) $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('platlib'))")
	@echo "Installation complete"
	@echo "You can now run: python3 -c 'import recpulse_cuda'"

test: $(TARGET)
	@echo "Running tensor tests..."
	@$(PYTHON) tests/unit/test_tensor.py

test_functional: $(TARGET)
	@echo "Running functional layer tests..."
	@$(PYTHON) tests/unit/test_functional.py

test_sum_mean: $(TARGET)
	@echo "Running sum_all/mean_all autograd tests..."
	@$(PYTHON) tests/autograd/test_sum_mean_autograd.py

test_scalar: $(TARGET)
	@echo "Running scalar autograd tests..."
	@$(PYTHON) tests/autograd/test_scalar_autograd.py

test_split: $(TARGET)
	@echo "Running split/split_equal autograd tests..."
	@$(PYTHON) tests/autograd/test_split_autograd.py

test_shape_ops: $(TARGET)
	@echo "Running shape operations tests..."
	@$(PYTHON) tests/autograd/test_shape_ops.py

test_backward: $(TARGET)
	@echo "Running backward topological sort tests..."
	@$(PYTHON) tests/autograd/test_backward_topo.py

test_optim: $(TARGET)
	@echo "Running optimizer tests..."
	@$(PYTHON) tests/integration/test_optimizers.py

test_all: $(TARGET)
	@echo "Running all tests..."
	@for f in tests/unit/*.py tests/autograd/*.py tests/integration/*.py; do \
		echo "--- $$f ---"; \
		$(PYTHON) $$f || exit 1; \
	done
	@echo ""
	@echo "All tests passed."

config:
	@echo "Build Configuration"
	@echo "  CC:              $(CC)"
	@echo "  NVCC:            $(NVCC)"
	@echo "  Python:          $(PYTHON)"
	@echo "  Python includes: $(PYTHON_INCLUDES)"
	@echo "  CUDA arch:       $(CUDA_ARCH)"
	@echo ""
	@echo "Source files:"
	@echo "  C sources:       $(C_SOURCES)"
	@echo "  CUDA sources:    $(CU_SOURCES)"
	@echo "  Python sources:  $(PY_SOURCES)"
	@echo ""
	@echo "Output:"
	@echo "  Target:          $(TARGET)"
	@echo "  Build dir:       $(BUILD_DIR)"

help:
	@echo "RecPulse CUDA Tensor Library - Makefile Help"
	@echo ""
	@echo "Targets:"
	@echo "  all              - Build the Python extension module (default)"
	@echo "  clean            - Remove all build artifacts"
	@echo "  install          - Install module to Python site-packages"
	@echo "  test             - Run tensor tests (tests/unit/)"
	@echo "  test_functional  - Run functional layer tests (tests/unit/)"
	@echo "  test_sum_mean    - Run sum_all/mean_all autograd tests"
	@echo "  test_scalar      - Run scalar autograd tests"
	@echo "  test_split       - Run split/split_equal autograd tests"
	@echo "  test_shape_ops   - Run shape operations tests"
	@echo "  test_backward    - Run backward topological sort tests"
	@echo "  test_optim       - Run optimizer tests (tests/integration/)"
	@echo "  test_all         - Run all tests"
	@echo "  config           - Show build configuration"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Quick Start:"
	@echo "  make                   # Build the module"
	@echo "  make test_all          # Run all tests"
	@echo "  make test              # Run tensor tests"
	@echo "  make test_functional   # Run functional layer tests"
	@echo "  make test_optim        # Run optimizer tests"
	@echo "  make install           # Install system-wide"
	@echo ""
	@echo "Test Directory Structure:"
	@echo "  tests/unit/        - Forward-pass correctness, tensor creation"
	@echo "  tests/autograd/    - Backward passes, gradient computation"
	@echo "  tests/integration/ - Multi-component workflows (optimizers)"
	@echo ""
	@echo "Development Tips:"
	@echo "  - For faster builds, edit CUDA_ARCH to target only your GPU"
	@echo "  - Uncomment debug flags in CFLAGS/NVCCFLAGS for debugging"
	@echo "  - Use 'make clean' before rebuilding after header changes"
	@echo ""
