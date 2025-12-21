# Python
PYTHON := python3
PYTHON_CONFIG := python3-config
PYTHON_INCLUDES := $(shell $(PYTHON_CONFIG) --includes)
PYTHON_LDFLAGS := $(shell $(PYTHON_CONFIG) --ldflags)

# Compilers
CC := gcc
NVCC := nvcc

# Directories
CORE_DIR := core
PYTHON_DIR := python
BUILD_DIR := build

# Output
TARGET := recpulse_cuda.so

# Source files
C_SOURCES := $(CORE_DIR)/tensor.c $(CORE_DIR)/tensor_host.c
CU_SOURCES := $(CORE_DIR)/tensor_device.cu $(CORE_DIR)/cuda_helpers.cu
PY_SOURCES := $(PYTHON_DIR)/tensor_python.c

# Object files
C_OBJECTS := $(patsubst $(CORE_DIR)/%.c,$(BUILD_DIR)/%.o,$(C_SOURCES))
CU_OBJECTS := $(patsubst $(CORE_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SOURCES))
PY_OBJECTS := $(patsubst $(PYTHON_DIR)/%.c,$(BUILD_DIR)/%.o,$(PY_SOURCES))

ALL_OBJECTS := $(C_OBJECTS) $(CU_OBJECTS) $(PY_OBJECTS)

# CUDA Architecture Support - Multi-GPU Compatibility
CUDA_ARCH := -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_86,code=sm_86 \
             -gencode arch=compute_89,code=sm_89 \
             -gencode arch=compute_90,code=sm_90

# C compiler flags
CFLAGS := -O3 -fPIC -std=c11 -Wall -Wextra
CFLAGS += -I$(CORE_DIR) $(PYTHON_INCLUDES)
CFLAGS += -I/opt/cuda/include

# NVCC flags
NVCCFLAGS := -O3 --shared -Xcompiler -fPIC
NVCCFLAGS += -I$(CORE_DIR) $(PYTHON_INCLUDES)
NVCCFLAGS += $(CUDA_ARCH)

# Linker flags
LDFLAGS := -shared
LDFLAGS += -L/opt/cuda/lib64 -lcudart
LDFLAGS += $(PYTHON_LDFLAGS)

# Build Targets
.PHONY: all clean help test install config

all: $(TARGET)

# Link all object files into final shared library
$(TARGET): $(ALL_OBJECTS)
	@echo "Linking $@..."
	$(NVCC) $(LDFLAGS) -o $@ $^
	@echo ""
	@echo "Build complete: $@"
	@echo ""
	@echo "Usage:"
	@echo "  python3 -c 'import recpulse_cuda; print(recpulse_cuda.zeros([2, 3]))'"
	@echo ""

# Compile C source files
$(BUILD_DIR)/%.o: $(CORE_DIR)/%.c | $(BUILD_DIR)
	@echo "Compiling C: $<"
	$(CC) $(CFLAGS) -c $< -o $@

# Compile CUDA source files
$(BUILD_DIR)/%.o: $(CORE_DIR)/%.cu | $(BUILD_DIR)
	@echo "Compiling CUDA: $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile Python binding source
$(BUILD_DIR)/%.o: $(PYTHON_DIR)/%.c | $(BUILD_DIR)
	@echo "Compiling Python binding: $<"
	$(CC) $(CFLAGS) -c $< -o $@

# Create build directory
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR) $(TARGET)
	@echo "Clean complete"

# Install to Python site-packages
install: $(TARGET)
	@echo "Installing $(TARGET)..."
	cp $(TARGET) $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('platlib'))")
	@echo "Installation complete"
	@echo "You can now run: python3 -c 'import recpulse_cuda'"

# Run test script
test: $(TARGET)
	@echo "Running tests..."
	@cd .. && $(PYTHON) -c "import sys; sys.path.insert(0, 'cuda_utils'); import test_tensor"

# Show configuration
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

# Show help
help:
	@echo "RecPulse CUDA Tensor Library - Makefile Help"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build the Python extension module (default)"
	@echo "  clean    - Remove all build artifacts"
	@echo "  install  - Install module to Python site-packages"
	@echo "  test     - Run test script"
	@echo "  config   - Show build configuration"
	@echo "  help     - Show this help message"
	@echo ""
	@echo "Quick Start:"
	@echo "  make              # Build the module"
	@echo "  make test         # Run tests"
	@echo "  make install      # Install system-wide"
	@echo ""
	@echo "Development Tips:"
	@echo "  - For faster builds, edit CUDA_ARCH to target only your GPU"
	@echo "  - Uncomment debug flags in CFLAGS/NVCCFLAGS for debugging"
	@echo "  - Use 'make clean' before rebuilding after header changes"
	@echo ""
