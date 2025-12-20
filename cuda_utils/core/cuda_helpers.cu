#include "cuda_helpers.h"
#include <stdio.h>
#include <limits.h>

bool check_cuda_call(cudaError_t err, const char* operation) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool check_cuda_kernel() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool check_shape_valid(int ndim, int* shape) {
    if (ndim <= 0 || !shape) return false;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) return false;
    }
    return true;
}

bool safe_multiply_size(size_t a, size_t b, size_t* result) {
    if (b != 0 && a > SIZE_MAX / b) return false;
    *result = a * b;
    return true;
}

bool calculate_total_elements(int ndim, int* shape, size_t* total) {
    *total = 1;
    for (int i = 0; i < ndim; i++) {
        if (!safe_multiply_size(*total, (size_t)shape[i], total)) {
            fprintf(stderr, "Tensor size overflow\n");
            return false;
        }
    }
    return true;
}
