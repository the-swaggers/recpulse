#include "ops.h"
#include "../core/tensor.h"
#include <cuda_runtime.h>

int backwards_add_a_device(const void* grad_c, void* grad_a, size_t size, DType dtype) {
    if (!grad_c || !grad_a) return -1;

    size_t element_size = (dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);
    size_t nbytes = size * element_size;

    cudaError_t err = cudaMemcpy(grad_a, grad_c, nbytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        return -1;
    }

    return 0;
}

int backwards_add_b_device(const void* grad_c, void* grad_b, size_t size, DType dtype) {
    if (!grad_c || !grad_b) return -1;

    size_t element_size = (dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);
    size_t nbytes = size * element_size;

    cudaError_t err = cudaMemcpy(grad_b, grad_c, nbytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        return -1;
    }

    return 0;
}
