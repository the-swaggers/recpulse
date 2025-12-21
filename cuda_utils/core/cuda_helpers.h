#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include <stdbool.h>
#include <stddef.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

bool check_cuda_call(cudaError_t err, const char* operation);
bool check_cuda_kernel();
bool check_shape_valid(int ndim, int* shape);
bool safe_multiply_size(size_t a, size_t b, size_t* result);
bool calculate_total_elements(int ndim, int* shape, size_t* total);

#ifdef __cplusplus
}
#endif

#endif
