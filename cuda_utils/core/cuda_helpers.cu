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

Tensor* move_device_to_device(Tensor* tensor, int device_id, DType target_dtype) {
    if (tensor->device != DEVICE) return NULL;

    Tensor* copy = (Tensor*)malloc(sizeof(Tensor));
    if (!copy) return NULL;

    copy->shape = NULL;
    copy->strides = NULL;
    copy->data = NULL;
    copy->metadata = NULL;

    copy->shape = (int*)malloc(tensor->ndim * sizeof(int));
    if (!copy->shape) goto cleanup;
    memcpy(copy->shape, tensor->shape, tensor->ndim * sizeof(int));

    copy->strides = (int*)malloc(tensor->ndim * sizeof(int));
    if (!copy->strides) goto cleanup;
    memcpy(copy->strides, tensor->strides, tensor->ndim * sizeof(int));

    size_t target_dtype_size = (target_dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);
    size_t data_size = tensor->size * target_dtype_size;

    if (!check_cuda_call(cudaMalloc(&copy->data, data_size), "cudaMalloc")) goto cleanup;

    if (tensor->dtype == target_dtype) {
        memcpy(copy->data, tensor->data, data_size);
    } else if (tensor->dtype == DTYPE_FLOAT32 && target_dtype == DTYPE_FLOAT64) {
        float* src = (float*)tensor->data;
        double* dst = (double*)copy->data;
        for (size_t i = 0; i < tensor->size; i++) {
            dst[i] = (double)src[i];
        }
    } else if (tensor->dtype == DTYPE_FLOAT64 && target_dtype == DTYPE_FLOAT32) {
        double* src = (double*)tensor->data;
        float* dst = (float*)copy->data;
        for (size_t i = 0; i < tensor->size; i++) {
            dst[i] = (float)src[i];
        }
    }

    copy->dtype = target_dtype;
    copy->ndim = tensor->ndim;
    copy->size = tensor->size;
    copy->device = HOST;
    copy->device_id = 0;
    copy->owns_data = true;

    if (tensor->metadata) {
        copy->metadata = (Meta*)malloc(sizeof(Meta));
        if (copy->metadata) {
            memcpy(copy->metadata, tensor->metadata, sizeof(Meta));
            copy->metadata->grad = NULL;
            copy->metadata->grad_fn = NULL;
        }
    }

    return copy;

cleanup:
    if (copy->data) cudaFree(copy->data);
    if (copy->strides) free(copy->strides);
    if (copy->shape) free(copy->shape);
    free(copy);
    return NULL;
}

