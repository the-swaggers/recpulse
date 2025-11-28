#include "tensor.h"
#include "cuda_helpers.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <cuda_runtime.h>

Tensor* zeros_device_tensor(DType dtype, int device_id, int ndim, int* shape, Meta* metadata) {
    if (!check_shape_valid(ndim, shape)) return NULL;

    size_t total_elements;
    if (!calculate_total_elements(ndim, shape, &total_elements)) return NULL;

    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) return NULL;

    tensor->shape = (int*)malloc(ndim * sizeof(int));
    if (!tensor->shape) {
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, ndim * sizeof(int));

    tensor->strides = (int*)malloc(ndim * sizeof(int));
    if (!tensor->strides) {
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        tensor->strides[i] = stride;
        stride *= shape[i];
    }

    size_t dtype_size = (dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);
    size_t data_size = total_elements * dtype_size;

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) {
        free(tensor->strides);
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    if (!check_cuda_call(cudaMalloc(&tensor->data, data_size), "cudaMalloc")) {
        free(tensor->strides);
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    if (!check_cuda_call(cudaMemset(tensor->data, 0, data_size), "cudaMemset")) {
        cudaFree(tensor->data);
        free(tensor->strides);
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    tensor->dtype = dtype;
    tensor->ndim = ndim;
    tensor->size = total_elements;
    tensor->device = DEVICE;
    tensor->device_id = device_id;
    tensor->owns_data = true;
    tensor->metadata = metadata;

    return tensor;
}


template<typename T>
__global__ void fill_value_kernel(T* data, size_t size, T value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

template<typename SrcType, typename DstType>
__global__ void copy_value_kernel(DstType* dst, SrcType* src, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = (DstType)src[idx];
    }
}

Tensor* fill_value_device_tensor(double value, Tensor* tensor){
    if (!tensor || !tensor->data) return NULL;

    if (!check_cuda_call(cudaSetDevice(tensor->device_id), "cudaSetDevice")) {
        return NULL;
    }

    size_t threads_per_block = 256;
    size_t num_blocks = (tensor->size + threads_per_block - 1) / threads_per_block;

    if (tensor->dtype == DTYPE_FLOAT32) {
        float f_value = (float)value;
        fill_value_kernel<float><<<num_blocks, threads_per_block>>>((float*)tensor->data, tensor->size, f_value);
    } else {
        fill_value_kernel<double><<<num_blocks, threads_per_block>>>((double*)tensor->data, tensor->size, value);
    }

    if (!check_cuda_kernel()) return NULL;

    return tensor;
}

Tensor* ones_device_tensor(DType dtype, int device_id, int ndim, int* shape, Meta* metadata) {
    if (!check_shape_valid(ndim, shape)) return NULL;

    size_t total_elements;
    if (!calculate_total_elements(ndim, shape, &total_elements)) return NULL;

    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) return NULL;

    tensor->shape = (int*)malloc(ndim * sizeof(int));
    if (!tensor->shape) {
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, ndim * sizeof(int));

    tensor->strides = (int*)malloc(ndim * sizeof(int));
    if (!tensor->strides) {
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        tensor->strides[i] = stride;
        stride *= shape[i];
    }

    size_t dtype_size = (dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);
    size_t data_size = total_elements * dtype_size;

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) {
        free(tensor->strides);
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    if (!check_cuda_call(cudaMalloc(&tensor->data, data_size), "cudaMalloc")) {
        free(tensor->strides);
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    tensor->dtype = dtype;
    tensor->ndim = ndim;
    tensor->size = total_elements;
    tensor->device = DEVICE;
    tensor->device_id = device_id;
    tensor->owns_data = true;
    tensor->metadata = metadata;

    if (!fill_value_device_tensor(1, tensor)) {
        free_tensor_device(tensor);
        return NULL;
    }

    return tensor;
}

Tensor* values_device_tensor(void* vals, DType vals_dtype, DType target_dtype, DeviceType source_device, int ndim, int* shape, int device_id, Meta* metadata) {
    if (!check_shape_valid(ndim, shape)) return NULL;

    size_t total_elements;
    if (!calculate_total_elements(ndim, shape, &total_elements)) return NULL;

    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) return NULL;

    tensor->shape = (int*)malloc(ndim * sizeof(int));
    if (!tensor->shape) {
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, ndim * sizeof(int));

    tensor->strides = (int*)malloc(ndim * sizeof(int));
    if (!tensor->strides) {
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        tensor->strides[i] = stride;
        stride *= shape[i];
    }

    size_t target_dtype_size = (target_dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);
    size_t vals_dtype_size = (vals_dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);
    size_t data_size = total_elements * target_dtype_size;

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) {
        free(tensor->strides);
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    if (!check_cuda_call(cudaMalloc(&tensor->data, data_size), "cudaMalloc")) {
        free(tensor->strides);
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    tensor->dtype = target_dtype;
    tensor->ndim = ndim;
    tensor->size = total_elements;
    tensor->device = DEVICE;
    tensor->device_id = device_id;
    tensor->owns_data = true;
    tensor->metadata = metadata;

    size_t threads_per_block = 256;
    size_t num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    if (source_device == HOST) {
        if (vals_dtype == target_dtype) {
            if (!check_cuda_call(cudaMemcpy(tensor->data, vals, data_size, cudaMemcpyHostToDevice), "cudaMemcpy")) {
                free_tensor_device(tensor);
                return NULL;
            }
        } else if (vals_dtype == DTYPE_FLOAT32 && target_dtype == DTYPE_FLOAT64) {
            float* device_src;
            if (!check_cuda_call(cudaMalloc(&device_src, total_elements * sizeof(float)), "cudaMalloc")) {
                free_tensor_device(tensor);
                return NULL;
            }

            if (!check_cuda_call(cudaMemcpy(device_src, vals, total_elements * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy")) {
                cudaFree(device_src);
                free_tensor_device(tensor);
                return NULL;
            }

            copy_value_kernel<float, double><<<num_blocks, threads_per_block>>>(
                (double*)tensor->data, device_src, total_elements
            );

            if (!check_cuda_kernel()) {
                cudaFree(device_src);
                free_tensor_device(tensor);
                return NULL;
            }

            cudaFree(device_src);
        } else if (vals_dtype == DTYPE_FLOAT64 && target_dtype == DTYPE_FLOAT32) {
            double* device_src;
            if (!check_cuda_call(cudaMalloc(&device_src, total_elements * sizeof(double)), "cudaMalloc")) {
                free_tensor_device(tensor);
                return NULL;
            }

            if (!check_cuda_call(cudaMemcpy(device_src, vals, total_elements * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy")) {
                cudaFree(device_src);
                free_tensor_device(tensor);
                return NULL;
            }

            copy_value_kernel<double, float><<<num_blocks, threads_per_block>>>(
                (float*)tensor->data, device_src, total_elements
            );

            if (!check_cuda_kernel()) {
                cudaFree(device_src);
                free_tensor_device(tensor);
                return NULL;
            }

            cudaFree(device_src);
        }
    }
    else {
        if (vals_dtype == target_dtype) {
            if (!check_cuda_call(cudaMemcpy(tensor->data, vals, data_size, cudaMemcpyDeviceToDevice), "cudaMemcpy")) {
                free_tensor_device(tensor);
                return NULL;
            }
        } else if (vals_dtype == DTYPE_FLOAT32 && target_dtype == DTYPE_FLOAT64) {
            copy_value_kernel<float, double><<<num_blocks, threads_per_block>>>(
                (double*)tensor->data, (float*)vals, total_elements
            );

            if (!check_cuda_kernel()) {
                free_tensor_device(tensor);
                return NULL;
            }
        } else if (vals_dtype == DTYPE_FLOAT64 && target_dtype == DTYPE_FLOAT32) {
            copy_value_kernel<double, float><<<num_blocks, threads_per_block>>>(
                (float*)tensor->data, (double*)vals, total_elements
            );

            if (!check_cuda_kernel()) {
                free_tensor_device(tensor);
                return NULL;
            }
        }
    }

    return tensor;
}

void free_tensor_device(Tensor* tensor){
    
    if (!tensor) return;
    if (tensor->device != DEVICE) return;
    if (tensor->metadata){
        if (tensor->metadata->grad){
            free_tensor(tensor->metadata->grad);
        };
    }

    if (tensor->owns_data){
        cudaSetDevice(tensor->device_id);
        cudaFree(tensor->data);
    }

    if (tensor->strides) free(tensor->strides);
    if (tensor->shape) free(tensor->shape);
    if (tensor->metadata) free(tensor->metadata);

    free(tensor);
};
