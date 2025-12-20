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

    tensor->shape = NULL;
    tensor->strides = NULL;
    tensor->data = NULL;
    tensor->metadata = metadata;

    tensor->shape = (int*)malloc(ndim * sizeof(int));
    if (!tensor->shape) goto cleanup;
    memcpy(tensor->shape, shape, ndim * sizeof(int));

    tensor->strides = (int*)malloc(ndim * sizeof(int));
    if (!tensor->strides) goto cleanup;

    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        tensor->strides[i] = stride;
        stride *= shape[i];
    }

    size_t dtype_size = (dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);
    size_t data_size = total_elements * dtype_size;

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) goto cleanup;
    if (!check_cuda_call(cudaMalloc(&tensor->data, data_size), "cudaMalloc")) goto cleanup;
    if (!check_cuda_call(cudaMemset(tensor->data, 0, data_size), "cudaMemset")) goto cleanup;

    tensor->dtype = dtype;
    tensor->ndim = ndim;
    tensor->size = total_elements;
    tensor->device = DEVICE;
    tensor->device_id = device_id;
    tensor->owns_data = true;

    return tensor;

cleanup:
    if (tensor->data) cudaFree(tensor->data);
    if (tensor->strides) free(tensor->strides);
    if (tensor->shape) free(tensor->shape);
    free(tensor);
    return NULL;
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

    tensor->shape = NULL;
    tensor->strides = NULL;
    tensor->data = NULL;
    tensor->metadata = metadata;

    tensor->shape = (int*)malloc(ndim * sizeof(int));
    if (!tensor->shape) goto cleanup;
    memcpy(tensor->shape, shape, ndim * sizeof(int));

    tensor->strides = (int*)malloc(ndim * sizeof(int));
    if (!tensor->strides) goto cleanup;

    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        tensor->strides[i] = stride;
        stride *= shape[i];
    }

    size_t dtype_size = (dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);
    size_t data_size = total_elements * dtype_size;

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) goto cleanup;
    if (!check_cuda_call(cudaMalloc(&tensor->data, data_size), "cudaMalloc")) goto cleanup;

    tensor->dtype = dtype;
    tensor->ndim = ndim;
    tensor->size = total_elements;
    tensor->device = DEVICE;
    tensor->device_id = device_id;
    tensor->owns_data = true;

    if (!fill_value_device_tensor(1, tensor)) goto cleanup;

    return tensor;

cleanup:
    if (tensor->data) cudaFree(tensor->data);
    if (tensor->strides) free(tensor->strides);
    if (tensor->shape) free(tensor->shape);
    free(tensor);
    return NULL;
}

Tensor* values_device_tensor(void* vals, DType vals_dtype, DType target_dtype, DeviceType source_device, int ndim, int* shape, int device_id, Meta* metadata) {
    if (!check_shape_valid(ndim, shape)) return NULL;

    size_t total_elements;
    if (!calculate_total_elements(ndim, shape, &total_elements)) return NULL;

    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) return NULL;

    tensor->shape = NULL;
    tensor->strides = NULL;
    tensor->data = NULL;
    tensor->metadata = metadata;
    void* device_src = NULL;

    tensor->shape = (int*)malloc(ndim * sizeof(int));
    if (!tensor->shape) goto cleanup;
    memcpy(tensor->shape, shape, ndim * sizeof(int));

    tensor->strides = (int*)malloc(ndim * sizeof(int));
    if (!tensor->strides) goto cleanup;

    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        tensor->strides[i] = stride;
        stride *= shape[i];
    }

    size_t target_dtype_size = (target_dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);
    size_t vals_dtype_size = (vals_dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);
    size_t data_size = total_elements * target_dtype_size;

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) goto cleanup;
    if (!check_cuda_call(cudaMalloc(&tensor->data, data_size), "cudaMalloc")) goto cleanup;

    tensor->dtype = target_dtype;
    tensor->ndim = ndim;
    tensor->size = total_elements;
    tensor->device = DEVICE;
    tensor->device_id = device_id;
    tensor->owns_data = true;

    size_t threads_per_block = 256;
    size_t num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    if (source_device == HOST) {
        if (vals_dtype == target_dtype) {
            if (!check_cuda_call(cudaMemcpy(tensor->data, vals, data_size, cudaMemcpyHostToDevice), "cudaMemcpy")) goto cleanup;
        } else if (vals_dtype == DTYPE_FLOAT32 && target_dtype == DTYPE_FLOAT64) {
            if (!check_cuda_call(cudaMalloc(&device_src, total_elements * sizeof(float)), "cudaMalloc")) goto cleanup;
            if (!check_cuda_call(cudaMemcpy(device_src, vals, total_elements * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy")) goto cleanup;

            copy_value_kernel<float, double><<<num_blocks, threads_per_block>>>((double*)tensor->data, (float*)device_src, total_elements);
            if (!check_cuda_kernel()) goto cleanup;

            cudaFree(device_src);
            device_src = NULL;
        } else if (vals_dtype == DTYPE_FLOAT64 && target_dtype == DTYPE_FLOAT32) {
            if (!check_cuda_call(cudaMalloc(&device_src, total_elements * sizeof(double)), "cudaMalloc")) goto cleanup;
            if (!check_cuda_call(cudaMemcpy(device_src, vals, total_elements * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy")) goto cleanup;

            copy_value_kernel<double, float><<<num_blocks, threads_per_block>>>((float*)tensor->data, (double*)device_src, total_elements);
            if (!check_cuda_kernel()) goto cleanup;

            cudaFree(device_src);
            device_src = NULL;
        }
    } else {
        if (vals_dtype == target_dtype) {
            if (!check_cuda_call(cudaMemcpy(tensor->data, vals, data_size, cudaMemcpyDeviceToDevice), "cudaMemcpy")) goto cleanup;
        } else if (vals_dtype == DTYPE_FLOAT32 && target_dtype == DTYPE_FLOAT64) {
            copy_value_kernel<float, double><<<num_blocks, threads_per_block>>>((double*)tensor->data, (float*)vals, total_elements);
            if (!check_cuda_kernel()) goto cleanup;
        } else if (vals_dtype == DTYPE_FLOAT64 && target_dtype == DTYPE_FLOAT32) {
            copy_value_kernel<double, float><<<num_blocks, threads_per_block>>>((float*)tensor->data, (double*)vals, total_elements);
            if (!check_cuda_kernel()) goto cleanup;
        }
    }

    return tensor;

cleanup:
    if (device_src) cudaFree(device_src);
    if (tensor->data) cudaFree(tensor->data);
    if (tensor->strides) free(tensor->strides);
    if (tensor->shape) free(tensor->shape);
    free(tensor);
    return NULL;
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

Tensor* move_device_to_device(Tensor* tensor, int device_id, DType target_dtype) {
    if (tensor->device != DEVICE) return NULL;

    Tensor* copy = (Tensor*)malloc(sizeof(Tensor));
    if (!copy) return NULL;

    copy->shape = NULL;
    copy->strides = NULL;
    copy->data = NULL;
    copy->metadata = NULL;
    void* temp_buffer = NULL;

    copy->shape = (int*)malloc(tensor->ndim * sizeof(int));
    if (!copy->shape) goto cleanup;
    memcpy(copy->shape, tensor->shape, tensor->ndim * sizeof(int));

    copy->strides = (int*)malloc(tensor->ndim * sizeof(int));
    if (!copy->strides) goto cleanup;
    memcpy(copy->strides, tensor->strides, tensor->ndim * sizeof(int));

    size_t target_dtype_size = (target_dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);
    size_t data_size = tensor->size * target_dtype_size;

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) goto cleanup;
    if (!check_cuda_call(cudaMalloc(&copy->data, data_size), "cudaMalloc")) goto cleanup;

    bool same_gpu = (tensor->device_id == device_id);
    size_t threads_per_block = 256;
    size_t num_blocks = (tensor->size + threads_per_block - 1) / threads_per_block;

    if (same_gpu) {
        if (tensor->dtype == target_dtype) {
            if (!check_cuda_call(cudaMemcpy(copy->data, tensor->data, data_size, cudaMemcpyDeviceToDevice), "cudaMemcpy")) goto cleanup;
        } else if (tensor->dtype == DTYPE_FLOAT32 && target_dtype == DTYPE_FLOAT64) {
            copy_value_kernel<float, double><<<num_blocks, threads_per_block>>>((double*)copy->data, (float*)tensor->data, tensor->size);
            if (!check_cuda_kernel()) goto cleanup;
        } else if (tensor->dtype == DTYPE_FLOAT64 && target_dtype == DTYPE_FLOAT32) {
            copy_value_kernel<double, float><<<num_blocks, threads_per_block>>>((float*)copy->data, (double*)tensor->data, tensor->size);
            if (!check_cuda_kernel()) goto cleanup;
        }
    } else {
        cudaError_t p2p_err = cudaDeviceEnablePeerAccess(tensor->device_id, 0);
        if (p2p_err != cudaSuccess && p2p_err != cudaErrorPeerAccessAlreadyEnabled) {
            cudaGetLastError();
        }

        if (tensor->dtype == target_dtype) {
            if (!check_cuda_call(cudaMemcpyPeer(copy->data, device_id, tensor->data, tensor->device_id, data_size), "cudaMemcpyPeer")) goto cleanup;
        } else {
            size_t src_dtype_size = (tensor->dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);
            size_t src_data_size = tensor->size * src_dtype_size;

            if (!check_cuda_call(cudaMalloc(&temp_buffer, src_data_size), "cudaMalloc temp")) goto cleanup;
            if (!check_cuda_call(cudaMemcpyPeer(temp_buffer, device_id, tensor->data, tensor->device_id, src_data_size), "cudaMemcpyPeer")) goto cleanup;

            if (tensor->dtype == DTYPE_FLOAT32 && target_dtype == DTYPE_FLOAT64) {
                copy_value_kernel<float, double><<<num_blocks, threads_per_block>>>((double*)copy->data, (float*)temp_buffer, tensor->size);
            } else if (tensor->dtype == DTYPE_FLOAT64 && target_dtype == DTYPE_FLOAT32) {
                copy_value_kernel<double, float><<<num_blocks, threads_per_block>>>((float*)copy->data, (double*)temp_buffer, tensor->size);
            }

            if (!check_cuda_kernel()) goto cleanup;

            cudaFree(temp_buffer);
            temp_buffer = NULL;
        }
    }

    copy->dtype = target_dtype;
    copy->ndim = tensor->ndim;
    copy->size = tensor->size;
    copy->device = DEVICE;
    copy->device_id = device_id;
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
    if (temp_buffer) cudaFree(temp_buffer);
    if (copy->data) cudaFree(copy->data);
    if (copy->strides) free(copy->strides);
    if (copy->shape) free(copy->shape);
    free(copy);
    return NULL;
