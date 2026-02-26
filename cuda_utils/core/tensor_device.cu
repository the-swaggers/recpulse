#include "tensor.h"
#include "cuda_helpers.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "half_precision.h"

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

    // Declare variables before any goto statements
    int stride = 1;
    size_t data_size = total_elements * dtype_size(dtype);

    tensor->shape = (int*)malloc(ndim * sizeof(int));
    if (!tensor->shape) goto cleanup;
    memcpy(tensor->shape, shape, ndim * sizeof(int));

    tensor->strides = (int*)malloc(ndim * sizeof(int));
    if (!tensor->strides) goto cleanup;

    for (int i = ndim - 1; i >= 0; i--) {
        tensor->strides[i] = stride;
        stride *= shape[i];
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) goto cleanup;
    if (!check_cuda_call(cudaMalloc(&tensor->data, data_size), "cudaMalloc")) goto cleanup;
    if (!check_cuda_call(cudaMemset(tensor->data, 0, data_size), "cudaMemset")) goto cleanup;

    tensor->dtype = dtype;
    tensor->ndim = ndim;
    tensor->size = total_elements;
    tensor->device_id = device_id;
    tensor->owns_data = true;
    tensor->base_tensor = NULL;

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

static int device_convert_dtype(void* dst, DType dst_dtype, const void* src, DType src_dtype, size_t size) {
    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (src_dtype == DTYPE_FLOAT32 && dst_dtype == DTYPE_FLOAT64) {
        copy_value_kernel<float, double><<<blocks, threads>>>((double*)dst, (float*)src, size);
    } else if (src_dtype == DTYPE_FLOAT64 && dst_dtype == DTYPE_FLOAT32) {
        copy_value_kernel<double, float><<<blocks, threads>>>((float*)dst, (double*)src, size);
    } else if (src_dtype == DTYPE_FLOAT32 && dst_dtype == DTYPE_FLOAT16) {
        copy_value_kernel<float, __half><<<blocks, threads>>>((__half*)dst, (float*)src, size);
    } else if (src_dtype == DTYPE_FLOAT16 && dst_dtype == DTYPE_FLOAT32) {
        copy_value_kernel<__half, float><<<blocks, threads>>>((float*)dst, (__half*)src, size);
    } else if (src_dtype == DTYPE_FLOAT32 && dst_dtype == DTYPE_BFLOAT16) {
        copy_value_kernel<float, __nv_bfloat16><<<blocks, threads>>>((__nv_bfloat16*)dst, (float*)src, size);
    } else if (src_dtype == DTYPE_BFLOAT16 && dst_dtype == DTYPE_FLOAT32) {
        copy_value_kernel<__nv_bfloat16, float><<<blocks, threads>>>((float*)dst, (__nv_bfloat16*)src, size);
    } else if (src_dtype == DTYPE_FLOAT64 && dst_dtype == DTYPE_FLOAT16) {
        copy_value_kernel<double, __half><<<blocks, threads>>>((__half*)dst, (double*)src, size);
    } else if (src_dtype == DTYPE_FLOAT16 && dst_dtype == DTYPE_FLOAT64) {
        copy_value_kernel<__half, double><<<blocks, threads>>>((double*)dst, (__half*)src, size);
    } else if (src_dtype == DTYPE_FLOAT64 && dst_dtype == DTYPE_BFLOAT16) {
        copy_value_kernel<double, __nv_bfloat16><<<blocks, threads>>>((__nv_bfloat16*)dst, (double*)src, size);
    } else if (src_dtype == DTYPE_BFLOAT16 && dst_dtype == DTYPE_FLOAT64) {
        copy_value_kernel<__nv_bfloat16, double><<<blocks, threads>>>((double*)dst, (__nv_bfloat16*)src, size);
    } else if (src_dtype == DTYPE_FLOAT16 && dst_dtype == DTYPE_BFLOAT16) {
        copy_value_kernel<__half, __nv_bfloat16><<<blocks, threads>>>((__nv_bfloat16*)dst, (__half*)src, size);
    } else if (src_dtype == DTYPE_BFLOAT16 && dst_dtype == DTYPE_FLOAT16) {
        copy_value_kernel<__nv_bfloat16, __half><<<blocks, threads>>>((__half*)dst, (__nv_bfloat16*)src, size);
    } else {
        return -1;
    }
    return check_cuda_kernel() ? 0 : -1;
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
    } else if (tensor->dtype == DTYPE_FLOAT64) {
        fill_value_kernel<double><<<num_blocks, threads_per_block>>>((double*)tensor->data, tensor->size, value);
    } else if (tensor->dtype == DTYPE_FLOAT16) {
        __half h_value = __float2half((float)value);
        fill_value_kernel<__half><<<num_blocks, threads_per_block>>>((__half*)tensor->data, tensor->size, h_value);
    } else if (tensor->dtype == DTYPE_BFLOAT16) {
        __nv_bfloat16 b_value = __float2bfloat16((float)value);
        fill_value_kernel<__nv_bfloat16><<<num_blocks, threads_per_block>>>((__nv_bfloat16*)tensor->data, tensor->size, b_value);
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

    // Declare variables before any goto statements
    int stride = 1;
    size_t data_size = total_elements * dtype_size(dtype);

    tensor->shape = (int*)malloc(ndim * sizeof(int));
    if (!tensor->shape) goto cleanup;
    memcpy(tensor->shape, shape, ndim * sizeof(int));

    tensor->strides = (int*)malloc(ndim * sizeof(int));
    if (!tensor->strides) goto cleanup;

    for (int i = ndim - 1; i >= 0; i--) {
        tensor->strides[i] = stride;
        stride *= shape[i];
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) goto cleanup;
    if (!check_cuda_call(cudaMalloc(&tensor->data, data_size), "cudaMalloc")) goto cleanup;

    tensor->dtype = dtype;
    tensor->ndim = ndim;
    tensor->size = total_elements;
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

Tensor* values_device_tensor(void* vals, DType vals_dtype, DType target_dtype, int source_device_id, int ndim, int* shape, int device_id, Meta* metadata) {
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

    // Declare variables before any goto statements
    int stride = 1;
    size_t data_size = total_elements * dtype_size(target_dtype);

    tensor->shape = (int*)malloc(ndim * sizeof(int));
    if (!tensor->shape) goto cleanup;
    memcpy(tensor->shape, shape, ndim * sizeof(int));

    tensor->strides = (int*)malloc(ndim * sizeof(int));
    if (!tensor->strides) goto cleanup;

    for (int i = ndim - 1; i >= 0; i--) {
        tensor->strides[i] = stride;
        stride *= shape[i];
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) goto cleanup;
    if (!check_cuda_call(cudaMalloc(&tensor->data, data_size), "cudaMalloc")) goto cleanup;

    tensor->dtype = target_dtype;
    tensor->ndim = ndim;
    tensor->size = total_elements;
    tensor->device_id = device_id;
    tensor->owns_data = true;

    if (source_device_id == -1) {
        if (vals_dtype == target_dtype) {
            if (!check_cuda_call(cudaMemcpy(tensor->data, vals, data_size, cudaMemcpyHostToDevice), "cudaMemcpy")) goto cleanup;
        } else {
            size_t src_data_size = total_elements * dtype_size(vals_dtype);
            if (!check_cuda_call(cudaMalloc(&device_src, src_data_size), "cudaMalloc")) goto cleanup;
            if (!check_cuda_call(cudaMemcpy(device_src, vals, src_data_size, cudaMemcpyHostToDevice), "cudaMemcpy")) goto cleanup;
            if (device_convert_dtype(tensor->data, target_dtype, device_src, vals_dtype, total_elements) != 0) goto cleanup;
            cudaFree(device_src);
            device_src = NULL;
        }
    } else {
        if (vals_dtype == target_dtype) {
            if (!check_cuda_call(cudaMemcpy(tensor->data, vals, data_size, cudaMemcpyDeviceToDevice), "cudaMemcpy")) goto cleanup;
        } else {
            if (device_convert_dtype(tensor->data, target_dtype, vals, vals_dtype, total_elements) != 0) goto cleanup;
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
    if (tensor->device_id < 0) return;
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

Tensor* tensor_copy_device(Tensor* tensor, int device_id, DType target_dtype) {
    if (tensor->device_id < 0) return NULL;

    Tensor* copy = (Tensor*)malloc(sizeof(Tensor));
    if (!copy) return NULL;

    copy->shape = NULL;
    copy->strides = NULL;
    copy->data = NULL;
    copy->metadata = NULL;
    void* temp_buffer = NULL;

    // Declare variables before any goto statements
    size_t data_size = tensor->size * dtype_size(target_dtype);
    bool same_gpu = (tensor->device_id == device_id);

    copy->shape = (int*)malloc(tensor->ndim * sizeof(int));
    if (!copy->shape) goto cleanup;
    memcpy(copy->shape, tensor->shape, tensor->ndim * sizeof(int));

    copy->strides = (int*)malloc(tensor->ndim * sizeof(int));
    if (!copy->strides) goto cleanup;
    memcpy(copy->strides, tensor->strides, tensor->ndim * sizeof(int));

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) goto cleanup;
    if (!check_cuda_call(cudaMalloc(&copy->data, data_size), "cudaMalloc")) goto cleanup;

    if (same_gpu) {
        if (tensor->dtype == target_dtype) {
            if (!check_cuda_call(cudaMemcpy(copy->data, tensor->data, data_size, cudaMemcpyDeviceToDevice), "cudaMemcpy")) goto cleanup;
        } else {
            if (device_convert_dtype(copy->data, target_dtype, tensor->data, tensor->dtype, tensor->size) != 0) goto cleanup;
        }
    } else {
        cudaError_t p2p_err = cudaDeviceEnablePeerAccess(tensor->device_id, 0);
        if (p2p_err != cudaSuccess && p2p_err != cudaErrorPeerAccessAlreadyEnabled) {
            cudaGetLastError();
        }

        if (tensor->dtype == target_dtype) {
            if (!check_cuda_call(cudaMemcpyPeer(copy->data, device_id, tensor->data, tensor->device_id, data_size), "cudaMemcpyPeer")) goto cleanup;
        } else {
            size_t src_data_size = tensor->size * dtype_size(tensor->dtype);
            if (!check_cuda_call(cudaMalloc(&temp_buffer, src_data_size), "cudaMalloc temp")) goto cleanup;
            if (!check_cuda_call(cudaMemcpyPeer(temp_buffer, device_id, tensor->data, tensor->device_id, src_data_size), "cudaMemcpyPeer")) goto cleanup;
            if (device_convert_dtype(copy->data, target_dtype, temp_buffer, tensor->dtype, tensor->size) != 0) goto cleanup;
            cudaFree(temp_buffer);
            temp_buffer = NULL;
        }
    }

    copy->dtype = target_dtype;
    copy->ndim = tensor->ndim;
    copy->size = tensor->size;
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
}

Tensor* move_host_to_device(Tensor* tensor, int device_id, DType target_dtype) {
    if (tensor->device_id != -1) return NULL;

    Tensor* result = (Tensor*)malloc(sizeof(Tensor));
    if (!result) return NULL;

    result->shape = NULL;
    result->strides = NULL;
    result->data = NULL;
    result->metadata = NULL;
    void* temp_buffer = NULL;

    // Declare variables before any goto statements
    size_t data_size = tensor->size * dtype_size(target_dtype);

    result->shape = (int*)malloc(tensor->ndim * sizeof(int));
    if (!result->shape) goto cleanup;
    memcpy(result->shape, tensor->shape, tensor->ndim * sizeof(int));

    result->strides = (int*)malloc(tensor->ndim * sizeof(int));
    if (!result->strides) goto cleanup;
    memcpy(result->strides, tensor->strides, tensor->ndim * sizeof(int));

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) goto cleanup;
    if (!check_cuda_call(cudaMalloc(&result->data, data_size), "cudaMalloc")) goto cleanup;

    if (tensor->dtype == target_dtype) {
        if (!check_cuda_call(cudaMemcpy(result->data, tensor->data, data_size, cudaMemcpyHostToDevice), "cudaMemcpy")) goto cleanup;
    } else {
        size_t src_data_size = tensor->size * dtype_size(tensor->dtype);
        if (!check_cuda_call(cudaMalloc(&temp_buffer, src_data_size), "cudaMalloc temp")) goto cleanup;
        if (!check_cuda_call(cudaMemcpy(temp_buffer, tensor->data, src_data_size, cudaMemcpyHostToDevice), "cudaMemcpy")) goto cleanup;
        if (device_convert_dtype(result->data, target_dtype, temp_buffer, tensor->dtype, tensor->size) != 0) goto cleanup;
        cudaFree(temp_buffer);
        temp_buffer = NULL;
    }

    result->dtype = target_dtype;
    result->ndim = tensor->ndim;
    result->size = tensor->size;
    result->device_id = device_id;
    result->owns_data = true;

    if (tensor->metadata) {
        result->metadata = (Meta*)malloc(sizeof(Meta));
        if (result->metadata) {
            memcpy(result->metadata, tensor->metadata, sizeof(Meta));
            result->metadata->grad = NULL;
            result->metadata->grad_fn = NULL;
        }
    }

    return result;

cleanup:
    if (temp_buffer) cudaFree(temp_buffer);
    if (result->data) cudaFree(result->data);
    if (result->strides) free(result->strides);
    if (result->shape) free(result->shape);
    free(result);
    return NULL;
}

Tensor* move_device_to_host(Tensor* tensor, DType target_dtype) {
    if (tensor->device_id < 0) return NULL;

    Tensor* result = (Tensor*)malloc(sizeof(Tensor));
    if (!result) return NULL;

    result->shape = NULL;
    result->strides = NULL;
    result->data = NULL;
    result->metadata = NULL;
    void* temp_buffer = NULL;

    size_t data_size = tensor->size * dtype_size(target_dtype);
    float* fp32_buf = NULL;
    bool need_free_fp32 = false;

    result->shape = (int*)malloc(tensor->ndim * sizeof(int));
    if (!result->shape) goto cleanup;
    memcpy(result->shape, tensor->shape, tensor->ndim * sizeof(int));

    result->strides = (int*)malloc(tensor->ndim * sizeof(int));
    if (!result->strides) goto cleanup;
    memcpy(result->strides, tensor->strides, tensor->ndim * sizeof(int));

    result->data = malloc(data_size);
    if (!result->data) goto cleanup;

    if (!check_cuda_call(cudaSetDevice(tensor->device_id), "cudaSetDevice")) goto cleanup;

    if (tensor->dtype == target_dtype) {
        if (!check_cuda_call(cudaMemcpy(result->data, tensor->data, data_size, cudaMemcpyDeviceToHost), "cudaMemcpy")) goto cleanup;
    } else {
        size_t src_data_size = tensor->size * dtype_size(tensor->dtype);
        temp_buffer = malloc(src_data_size);
        if (!temp_buffer) goto cleanup;
        if (!check_cuda_call(cudaMemcpy(temp_buffer, tensor->data, src_data_size, cudaMemcpyDeviceToHost), "cudaMemcpy")) goto cleanup;

        if (tensor->dtype == DTYPE_FLOAT32) {
            fp32_buf = (float*)temp_buffer;
        } else if (tensor->dtype == DTYPE_FLOAT64) {
            fp32_buf = (float*)malloc(tensor->size * sizeof(float));
            if (!fp32_buf) goto cleanup;
            need_free_fp32 = true;
            double* src = (double*)temp_buffer;
            for (size_t i = 0; i < tensor->size; i++) fp32_buf[i] = (float)src[i];
        } else {
            fp32_buf = (float*)malloc(tensor->size * sizeof(float));
            if (!fp32_buf) goto cleanup;
            need_free_fp32 = true;
            half_to_fp32_array(temp_buffer, fp32_buf, tensor->size, tensor->dtype);
        }

        if (target_dtype == DTYPE_FLOAT32) {
            memcpy(result->data, fp32_buf, tensor->size * sizeof(float));
        } else if (target_dtype == DTYPE_FLOAT64) {
            double* dst = (double*)result->data;
            for (size_t i = 0; i < tensor->size; i++) dst[i] = (double)fp32_buf[i];
        } else {
            fp32_to_half_array(fp32_buf, result->data, tensor->size, target_dtype);
        }

        if (need_free_fp32) free(fp32_buf);
        free(temp_buffer);
        temp_buffer = NULL;
    }

    result->dtype = target_dtype;
    result->ndim = tensor->ndim;
    result->size = tensor->size;
    result->device_id = -1;
    result->owns_data = true;

    if (tensor->metadata) {
        result->metadata = (Meta*)malloc(sizeof(Meta));
        if (result->metadata) {
            memcpy(result->metadata, tensor->metadata, sizeof(Meta));
            result->metadata->grad = NULL;
            result->metadata->grad_fn = NULL;
        }
    }

    return result;

cleanup:
    if (temp_buffer) free(temp_buffer);
    if (result->data) free(result->data);
    if (result->strides) free(result->strides);
    if (result->shape) free(result->shape);
    free(result);
    return NULL;
}

Tensor* tensor_reshape_device(Tensor* tensor, int new_ndim, int* new_shape) {
    if (tensor->device_id < 0) return NULL;

    Tensor* reshaped = (Tensor*)malloc(sizeof(Tensor));
    if (!reshaped) return NULL;

    reshaped->shape = NULL;
    reshaped->strides = NULL;
    reshaped->data = NULL;
    reshaped->metadata = NULL;

    int stride = 1;
    size_t data_size = tensor->size * dtype_size(tensor->dtype);

    reshaped->shape = (int*)malloc(new_ndim * sizeof(int));
    if (!reshaped->shape) goto cleanup;
    memcpy(reshaped->shape, new_shape, new_ndim * sizeof(int));

    reshaped->strides = (int*)malloc(new_ndim * sizeof(int));
    if (!reshaped->strides) goto cleanup;

    for (int i = new_ndim - 1; i >= 0; i--) {
        reshaped->strides[i] = stride;
        stride *= new_shape[i];
    }

    if (!check_cuda_call(cudaSetDevice(tensor->device_id), "cudaSetDevice")) goto cleanup;
    if (!check_cuda_call(cudaMalloc(&reshaped->data, data_size), "cudaMalloc")) goto cleanup;
    if (!check_cuda_call(cudaMemcpy(reshaped->data, tensor->data, data_size, cudaMemcpyDeviceToDevice), "cudaMemcpy")) goto cleanup;

    reshaped->dtype = tensor->dtype;
    reshaped->ndim = new_ndim;
    reshaped->size = tensor->size;
    reshaped->device_id = tensor->device_id;
    reshaped->owns_data = true;
    reshaped->base_tensor = NULL;

    if (tensor->metadata) {
        reshaped->metadata = (Meta*)malloc(sizeof(Meta));
        if (reshaped->metadata) {
            memcpy(reshaped->metadata, tensor->metadata, sizeof(Meta));
            reshaped->metadata->grad = NULL;
            reshaped->metadata->grad_fn = NULL;
        }
    }

    return reshaped;

cleanup:
    if (reshaped->data) cudaFree(reshaped->data);
    if (reshaped->strides) free(reshaped->strides);
    if (reshaped->shape) free(reshaped->shape);
    free(reshaped);
    return NULL;
}
