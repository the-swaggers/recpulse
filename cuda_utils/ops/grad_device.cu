#include "ops.h"
#include "../core/tensor.h"
#include <cuda_runtime.h>

int backwards_add_x1_device(const void* grad_c, void* grad_x1, size_t size, DType dtype) {
    if (!grad_c || !grad_x1) return -1;

    size_t element_size = (dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);
    size_t nbytes = size * element_size;

    cudaError_t err = cudaMemcpy(grad_x1, grad_c, nbytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        return -1;
    }

    return 0;
}

int backwards_add_x2_device(const void* grad_c, void* grad_x2, size_t size, DType dtype) {
    if (!grad_c || !grad_x2) return -1;

    size_t element_size = (dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);
    size_t nbytes = size * element_size;

    cudaError_t err = cudaMemcpy(grad_x2, grad_c, nbytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        return -1;
    }

    return 0;
}

template<typename T>
__global__ void backwards_mul_kernel(const T* grad_c, const T* x, T* grad_out, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_out[idx] = grad_c[idx] * x[idx];
    }
}

int backwards_mul_x1_device(const void* grad_c, const void* x2, void* grad_x1, size_t size, DType dtype) {
    if (!grad_c || !x2 || !grad_x1) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_mul_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (const float*)x2, (float*)grad_x1, size);
    } else {
        backwards_mul_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)x2, (double*)grad_x1, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

int backwards_mul_x2_device(const void* grad_c, const void* x1, void* grad_x2, size_t size, DType dtype) {
    if (!grad_c || !x1 || !grad_x2) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_mul_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (const float*)x1, (float*)grad_x2, size);
    } else {
        backwards_mul_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)x1, (double*)grad_x2, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

int backwards_sub_x1_device(const void* grad_c, void* grad_x1, size_t size, DType dtype) {
    if (!grad_c || !grad_x1) return -1;

    size_t element_size = (dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);
    size_t nbytes = size * element_size;

    cudaError_t err = cudaMemcpy(grad_x1, grad_c, nbytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        return -1;
    }

    return 0;
}

template<typename T>
__global__ void backwards_sub_x2_kernel(const T* grad_c, T* grad_x2, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_x2[idx] = -grad_c[idx];
    }
}

int backwards_sub_x2_device(const void* grad_c, void* grad_x2, size_t size, DType dtype) {
    if (!grad_c || !grad_x2) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_sub_x2_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (float*)grad_x2, size);
    } else {
        backwards_sub_x2_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (double*)grad_x2, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

template<typename T>
__global__ void backwards_div_x1_kernel(const T* grad_c, const T* x2, T* grad_x1, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_x1[idx] = grad_c[idx] / x2[idx];
    }
}

int backwards_div_x1_device(const void* grad_c, const void* x2, void* grad_x1, size_t size, DType dtype) {
    if (!grad_c || !x2 || !grad_x1) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_div_x1_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (const float*)x2, (float*)grad_x1, size);
    } else {
        backwards_div_x1_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)x2, (double*)grad_x1, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

template<typename T>
__global__ void backwards_div_x2_kernel(const T* grad_c, const T* x1, const T* x2, T* grad_x2, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_x2[idx] = -grad_c[idx] * x1[idx] / (x2[idx] * x2[idx]);
    }
}

int backwards_div_x2_device(const void* grad_c, const void* x1, const void* x2, void* grad_x2, size_t size, DType dtype) {
    if (!grad_c || !x1 || !x2 || !grad_x2) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_div_x2_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (const float*)x1, (const float*)x2, (float*)grad_x2, size);
    } else {
        backwards_div_x2_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)x1, (const double*)x2, (double*)grad_x2, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}
