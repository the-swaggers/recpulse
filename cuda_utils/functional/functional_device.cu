#include "functional.h"
#include "../core/cuda_helpers.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <type_traits>

__device__ __half rp_pow(__half a, __half b) { return __float2half(powf(__half2float(a), __half2float(b))); }
__device__ __half rp_log(__half a) { return hlog(a); }
__device__ __half rp_exp(__half a) { return hexp(a); }
__device__ __half rp_sqrt(__half a) { return hsqrt(a); }
__device__ __half rp_rsqrt(__half a) { return hrsqrt(a); }
__device__ __half rp_sin(__half a) { return hsin(a); }
__device__ __half rp_cos(__half a) { return hcos(a); }
__device__ __half rp_tan(__half a) { return __float2half(tanf(__half2float(a))); }
__device__ __half rp_asin(__half a) { return __float2half(asinf(__half2float(a))); }
__device__ __half rp_acos(__half a) { return __float2half(acosf(__half2float(a))); }
__device__ __half rp_atan(__half a) { return __float2half(atanf(__half2float(a))); }
__device__ __half rp_sinh(__half a) { return __float2half(sinhf(__half2float(a))); }
__device__ __half rp_cosh(__half a) { return __float2half(coshf(__half2float(a))); }
__device__ __half rp_tanh(__half a) { return __float2half(tanhf(__half2float(a))); }
__device__ __half rp_abs(__half a) { return __habs(a); }
__device__ __half rp_ceil(__half a) { return hceil(a); }
__device__ __half rp_floor(__half a) { return hfloor(a); }
__device__ __half rp_rint(__half a) { return hrint(a); }
__device__ __half rp_trunc(__half a) { return htrunc(a); }
__device__ __half rp_fabs(__half a) { return __habs(a); }

__device__ __nv_bfloat16 rp_pow(__nv_bfloat16 a, __nv_bfloat16 b) { return __float2bfloat16(powf(__bfloat162float(a), __bfloat162float(b))); }
__device__ __nv_bfloat16 rp_log(__nv_bfloat16 a) { return __float2bfloat16(logf(__bfloat162float(a))); }
__device__ __nv_bfloat16 rp_exp(__nv_bfloat16 a) { return __float2bfloat16(expf(__bfloat162float(a))); }
__device__ __nv_bfloat16 rp_sqrt(__nv_bfloat16 a) { return __float2bfloat16(sqrtf(__bfloat162float(a))); }
__device__ __nv_bfloat16 rp_rsqrt(__nv_bfloat16 a) { return __float2bfloat16(rsqrtf(__bfloat162float(a))); }
__device__ __nv_bfloat16 rp_sin(__nv_bfloat16 a) { return __float2bfloat16(sinf(__bfloat162float(a))); }
__device__ __nv_bfloat16 rp_cos(__nv_bfloat16 a) { return __float2bfloat16(cosf(__bfloat162float(a))); }
__device__ __nv_bfloat16 rp_tan(__nv_bfloat16 a) { return __float2bfloat16(tanf(__bfloat162float(a))); }
__device__ __nv_bfloat16 rp_asin(__nv_bfloat16 a) { return __float2bfloat16(asinf(__bfloat162float(a))); }
__device__ __nv_bfloat16 rp_acos(__nv_bfloat16 a) { return __float2bfloat16(acosf(__bfloat162float(a))); }
__device__ __nv_bfloat16 rp_atan(__nv_bfloat16 a) { return __float2bfloat16(atanf(__bfloat162float(a))); }
__device__ __nv_bfloat16 rp_sinh(__nv_bfloat16 a) { return __float2bfloat16(sinhf(__bfloat162float(a))); }
__device__ __nv_bfloat16 rp_cosh(__nv_bfloat16 a) { return __float2bfloat16(coshf(__bfloat162float(a))); }
__device__ __nv_bfloat16 rp_tanh(__nv_bfloat16 a) { return __float2bfloat16(tanhf(__bfloat162float(a))); }
__device__ __nv_bfloat16 rp_abs(__nv_bfloat16 a) { return __float2bfloat16(fabsf(__bfloat162float(a))); }
__device__ __nv_bfloat16 rp_ceil(__nv_bfloat16 a) { return __float2bfloat16(ceilf(__bfloat162float(a))); }
__device__ __nv_bfloat16 rp_floor(__nv_bfloat16 a) { return __float2bfloat16(floorf(__bfloat162float(a))); }
__device__ __nv_bfloat16 rp_rint(__nv_bfloat16 a) { return __float2bfloat16(rintf(__bfloat162float(a))); }
__device__ __nv_bfloat16 rp_trunc(__nv_bfloat16 a) { return __float2bfloat16(truncf(__bfloat162float(a))); }
__device__ __nv_bfloat16 rp_fabs(__nv_bfloat16 a) { return __float2bfloat16(fabsf(__bfloat162float(a))); }

__device__ float rp_pow(float a, float b) { return powf(a, b); }
__device__ float rp_log(float a) { return logf(a); }
__device__ float rp_exp(float a) { return expf(a); }
__device__ float rp_sqrt(float a) { return sqrtf(a); }
__device__ float rp_rsqrt(float a) { return rsqrtf(a); }
__device__ float rp_sin(float a) { return sinf(a); }
__device__ float rp_cos(float a) { return cosf(a); }
__device__ float rp_tan(float a) { return tanf(a); }
__device__ float rp_asin(float a) { return asinf(a); }
__device__ float rp_acos(float a) { return acosf(a); }
__device__ float rp_atan(float a) { return atanf(a); }
__device__ float rp_sinh(float a) { return sinhf(a); }
__device__ float rp_cosh(float a) { return coshf(a); }
__device__ float rp_tanh(float a) { return tanhf(a); }
__device__ float rp_abs(float a) { return fabsf(a); }
__device__ float rp_ceil(float a) { return ceilf(a); }
__device__ float rp_floor(float a) { return floorf(a); }
__device__ float rp_rint(float a) { return rintf(a); }
__device__ float rp_trunc(float a) { return truncf(a); }
__device__ float rp_fabs(float a) { return fabsf(a); }

__device__ double rp_pow(double a, double b) { return pow(a, b); }
__device__ double rp_log(double a) { return log(a); }
__device__ double rp_exp(double a) { return exp(a); }
__device__ double rp_sqrt(double a) { return sqrt(a); }
__device__ double rp_rsqrt(double a) { return rsqrt(a); }
__device__ double rp_sin(double a) { return sin(a); }
__device__ double rp_cos(double a) { return cos(a); }
__device__ double rp_tan(double a) { return tan(a); }
__device__ double rp_asin(double a) { return asin(a); }
__device__ double rp_acos(double a) { return acos(a); }
__device__ double rp_atan(double a) { return atan(a); }
__device__ double rp_sinh(double a) { return sinh(a); }
__device__ double rp_cosh(double a) { return cosh(a); }
__device__ double rp_tanh(double a) { return tanh(a); }
__device__ double rp_abs(double a) { return fabs(a); }
__device__ double rp_ceil(double a) { return ceil(a); }
__device__ double rp_floor(double a) { return floor(a); }
__device__ double rp_rint(double a) { return rint(a); }
__device__ double rp_trunc(double a) { return trunc(a); }
__device__ double rp_fabs(double a) { return fabs(a); }

template<typename T>
__global__ void add_kernel(T* out, const T* a, const T* b, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

template<typename T>
__global__ void sub_kernel(T* out, const T* a, const T* b, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] - b[idx];
    }
}

template<typename T>
__global__ void mul_kernel(T* out, const T* a, const T* b, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * b[idx];
    }
}

template<typename T>
__global__ void div_kernel(T* out, const T* a, const T* b, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] / b[idx];
    }
}

template<typename T>
__global__ void pow_kernel(T* out, const T* a, const T* b, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = rp_pow(a[idx], b[idx]);
    }
}

template<typename T>
__global__ void logb_kernel(T* out, const T* a, const T* b, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = rp_log(a[idx]) / rp_log(b[idx]);
    }
}

template<typename T>
__global__ void add_scalar_kernel(T* out, const T* a, T scalar, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + scalar;
    }
}

template<typename T>
__global__ void sub_scalar_kernel(T* out, const T* a, T scalar, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] - scalar;
    }
}

template<typename T>
__global__ void mul_scalar_kernel(T* out, const T* a, T scalar, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * scalar;
    }
}

template<typename T>
__global__ void div_scalar_kernel(T* out, const T* a, T scalar, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] / scalar;
    }
}

template<typename T>
__global__ void pow_scalar_kernel(T* out, const T* a, T scalar, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = rp_pow(a[idx], scalar);
    }
}

template<typename T>
__global__ void rsub_scalar_kernel(T* out, T scalar, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = scalar - a[idx];
    }
}

template<typename T>
__global__ void rdiv_scalar_kernel(T* out, T scalar, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = scalar / a[idx];
    }
}

template<typename T>
__global__ void rpow_scalar_kernel(T* out, T scalar, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = rp_pow(scalar, a[idx]);
    }
}

template<typename T>
__global__ void logb_scalar_kernel(T* out, const T* a, T scalar, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = rp_log(a[idx]) / rp_log(scalar);
    }
}

template<typename T>
__global__ void rlogb_scalar_kernel(T* out, T scalar, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = rp_log(scalar) / rp_log(a[idx]);
    }
}

#define BINARY_DEVICE_WRAPPER(name) \
int name##_kernel_device(void* out, const void* x1, const void* x2, size_t size, DType dtype) { \
    if (!out || !x1 || !x2 || size == 0) return -1; \
    size_t threads = 256; \
    size_t blocks = (size + threads - 1) / threads; \
    if (dtype == DTYPE_FLOAT32) { \
        name##_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x1, (const float*)x2, size); \
    } else if (dtype == DTYPE_FLOAT64) { \
        name##_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x1, (const double*)x2, size); \
    } else if (dtype == DTYPE_FLOAT16) { \
        name##_kernel<__half><<<blocks, threads>>>((__half*)out, (const __half*)x1, (const __half*)x2, size); \
    } else if (dtype == DTYPE_BFLOAT16) { \
        name##_kernel<__nv_bfloat16><<<blocks, threads>>>((__nv_bfloat16*)out, (const __nv_bfloat16*)x1, (const __nv_bfloat16*)x2, size); \
    } else { \
        return -1; \
    } \
    return check_cuda_kernel() ? 0 : -1; \
}

BINARY_DEVICE_WRAPPER(add)
BINARY_DEVICE_WRAPPER(sub)
BINARY_DEVICE_WRAPPER(mul)
BINARY_DEVICE_WRAPPER(div)
BINARY_DEVICE_WRAPPER(pow)
BINARY_DEVICE_WRAPPER(logb)

template<typename T>
__device__ void compute_strided_indices(size_t flat_idx, int ndim, const int* out_shape,
                                         const int* x1_strides, const int* x2_strides,
                                         size_t* idx1, size_t* idx2) {
    size_t rem = flat_idx;
    *idx1 = 0;
    *idx2 = 0;
    for (int d = ndim - 1; d >= 0; d--) {
        int coord = rem % out_shape[d];
        rem /= out_shape[d];
        *idx1 += coord * x1_strides[d];
        *idx2 += coord * x2_strides[d];
    }
}

#define STRIDED_BINARY_CUDA_KERNEL(name, op_expr) \
template<typename T> \
__global__ void name##_strided_kernel(T* out, const T* a, const T* b, \
                                       int ndim, const int* out_shape, \
                                       const int* a_strides, const int* b_strides, \
                                       size_t out_size) { \
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx < out_size) { \
        size_t idx1, idx2; \
        compute_strided_indices<T>(idx, ndim, out_shape, a_strides, b_strides, &idx1, &idx2); \
        T va = a[idx1]; \
        T vb = b[idx2]; \
        out[idx] = op_expr; \
    } \
}

STRIDED_BINARY_CUDA_KERNEL(add, va + vb)
STRIDED_BINARY_CUDA_KERNEL(sub, va - vb)
STRIDED_BINARY_CUDA_KERNEL(mul, va * vb)
STRIDED_BINARY_CUDA_KERNEL(div, va / vb)
STRIDED_BINARY_CUDA_KERNEL(pow, rp_pow(va, vb))
STRIDED_BINARY_CUDA_KERNEL(logb, rp_log(va) / rp_log(vb))

#define STRIDED_BINARY_DEVICE_WRAPPER(name, kernel_name) \
int name##_strided_kernel_device(void* out, const void* x1, const void* x2, \
                                  int ndim, const int* out_shape, \
                                  const int* x1_strides, const int* x2_strides, \
                                  size_t out_size, DType dtype) { \
    if (!out || !x1 || !x2 || out_size == 0) return -1; \
    int* d_shape, *d_s1, *d_s2; \
    cudaMalloc(&d_shape, ndim * sizeof(int)); \
    cudaMalloc(&d_s1, ndim * sizeof(int)); \
    cudaMalloc(&d_s2, ndim * sizeof(int)); \
    cudaMemcpy(d_shape, out_shape, ndim * sizeof(int), cudaMemcpyHostToDevice); \
    cudaMemcpy(d_s1, x1_strides, ndim * sizeof(int), cudaMemcpyHostToDevice); \
    cudaMemcpy(d_s2, x2_strides, ndim * sizeof(int), cudaMemcpyHostToDevice); \
    size_t threads = 256; \
    size_t blocks = (out_size + threads - 1) / threads; \
    if (dtype == DTYPE_FLOAT32) { \
        kernel_name##_strided_kernel<float><<<blocks, threads>>>( \
            (float*)out, (const float*)x1, (const float*)x2, \
            ndim, d_shape, d_s1, d_s2, out_size); \
    } else if (dtype == DTYPE_FLOAT64) { \
        kernel_name##_strided_kernel<double><<<blocks, threads>>>( \
            (double*)out, (const double*)x1, (const double*)x2, \
            ndim, d_shape, d_s1, d_s2, out_size); \
    } else if (dtype == DTYPE_FLOAT16) { \
        kernel_name##_strided_kernel<__half><<<blocks, threads>>>( \
            (__half*)out, (const __half*)x1, (const __half*)x2, \
            ndim, d_shape, d_s1, d_s2, out_size); \
    } else if (dtype == DTYPE_BFLOAT16) { \
        kernel_name##_strided_kernel<__nv_bfloat16><<<blocks, threads>>>( \
            (__nv_bfloat16*)out, (const __nv_bfloat16*)x1, (const __nv_bfloat16*)x2, \
            ndim, d_shape, d_s1, d_s2, out_size); \
    } else { \
        cudaFree(d_shape); cudaFree(d_s1); cudaFree(d_s2); \
        return -1; \
    } \
    bool ok = check_cuda_kernel(); \
    cudaFree(d_shape); cudaFree(d_s1); cudaFree(d_s2); \
    return ok ? 0 : -1; \
}

STRIDED_BINARY_DEVICE_WRAPPER(add, add)
STRIDED_BINARY_DEVICE_WRAPPER(sub, sub)
STRIDED_BINARY_DEVICE_WRAPPER(mul, mul)
STRIDED_BINARY_DEVICE_WRAPPER(div, div)
STRIDED_BINARY_DEVICE_WRAPPER(pow, pow)
STRIDED_BINARY_DEVICE_WRAPPER(logb, logb)

#define SCALAR_DEVICE_WRAPPER(name) \
int name##_scalar_kernel_device(void* out, const void* x, const void* scalar, size_t size, DType dtype) { \
    if (!out || !x || !scalar || size == 0) return -1; \
    size_t threads = 256; \
    size_t blocks = (size + threads - 1) / threads; \
    if (dtype == DTYPE_FLOAT32) { \
        float scalar_val = *(const float*)scalar; \
        name##_scalar_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x, scalar_val, size); \
    } else if (dtype == DTYPE_FLOAT64) { \
        double scalar_val = *(const double*)scalar; \
        name##_scalar_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x, scalar_val, size); \
    } else if (dtype == DTYPE_FLOAT16) { \
        __half scalar_val = __float2half(*(const float*)scalar); \
        name##_scalar_kernel<__half><<<blocks, threads>>>((__half*)out, (const __half*)x, scalar_val, size); \
    } else if (dtype == DTYPE_BFLOAT16) { \
        __nv_bfloat16 scalar_val = __float2bfloat16(*(const float*)scalar); \
        name##_scalar_kernel<__nv_bfloat16><<<blocks, threads>>>((__nv_bfloat16*)out, (const __nv_bfloat16*)x, scalar_val, size); \
    } else { \
        return -1; \
    } \
    return check_cuda_kernel() ? 0 : -1; \
}

SCALAR_DEVICE_WRAPPER(add)
SCALAR_DEVICE_WRAPPER(sub)
SCALAR_DEVICE_WRAPPER(mul)
SCALAR_DEVICE_WRAPPER(div)
SCALAR_DEVICE_WRAPPER(pow)
SCALAR_DEVICE_WRAPPER(logb)

#define RSCALAR_DEVICE_WRAPPER(name) \
int name##_scalar_kernel_device(void* out, const void* scalar, const void* x, size_t size, DType dtype) { \
    if (!out || !x || !scalar || size == 0) return -1; \
    size_t threads = 256; \
    size_t blocks = (size + threads - 1) / threads; \
    if (dtype == DTYPE_FLOAT32) { \
        float scalar_val = *(const float*)scalar; \
        name##_scalar_kernel<float><<<blocks, threads>>>((float*)out, scalar_val, (const float*)x, size); \
    } else if (dtype == DTYPE_FLOAT64) { \
        double scalar_val = *(const double*)scalar; \
        name##_scalar_kernel<double><<<blocks, threads>>>((double*)out, scalar_val, (const double*)x, size); \
    } else if (dtype == DTYPE_FLOAT16) { \
        __half scalar_val = __float2half(*(const float*)scalar); \
        name##_scalar_kernel<__half><<<blocks, threads>>>((__half*)out, scalar_val, (const __half*)x, size); \
    } else if (dtype == DTYPE_BFLOAT16) { \
        __nv_bfloat16 scalar_val = __float2bfloat16(*(const float*)scalar); \
        name##_scalar_kernel<__nv_bfloat16><<<blocks, threads>>>((__nv_bfloat16*)out, scalar_val, (const __nv_bfloat16*)x, size); \
    } else { \
        return -1; \
    } \
    return check_cuda_kernel() ? 0 : -1; \
}

RSCALAR_DEVICE_WRAPPER(rsub)
RSCALAR_DEVICE_WRAPPER(rdiv)
RSCALAR_DEVICE_WRAPPER(rpow)
RSCALAR_DEVICE_WRAPPER(rlogb)

template<typename T>
__global__ void exp_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = rp_exp(a[idx]);
    }
}

template<typename T>
__global__ void log_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = rp_log(a[idx]);
    }
}

template<typename T>
__global__ void sqrt_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = rp_sqrt(a[idx]);
    }
}

template<typename T>
__global__ void abs_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = rp_fabs(a[idx]);
    }
}

template<typename T>
__global__ void sin_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = rp_sin(a[idx]);
    }
}

template<typename T>
__global__ void cos_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = rp_cos(a[idx]);
    }
}

template<typename T>
__global__ void tan_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = rp_tan(a[idx]);
    }
}

template<typename T>
__global__ void asin_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = rp_asin(a[idx]);
    }
}

template<typename T>
__global__ void acos_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = rp_acos(a[idx]);
    }
}

template<typename T>
__global__ void atan_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = rp_atan(a[idx]);
    }
}

template<typename T>
__global__ void sinh_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = rp_sinh(a[idx]);
    }
}

template<typename T>
__global__ void cosh_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = rp_cosh(a[idx]);
    }
}

template<typename T>
__global__ void tanh_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = rp_tanh(a[idx]);
    }
}

template<typename T>
__global__ void relu_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = (a[idx] > T(0)) ? a[idx] : T(0);
    }
}

template<>
__global__ void relu_kernel<__half>(__half* out, const __half* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        __half zero = __float2half(0.0f);
        out[idx] = __hgt(a[idx], zero) ? a[idx] : zero;
    }
}

template<>
__global__ void relu_kernel<__nv_bfloat16>(__nv_bfloat16* out, const __nv_bfloat16* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        __nv_bfloat16 zero = __float2bfloat16(0.0f);
#if __CUDA_ARCH__ >= 800
        out[idx] = __hgt(a[idx], zero) ? a[idx] : zero;
#else
        out[idx] = (__bfloat162float(a[idx]) > 0.0f) ? a[idx] : zero;
#endif
    }
}

template<typename T>
__global__ void sigmoid_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = T(1) / (T(1) + rp_exp(-a[idx]));
    }
}

template<>
__global__ void sigmoid_kernel<__half>(__half* out, const __half* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = __half2float(a[idx]);
        out[idx] = __float2half(1.0f / (1.0f + expf(-val)));
    }
}

template<>
__global__ void sigmoid_kernel<__nv_bfloat16>(__nv_bfloat16* out, const __nv_bfloat16* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = __bfloat162float(a[idx]);
        out[idx] = __float2bfloat16(1.0f / (1.0f + expf(-val)));
    }
}

template<typename T>
__global__ void leaky_relu_kernel(T* out, const T* a, T alpha, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = (a[idx] > T(0)) ? a[idx] : alpha * a[idx];
    }
}

template<>
__global__ void leaky_relu_kernel<__half>(__half* out, const __half* a, __half alpha, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        __half zero = __float2half(0.0f);
        out[idx] = __hgt(a[idx], zero) ? a[idx] : __hmul(alpha, a[idx]);
    }
}

template<>
__global__ void leaky_relu_kernel<__nv_bfloat16>(__nv_bfloat16* out, const __nv_bfloat16* a, __nv_bfloat16 alpha, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = __bfloat162float(a[idx]);
        float alpha_f = __bfloat162float(alpha);
        out[idx] = __float2bfloat16((val > 0.0f) ? val : alpha_f * val);
    }
}

template<typename T>
__global__ void square_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * a[idx];
    }
}

template<typename T>
__global__ void ceil_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = rp_ceil(a[idx]);
    }
}

template<typename T>
__global__ void floor_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = rp_floor(a[idx]);
    }
}

template<typename T>
__global__ void round_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = rp_rint(a[idx]);
    }
}

template<typename T>
__global__ void trunc_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = rp_trunc(a[idx]);
    }
}

template<typename T>
__global__ void rsqrt_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = rp_rsqrt(a[idx]);
    }
}

template<typename T>
__global__ void gelu_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T x_val = a[idx];
        T sqrt_2_over_pi = T(0.7978845608);
        T coeff = T(0.044715);
        T x_cubed = x_val * x_val * x_val;
        T inner = sqrt_2_over_pi * (x_val + coeff * x_cubed);
        out[idx] = T(0.5) * x_val * (T(1.0) + rp_tanh(inner));
    }
}

template<>
__global__ void gelu_kernel<__half>(__half* out, const __half* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x_val = __half2float(a[idx]);
        float x_cubed = x_val * x_val * x_val;
        float inner = 0.7978845608f * (x_val + 0.044715f * x_cubed);
        out[idx] = __float2half(0.5f * x_val * (1.0f + tanhf(inner)));
    }
}

template<>
__global__ void gelu_kernel<__nv_bfloat16>(__nv_bfloat16* out, const __nv_bfloat16* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x_val = __bfloat162float(a[idx]);
        float x_cubed = x_val * x_val * x_val;
        float inner = 0.7978845608f * (x_val + 0.044715f * x_cubed);
        out[idx] = __float2bfloat16(0.5f * x_val * (1.0f + tanhf(inner)));
    }
}

template<typename T>
__global__ void silu_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T x_val = a[idx];
        out[idx] = x_val / (T(1.0) + rp_exp(-x_val));
    }
}

template<>
__global__ void silu_kernel<__half>(__half* out, const __half* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x_val = __half2float(a[idx]);
        out[idx] = __float2half(x_val / (1.0f + expf(-x_val)));
    }
}

template<>
__global__ void silu_kernel<__nv_bfloat16>(__nv_bfloat16* out, const __nv_bfloat16* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x_val = __bfloat162float(a[idx]);
        out[idx] = __float2bfloat16(x_val / (1.0f + expf(-x_val)));
    }
}

#define UNARY_DEVICE_WRAPPER(name) \
int name##_kernel_device(void* out, const void* x, size_t size, DType dtype) { \
    if (!out || !x || size == 0) return -1; \
    size_t threads = 256; \
    size_t blocks = (size + threads - 1) / threads; \
    if (dtype == DTYPE_FLOAT32) { \
        name##_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x, size); \
    } else if (dtype == DTYPE_FLOAT64) { \
        name##_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x, size); \
    } else if (dtype == DTYPE_FLOAT16) { \
        name##_kernel<__half><<<blocks, threads>>>((__half*)out, (const __half*)x, size); \
    } else if (dtype == DTYPE_BFLOAT16) { \
        name##_kernel<__nv_bfloat16><<<blocks, threads>>>((__nv_bfloat16*)out, (const __nv_bfloat16*)x, size); \
    } else { \
        return -1; \
    } \
    return check_cuda_kernel() ? 0 : -1; \
}

UNARY_DEVICE_WRAPPER(exp)
UNARY_DEVICE_WRAPPER(log)
UNARY_DEVICE_WRAPPER(sqrt)
UNARY_DEVICE_WRAPPER(abs)
UNARY_DEVICE_WRAPPER(sin)
UNARY_DEVICE_WRAPPER(cos)
UNARY_DEVICE_WRAPPER(tan)
UNARY_DEVICE_WRAPPER(asin)
UNARY_DEVICE_WRAPPER(acos)
UNARY_DEVICE_WRAPPER(atan)
UNARY_DEVICE_WRAPPER(sinh)
UNARY_DEVICE_WRAPPER(cosh)
UNARY_DEVICE_WRAPPER(tanh)
UNARY_DEVICE_WRAPPER(relu)
UNARY_DEVICE_WRAPPER(sigmoid)
UNARY_DEVICE_WRAPPER(square)
UNARY_DEVICE_WRAPPER(ceil)
UNARY_DEVICE_WRAPPER(floor)
UNARY_DEVICE_WRAPPER(round)
UNARY_DEVICE_WRAPPER(trunc)
UNARY_DEVICE_WRAPPER(rsqrt)
UNARY_DEVICE_WRAPPER(gelu)
UNARY_DEVICE_WRAPPER(silu)

int leaky_relu_kernel_device(void* out, const void* x, const void* alpha, size_t size, DType dtype) {
    if (!out || !x || !alpha || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        float alpha_val = *(const float*)alpha;
        leaky_relu_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x, alpha_val, size);
    } else if (dtype == DTYPE_FLOAT64) {
        double alpha_val = *(const double*)alpha;
        leaky_relu_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x, alpha_val, size);
    } else if (dtype == DTYPE_FLOAT16) {
        __half alpha_val = __float2half(*(const float*)alpha);
        leaky_relu_kernel<__half><<<blocks, threads>>>((__half*)out, (const __half*)x, alpha_val, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        __nv_bfloat16 alpha_val = __float2bfloat16(*(const float*)alpha);
        leaky_relu_kernel<__nv_bfloat16><<<blocks, threads>>>((__nv_bfloat16*)out, (const __nv_bfloat16*)x, alpha_val, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

template<typename T>
__global__ void sum_reduction_kernel(const T* input, T* output, size_t size) {
    extern __shared__ char shared_mem[];
    T* sdata = (T*)shared_mem;

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        sdata[tid] = input[idx];
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1){
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

template<typename HalfT>
__global__ void half_to_float_kernel(float* out, const HalfT* in, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if constexpr (std::is_same_v<HalfT, __half>)
            out[idx] = __half2float(in[idx]);
        else
            out[idx] = __bfloat162float(in[idx]);
    }
}

template<typename HalfT>
__global__ void float_to_half_kernel(HalfT* out, const float* in, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if constexpr (std::is_same_v<HalfT, __half>)
            out[idx] = __float2half(in[idx]);
        else
            out[idx] = __float2bfloat16(in[idx]);
    }
}

static int sum_all_fp32_reduction(void* out, const float* d_x_f32, size_t size) {
    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;
    size_t shared_mem_size = threads * sizeof(float);

    float* d_partial_sums;
    cudaMalloc(&d_partial_sums, blocks * sizeof(float));

    sum_reduction_kernel<float><<<blocks, threads, shared_mem_size>>>(d_x_f32, d_partial_sums, size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_partial_sums);
        return -1;
    }

    if (blocks > 1) {
        size_t threads2 = (blocks < threads) ? blocks : threads;
        size_t shared_mem_size2 = threads2 * sizeof(float);
        sum_reduction_kernel<float><<<1, threads2, shared_mem_size2>>>(d_partial_sums, (float*)out, blocks);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(d_partial_sums);
            return -1;
        }
    } else {
        cudaMemcpy(out, d_partial_sums, sizeof(float), cudaMemcpyDeviceToDevice);
    }

    cudaFree(d_partial_sums);
    return 0;
}

int sum_all_kernel_device(void* out, const void* x, size_t size, DType dtype) {
    if (!out || !x || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;
    size_t shared_mem_size = threads * (dtype == DTYPE_FLOAT32 ? sizeof(float) : sizeof(double));

    if (dtype == DTYPE_FLOAT32) {
        float* d_partial_sums;
        cudaMalloc(&d_partial_sums, blocks * sizeof(float));

        sum_reduction_kernel<float><<<blocks, threads, shared_mem_size>>>(
            (const float*)x, d_partial_sums, size
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(d_partial_sums);
            return -1;
        }

        if (blocks > 1) {
            size_t threads2 = (blocks < threads) ? blocks : threads;
            size_t shared_mem_size2 = threads2 * sizeof(float);
            sum_reduction_kernel<float><<<1, threads2, shared_mem_size2>>>(
                d_partial_sums, (float*)out, blocks
            );
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                cudaFree(d_partial_sums);
                return -1;
            }
        } else {
            cudaMemcpy(out, d_partial_sums, sizeof(float), cudaMemcpyDeviceToDevice);
        }

        cudaFree(d_partial_sums);
    } else if (dtype == DTYPE_FLOAT64) {
        double* d_partial_sums;
        cudaMalloc(&d_partial_sums, blocks * sizeof(double));

        sum_reduction_kernel<double><<<blocks, threads, shared_mem_size>>>(
            (const double*)x, d_partial_sums, size
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(d_partial_sums);
            return -1;
        }

        if (blocks > 1) {
            size_t threads2 = (blocks < threads) ? blocks : threads;
            size_t shared_mem_size2 = threads2 * sizeof(double);
            sum_reduction_kernel<double><<<1, threads2, shared_mem_size2>>>(
                d_partial_sums, (double*)out, blocks
            );
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                cudaFree(d_partial_sums);
                return -1;
            }
        } else {
            cudaMemcpy(out, d_partial_sums, sizeof(double), cudaMemcpyDeviceToDevice);
        }

        cudaFree(d_partial_sums);
    } else if (dtype == DTYPE_FLOAT16 || dtype == DTYPE_BFLOAT16) {
        float* d_x_f32;
        cudaError_t err = cudaMalloc(&d_x_f32, size * sizeof(float));
        if (err != cudaSuccess) return -1;

        size_t conv_threads = 256;
        size_t conv_blocks = (size + conv_threads - 1) / conv_threads;

        if (dtype == DTYPE_FLOAT16) {
            half_to_float_kernel<__half><<<conv_blocks, conv_threads>>>(d_x_f32, (const __half*)x, size);
        } else {
            half_to_float_kernel<__nv_bfloat16><<<conv_blocks, conv_threads>>>(d_x_f32, (const __nv_bfloat16*)x, size);
        }

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(d_x_f32);
            return -1;
        }

        float* d_result_f32;
        err = cudaMalloc(&d_result_f32, sizeof(float));
        if (err != cudaSuccess) {
            cudaFree(d_x_f32);
            return -1;
        }

        int rc = sum_all_fp32_reduction(d_result_f32, d_x_f32, size);
        cudaFree(d_x_f32);

        if (rc != 0) {
            cudaFree(d_result_f32);
            return -1;
        }

        float h_result;
        cudaMemcpy(&h_result, d_result_f32, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_result_f32);

        if (dtype == DTYPE_FLOAT16) {
            __half h_val = __float2half(h_result);
            cudaMemcpy(out, &h_val, sizeof(__half), cudaMemcpyHostToDevice);
        } else {
            __nv_bfloat16 h_val = __float2bfloat16(h_result);
            cudaMemcpy(out, &h_val, sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        }
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int mean_all_kernel_device(void* out, const void* x, size_t size, DType dtype) {
    if (!out || !x || size == 0) return -1;

    if (sum_all_kernel_device(out, x, size, dtype) != 0) {
        return -1;
    }

    if (dtype == DTYPE_FLOAT32) {
        float result;
        cudaMemcpy(&result, out, sizeof(float), cudaMemcpyDeviceToHost);
        result /= (float)size;
        cudaMemcpy(out, &result, sizeof(float), cudaMemcpyHostToDevice);
    } else if (dtype == DTYPE_FLOAT64) {
        double result;
        cudaMemcpy(&result, out, sizeof(double), cudaMemcpyDeviceToHost);
        result /= (double)size;
        cudaMemcpy(out, &result, sizeof(double), cudaMemcpyHostToDevice);
    } else if (dtype == DTYPE_FLOAT16) {
        __half h_result;
        cudaMemcpy(&h_result, out, sizeof(__half), cudaMemcpyDeviceToHost);
        float result = __half2float(h_result) / (float)size;
        h_result = __float2half(result);
        cudaMemcpy(out, &h_result, sizeof(__half), cudaMemcpyHostToDevice);
    } else if (dtype == DTYPE_BFLOAT16) {
        __nv_bfloat16 h_result;
        cudaMemcpy(&h_result, out, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
        float result = __bfloat162float(h_result) / (float)size;
        h_result = __float2bfloat16(result);
        cudaMemcpy(out, &h_result, sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int matmul_kernel_device(void* C, const void* A, const void* B, int m, int k, int n, DType dtype) {
    if (!C || !A || !B || m <= 0 || k <= 0 || n <= 0) return -1;

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS initialization failed\n");
        return -1;
    }

    if (dtype == DTYPE_FLOAT32) {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        status = cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, k,
            &alpha,
            (const float*)B, n,
            (const float*)A, k,
            &beta,
            (float*)C, n
        );
    } else if (dtype == DTYPE_FLOAT64) {
        const double alpha = 1.0;
        const double beta = 0.0;

        status = cublasDgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, k,
            &alpha,
            (const double*)B, n,
            (const double*)A, k,
            &beta,
            (double*)C, n
        );
    } else if (dtype == DTYPE_FLOAT16) {
        const __half alpha = __float2half(1.0f);
        const __half beta = __float2half(0.0f);

        status = cublasHgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, k,
            &alpha,
            (const __half*)B, n,
            (const __half*)A, k,
            &beta,
            (__half*)C, n
        );
    } else if (dtype == DTYPE_BFLOAT16) {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        status = cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, k,
            &alpha,
            B, CUDA_R_16BF, n,
            A, CUDA_R_16BF, k,
            &beta,
            C, CUDA_R_16BF, n,
            CUDA_R_32F, CUBLAS_GEMM_DEFAULT
        );
    } else {
        cublasDestroy(handle);
        return -1;
    }

    cublasDestroy(handle);

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS matmul failed with error code: %d\n", status);
        return -1;
    }

    return 0;
}

template<typename T>
__global__ void cat_copy_kernel(T* out, const T* src, int* out_shape, int* src_shape,
                                 int ndim, int cat_dim, int offset_in_cat_dim, size_t src_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= src_size) return;

    int indices[8];
    size_t temp = idx;
    for (int d = ndim - 1; d >= 0; d--) {
        indices[d] = temp % src_shape[d];
        temp /= src_shape[d];
    }

    indices[cat_dim] += offset_in_cat_dim;

    size_t out_idx = 0;
    for (int d = 0; d < ndim; d++) {
        out_idx = out_idx * out_shape[d] + indices[d];
    }

    out[out_idx] = src[idx];
}

static size_t get_dtype_size(DType dtype) {
    switch (dtype) {
        case DTYPE_FLOAT32: return sizeof(float);
        case DTYPE_FLOAT64: return sizeof(double);
        case DTYPE_FLOAT16: return sizeof(__half);
        case DTYPE_BFLOAT16: return sizeof(__nv_bfloat16);
        default: return 0;
    }
}

Tensor* cat_kernel_device(Tensor** tensors, int num_tensors, int dim) {
    if (!tensors || num_tensors <= 0 || !tensors[0]) return NULL;

    Tensor* first = tensors[0];
    int ndim = first->ndim;
    DType dtype = first->dtype;
    int device_id = first->device_id;

    int* out_shape = (int*)malloc(ndim * sizeof(int));
    if (!out_shape) return NULL;

    for (int d = 0; d < ndim; d++) {
        out_shape[d] = first->shape[d];
    }

    for (int i = 1; i < num_tensors; i++) {
        out_shape[dim] += tensors[i]->shape[dim];
    }

    Tensor* out = zeros_tensor(dtype, device_id, ndim, out_shape, NULL);
    if (!out) {
        free(out_shape);
        return NULL;
    }

    int offset_in_cat_dim = 0;
    size_t elem_size = get_dtype_size(dtype);

    for (int t = 0; t < num_tensors; t++) {
        Tensor* src = tensors[t];

        if (ndim == 1) {
            void* dst_ptr = (char*)out->data + offset_in_cat_dim * elem_size;
            cudaError_t err = cudaMemcpy(dst_ptr, src->data, src->size * elem_size, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                free_tensor(out);
                free(out_shape);
                return NULL;
            }
        } else {
            int* d_out_shape;
            int* d_src_shape;
            cudaMalloc(&d_out_shape, ndim * sizeof(int));
            cudaMalloc(&d_src_shape, ndim * sizeof(int));
            cudaMemcpy(d_out_shape, out_shape, ndim * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_src_shape, src->shape, ndim * sizeof(int), cudaMemcpyHostToDevice);

            size_t threads = 256;
            size_t blocks = (src->size + threads - 1) / threads;

            if (dtype == DTYPE_FLOAT32) {
                cat_copy_kernel<float><<<blocks, threads>>>(
                    (float*)out->data, (const float*)src->data,
                    d_out_shape, d_src_shape, ndim, dim, offset_in_cat_dim, src->size
                );
            } else if (dtype == DTYPE_FLOAT64) {
                cat_copy_kernel<double><<<blocks, threads>>>(
                    (double*)out->data, (const double*)src->data,
                    d_out_shape, d_src_shape, ndim, dim, offset_in_cat_dim, src->size
                );
            } else if (dtype == DTYPE_FLOAT16) {
                cat_copy_kernel<__half><<<blocks, threads>>>(
                    (__half*)out->data, (const __half*)src->data,
                    d_out_shape, d_src_shape, ndim, dim, offset_in_cat_dim, src->size
                );
            } else if (dtype == DTYPE_BFLOAT16) {
                cat_copy_kernel<__nv_bfloat16><<<blocks, threads>>>(
                    (__nv_bfloat16*)out->data, (const __nv_bfloat16*)src->data,
                    d_out_shape, d_src_shape, ndim, dim, offset_in_cat_dim, src->size
                );
            }

            cudaError_t err = cudaGetLastError();
            cudaFree(d_out_shape);
            cudaFree(d_src_shape);

            if (err != cudaSuccess) {
                free_tensor(out);
                free(out_shape);
                return NULL;
            }
        }

        offset_in_cat_dim += src->shape[dim];
    }

    free(out_shape);
    cudaDeviceSynchronize();
    return out;
}

template<typename T>
__global__ void contiguous_copy_kernel(T* out, const T* in, int ndim, int* shape, int* strides, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    int temp = idx;
    size_t in_idx = 0;

    for (int d = ndim - 1; d >= 0; d--) {
        int coord = temp % shape[d];
        temp /= shape[d];
        in_idx += coord * strides[d];
    }

    out[idx] = in[in_idx];
}

int contiguous_copy_kernel_device(void* out, const void* in, int ndim, int* shape, int* strides, size_t size, DType dtype) {
    if (!out || !in || !shape || !strides || ndim <= 0 || size == 0) return -1;

    int* d_shape;
    int* d_strides;

    cudaError_t err = cudaMalloc(&d_shape, ndim * sizeof(int));
    if (err != cudaSuccess) return -1;

    err = cudaMalloc(&d_strides, ndim * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_shape);
        return -1;
    }

    err = cudaMemcpy(d_shape, shape, ndim * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_shape);
        cudaFree(d_strides);
        return -1;
    }

    err = cudaMemcpy(d_strides, strides, ndim * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_shape);
        cudaFree(d_strides);
        return -1;
    }

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        contiguous_copy_kernel<float><<<blocks, threads>>>((float*)out, (const float*)in, ndim, d_shape, d_strides, size);
    } else if (dtype == DTYPE_FLOAT64) {
        contiguous_copy_kernel<double><<<blocks, threads>>>((double*)out, (const double*)in, ndim, d_shape, d_strides, size);
    } else if (dtype == DTYPE_FLOAT16) {
        contiguous_copy_kernel<__half><<<blocks, threads>>>((__half*)out, (const __half*)in, ndim, d_shape, d_strides, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        contiguous_copy_kernel<__nv_bfloat16><<<blocks, threads>>>((__nv_bfloat16*)out, (const __nv_bfloat16*)in, ndim, d_shape, d_strides, size);
    } else {
        cudaFree(d_shape);
        cudaFree(d_strides);
        return -1;
    }

    err = cudaGetLastError();
    cudaFree(d_shape);
    cudaFree(d_strides);

    if (err != cudaSuccess) return -1;

    cudaDeviceSynchronize();
    return 0;
}

template<typename T>
__global__ void repeat_kernel(T* out, const T* in, int ndim, int* src_shape, int* repeats, size_t out_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;

    int temp_idx = idx;
    int src_idx = 0;
    int src_stride = 1;

    for (int d = ndim - 1; d >= 0; d--) {
        int out_shape_d = src_shape[d] * repeats[d];
        int out_coord = temp_idx % out_shape_d;
        int src_coord = out_coord % src_shape[d];

        src_idx += src_coord * src_stride;
        src_stride *= src_shape[d];
        temp_idx /= out_shape_d;
    }

    out[idx] = in[src_idx];
}

int repeat_kernel_device(void* out, const void* in, int ndim, int* src_shape, int* repeats, size_t out_size, DType dtype) {
    if (!out || !in || !src_shape || !repeats || ndim <= 0 || out_size == 0) return -1;

    int* d_src_shape;
    int* d_repeats;

    cudaError_t err = cudaMalloc(&d_src_shape, ndim * sizeof(int));
    if (err != cudaSuccess) return -1;

    err = cudaMalloc(&d_repeats, ndim * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_src_shape);
        return -1;
    }

    err = cudaMemcpy(d_src_shape, src_shape, ndim * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_src_shape);
        cudaFree(d_repeats);
        return -1;
    }

    err = cudaMemcpy(d_repeats, repeats, ndim * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_src_shape);
        cudaFree(d_repeats);
        return -1;
    }

    int threads = 256;
    int blocks = (out_size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        repeat_kernel<float><<<blocks, threads>>>((float*)out, (const float*)in, ndim, d_src_shape, d_repeats, out_size);
    } else if (dtype == DTYPE_FLOAT64) {
        repeat_kernel<double><<<blocks, threads>>>((double*)out, (const double*)in, ndim, d_src_shape, d_repeats, out_size);
    } else if (dtype == DTYPE_FLOAT16) {
        repeat_kernel<__half><<<blocks, threads>>>((__half*)out, (const __half*)in, ndim, d_src_shape, d_repeats, out_size);
    } else if (dtype == DTYPE_BFLOAT16) {
        repeat_kernel<__nv_bfloat16><<<blocks, threads>>>((__nv_bfloat16*)out, (const __nv_bfloat16*)in, ndim, d_src_shape, d_repeats, out_size);
    } else {
        cudaFree(d_src_shape);
        cudaFree(d_repeats);
        return -1;
    }

    err = cudaGetLastError();
    cudaFree(d_src_shape);
    cudaFree(d_repeats);

    if (err != cudaSuccess) return -1;

    cudaDeviceSynchronize();
    return 0;
}
