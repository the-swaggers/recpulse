#include "ops.h"
#include "../core/tensor.h"
#include "../core/half_precision.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

static __device__ float rp_log(float a) { return logf(a); }
static __device__ float rp_exp(float a) { return expf(a); }
static __device__ float rp_sqrt(float a) { return sqrtf(a); }
static __device__ float rp_cos(float a) { return cosf(a); }
static __device__ float rp_sin(float a) { return sinf(a); }
static __device__ float rp_cosh(float a) { return coshf(a); }
static __device__ float rp_sinh(float a) { return sinhf(a); }
static __device__ float rp_tanh(float a) { return tanhf(a); }

static __device__ double rp_log(double a) { return log(a); }
static __device__ double rp_exp(double a) { return exp(a); }
static __device__ double rp_sqrt(double a) { return sqrt(a); }
static __device__ double rp_cos(double a) { return cos(a); }
static __device__ double rp_sin(double a) { return sin(a); }
static __device__ double rp_cosh(double a) { return cosh(a); }
static __device__ double rp_sinh(double a) { return sinh(a); }
static __device__ double rp_tanh(double a) { return tanh(a); }

static __device__ __half rp_log(__half a) { return hlog(a); }
static __device__ __half rp_exp(__half a) { return hexp(a); }
static __device__ __half rp_sqrt(__half a) { return hsqrt(a); }
static __device__ __half rp_cos(__half a) { return hcos(a); }
static __device__ __half rp_sin(__half a) { return hsin(a); }
static __device__ __half rp_cosh(__half a) { return __float2half(coshf(__half2float(a))); }
static __device__ __half rp_sinh(__half a) { return __float2half(sinhf(__half2float(a))); }
static __device__ __half rp_tanh(__half a) { return __float2half(tanhf(__half2float(a))); }

static __device__ __nv_bfloat16 rp_log(__nv_bfloat16 a) { return __float2bfloat16(logf(__bfloat162float(a))); }
static __device__ __nv_bfloat16 rp_exp(__nv_bfloat16 a) { return __float2bfloat16(expf(__bfloat162float(a))); }
static __device__ __nv_bfloat16 rp_sqrt(__nv_bfloat16 a) { return __float2bfloat16(sqrtf(__bfloat162float(a))); }
static __device__ __nv_bfloat16 rp_cos(__nv_bfloat16 a) { return __float2bfloat16(cosf(__bfloat162float(a))); }
static __device__ __nv_bfloat16 rp_sin(__nv_bfloat16 a) { return __float2bfloat16(sinf(__bfloat162float(a))); }
static __device__ __nv_bfloat16 rp_cosh(__nv_bfloat16 a) { return __float2bfloat16(coshf(__bfloat162float(a))); }
static __device__ __nv_bfloat16 rp_sinh(__nv_bfloat16 a) { return __float2bfloat16(sinhf(__bfloat162float(a))); }
static __device__ __nv_bfloat16 rp_tanh(__nv_bfloat16 a) { return __float2bfloat16(tanhf(__bfloat162float(a))); }

int backwards_add_x1_device(const void* grad_c, void* grad_x1, size_t size, DType dtype) {
    if (!grad_c || !grad_x1) return -1;

    size_t element_size = dtype_size(dtype);
    size_t nbytes = size * element_size;

    cudaError_t err = cudaMemcpy(grad_x1, grad_c, nbytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        return -1;
    }

    return 0;
}

int backwards_add_x2_device(const void* grad_c, void* grad_x2, size_t size, DType dtype) {
    if (!grad_c || !grad_x2) return -1;

    size_t element_size = dtype_size(dtype);
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
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_mul_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)x2, (double*)grad_x1, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_mul_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (const __half*)x2, (__half*)grad_x1, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_mul_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (const __nv_bfloat16*)x2, (__nv_bfloat16*)grad_x1, size);
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
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_mul_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)x1, (double*)grad_x2, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_mul_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (const __half*)x1, (__half*)grad_x2, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_mul_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (const __nv_bfloat16*)x1, (__nv_bfloat16*)grad_x2, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

int backwards_sub_x1_device(const void* grad_c, void* grad_x1, size_t size, DType dtype) {
    if (!grad_c || !grad_x1) return -1;

    size_t element_size = dtype_size(dtype);
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
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_sub_x2_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (double*)grad_x2, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_sub_x2_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (__half*)grad_x2, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_sub_x2_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (__nv_bfloat16*)grad_x2, size);
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
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_div_x1_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)x2, (double*)grad_x1, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_div_x1_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (const __half*)x2, (__half*)grad_x1, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_div_x1_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (const __nv_bfloat16*)x2, (__nv_bfloat16*)grad_x1, size);
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
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_div_x2_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)x1, (const double*)x2, (double*)grad_x2, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_div_x2_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (const __half*)x1, (const __half*)x2, (__half*)grad_x2, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_div_x2_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (const __nv_bfloat16*)x1, (const __nv_bfloat16*)x2, (__nv_bfloat16*)grad_x2, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

template<typename T>
__global__ void backwards_exp_kernel(const T* grad_c, const T* fn_output, T* grad_x, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_x[idx] = grad_c[idx] * fn_output[idx];
    }
}

int backwards_exp_device(const void* grad_c, const void* fn_output, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !fn_output || !grad_x) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_exp_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (const float*)fn_output, (float*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_exp_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)fn_output, (double*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_exp_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (const __half*)fn_output, (__half*)grad_x, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_exp_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (const __nv_bfloat16*)fn_output, (__nv_bfloat16*)grad_x, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

template<typename T>
__global__ void backwards_log_kernel(const T* grad_c, const T* x, T* grad_x, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_x[idx] = grad_c[idx] / x[idx];
    }
}

int backwards_log_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_log_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (const float*)x, (float*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_log_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)x, (double*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_log_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (const __half*)x, (__half*)grad_x, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_log_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (const __nv_bfloat16*)x, (__nv_bfloat16*)grad_x, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

template<typename T>
__global__ void backwards_sqrt_kernel(const T* grad_c, const T* fn_output, T* grad_x, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_x[idx] = T(0.5) * grad_c[idx] / fn_output[idx];
    }
}

int backwards_sqrt_device(const void* grad_c, const void* fn_output, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !fn_output || !grad_x) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_sqrt_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (const float*)fn_output, (float*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_sqrt_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)fn_output, (double*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_sqrt_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (const __half*)fn_output, (__half*)grad_x, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_sqrt_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (const __nv_bfloat16*)fn_output, (__nv_bfloat16*)grad_x, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

template<typename T>
__global__ void backwards_tanh_kernel(const T* grad_c, const T* fn_output, T* grad_x, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_x[idx] = grad_c[idx] * (T(1.0) - fn_output[idx] * fn_output[idx]);
    }
}

int backwards_tanh_device(const void* grad_c, const void* fn_output, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !fn_output || !grad_x) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_tanh_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (const float*)fn_output, (float*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_tanh_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)fn_output, (double*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_tanh_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (const __half*)fn_output, (__half*)grad_x, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_tanh_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (const __nv_bfloat16*)fn_output, (__nv_bfloat16*)grad_x, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

template<typename T>
__global__ void backwards_power_x1_kernel(const T* grad_c, const T* x1, const T* x2, const T* out, T* grad_x1, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_x1[idx] = grad_c[idx] * x2[idx] * out[idx] / x1[idx];
    }
}

template<typename T>
__global__ void backwards_power_x2_kernel(const T* grad_c, const T* x1, const T* out, T* grad_x2, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_x2[idx] = grad_c[idx] * out[idx] * rp_log(x1[idx]);
    }
}

int backwards_power_x1_device(const void* grad_c, const void* x1, const void* x2, const void* out, void* grad_x1, size_t size, DType dtype) {
    if (!grad_c || !x1 || !x2 || !out || !grad_x1) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_power_x1_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (const float*)x1, (const float*)x2, (const float*)out, (float*)grad_x1, size);
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_power_x1_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)x1, (const double*)x2, (const double*)out, (double*)grad_x1, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_power_x1_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (const __half*)x1, (const __half*)x2, (const __half*)out, (__half*)grad_x1, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_power_x1_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (const __nv_bfloat16*)x1, (const __nv_bfloat16*)x2, (const __nv_bfloat16*)out, (__nv_bfloat16*)grad_x1, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

int backwards_power_x2_device(const void* grad_c, const void* x1, const void* out, void* grad_x2, size_t size, DType dtype) {
    if (!grad_c || !x1 || !out || !grad_x2) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_power_x2_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (const float*)x1, (const float*)out, (float*)grad_x2, size);
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_power_x2_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)x1, (const double*)out, (double*)grad_x2, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_power_x2_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (const __half*)x1, (const __half*)out, (__half*)grad_x2, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_power_x2_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (const __nv_bfloat16*)x1, (const __nv_bfloat16*)out, (__nv_bfloat16*)grad_x2, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

template<typename T>
__global__ void backwards_logb_x1_kernel(const T* grad_c, const T* x1, const T* x2, T* grad_x1, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_x1[idx] = grad_c[idx] / (x1[idx] * rp_log(x2[idx]));
    }
}

template<typename T>
__global__ void backwards_logb_x2_kernel(const T* grad_c, const T* x1, const T* x2, T* grad_x2, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T log_x2 = rp_log(x2[idx]);
        grad_x2[idx] = -grad_c[idx] * rp_log(x1[idx]) / (x2[idx] * log_x2 * log_x2);
    }
}

int backwards_logb_x1_device(const void* grad_c, const void* x1, const void* x2, void* grad_x1, size_t size, DType dtype) {
    if (!grad_c || !x1 || !x2 || !grad_x1) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_logb_x1_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (const float*)x1, (const float*)x2, (float*)grad_x1, size);
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_logb_x1_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)x1, (const double*)x2, (double*)grad_x1, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_logb_x1_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (const __half*)x1, (const __half*)x2, (__half*)grad_x1, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_logb_x1_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (const __nv_bfloat16*)x1, (const __nv_bfloat16*)x2, (__nv_bfloat16*)grad_x1, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

int backwards_logb_x2_device(const void* grad_c, const void* x1, const void* x2, void* grad_x2, size_t size, DType dtype) {
    if (!grad_c || !x1 || !x2 || !grad_x2) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_logb_x2_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (const float*)x1, (const float*)x2, (float*)grad_x2, size);
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_logb_x2_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)x1, (const double*)x2, (double*)grad_x2, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_logb_x2_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (const __half*)x1, (const __half*)x2, (__half*)grad_x2, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_logb_x2_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (const __nv_bfloat16*)x1, (const __nv_bfloat16*)x2, (__nv_bfloat16*)grad_x2, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

template<typename T>
__global__ void backwards_relu_kernel(const T* grad_c, const T* x, T* grad_x, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_x[idx] = (x[idx] > T(0)) ? grad_c[idx] : T(0);
    }
}

int backwards_relu_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_relu_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (const float*)x, (float*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_relu_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)x, (double*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_relu_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (const __half*)x, (__half*)grad_x, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_relu_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (const __nv_bfloat16*)x, (__nv_bfloat16*)grad_x, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

template<typename T>
__global__ void backwards_sigmoid_kernel(const T* grad_c, const T* fn_output, T* grad_x, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_x[idx] = grad_c[idx] * fn_output[idx] * (T(1) - fn_output[idx]);
    }
}

int backwards_sigmoid_device(const void* grad_c, const void* fn_output, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !fn_output || !grad_x) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_sigmoid_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (const float*)fn_output, (float*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_sigmoid_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)fn_output, (double*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_sigmoid_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (const __half*)fn_output, (__half*)grad_x, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_sigmoid_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (const __nv_bfloat16*)fn_output, (__nv_bfloat16*)grad_x, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

template<typename T>
__global__ void backwards_square_kernel(const T* grad_c, const T* x, T* grad_x, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_x[idx] = grad_c[idx] * T(2) * x[idx];
    }
}

int backwards_square_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_square_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (const float*)x, (float*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_square_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)x, (double*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_square_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (const __half*)x, (__half*)grad_x, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_square_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (const __nv_bfloat16*)x, (__nv_bfloat16*)grad_x, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

template<typename T>
__global__ void backwards_abs_kernel(const T* grad_c, const T* x, T* grad_x, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (x[idx] > T(0)) {
            grad_x[idx] = grad_c[idx];
        } else if (x[idx] < T(0)) {
            grad_x[idx] = -grad_c[idx];
        } else {
            grad_x[idx] = T(0);
        }
    }
}

int backwards_abs_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_abs_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (const float*)x, (float*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_abs_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)x, (double*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_abs_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (const __half*)x, (__half*)grad_x, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_abs_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (const __nv_bfloat16*)x, (__nv_bfloat16*)grad_x, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

template<typename T>
__global__ void backwards_sin_kernel(const T* grad_c, const T* x, T* grad_x, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_x[idx] = grad_c[idx] * rp_cos(x[idx]);
    }
}

int backwards_sin_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_sin_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (const float*)x, (float*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_sin_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)x, (double*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_sin_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (const __half*)x, (__half*)grad_x, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_sin_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (const __nv_bfloat16*)x, (__nv_bfloat16*)grad_x, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

template<typename T>
__global__ void backwards_cos_kernel(const T* grad_c, const T* x, T* grad_x, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_x[idx] = -grad_c[idx] * rp_sin(x[idx]);
    }
}

int backwards_cos_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_cos_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (const float*)x, (float*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_cos_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)x, (double*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_cos_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (const __half*)x, (__half*)grad_x, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_cos_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (const __nv_bfloat16*)x, (__nv_bfloat16*)grad_x, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

template<typename T>
__global__ void backwards_tan_kernel(const T* grad_c, const T* x, T* grad_x, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T cos_x = rp_cos(x[idx]);
        grad_x[idx] = grad_c[idx] / (cos_x * cos_x);
    }
}

int backwards_tan_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_tan_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (const float*)x, (float*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_tan_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)x, (double*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_tan_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (const __half*)x, (__half*)grad_x, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_tan_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (const __nv_bfloat16*)x, (__nv_bfloat16*)grad_x, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

template<typename T>
__global__ void backwards_asin_kernel(const T* grad_c, const T* x, T* grad_x, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_x[idx] = grad_c[idx] / rp_sqrt(T(1) - x[idx] * x[idx]);
    }
}

int backwards_asin_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_asin_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (const float*)x, (float*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_asin_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)x, (double*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_asin_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (const __half*)x, (__half*)grad_x, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_asin_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (const __nv_bfloat16*)x, (__nv_bfloat16*)grad_x, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

template<typename T>
__global__ void backwards_acos_kernel(const T* grad_c, const T* x, T* grad_x, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_x[idx] = -grad_c[idx] / rp_sqrt(T(1) - x[idx] * x[idx]);
    }
}

int backwards_acos_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_acos_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (const float*)x, (float*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_acos_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)x, (double*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_acos_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (const __half*)x, (__half*)grad_x, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_acos_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (const __nv_bfloat16*)x, (__nv_bfloat16*)grad_x, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

template<typename T>
__global__ void backwards_atan_kernel(const T* grad_c, const T* x, T* grad_x, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_x[idx] = grad_c[idx] / (T(1) + x[idx] * x[idx]);
    }
}

int backwards_atan_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_atan_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (const float*)x, (float*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_atan_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)x, (double*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_atan_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (const __half*)x, (__half*)grad_x, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_atan_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (const __nv_bfloat16*)x, (__nv_bfloat16*)grad_x, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

template<typename T>
__global__ void backwards_sinh_kernel(const T* grad_c, const T* x, T* grad_x, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_x[idx] = grad_c[idx] * rp_cosh(x[idx]);
    }
}

int backwards_sinh_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_sinh_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (const float*)x, (float*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_sinh_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)x, (double*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_sinh_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (const __half*)x, (__half*)grad_x, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_sinh_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (const __nv_bfloat16*)x, (__nv_bfloat16*)grad_x, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

template<typename T>
__global__ void backwards_cosh_kernel(const T* grad_c, const T* x, T* grad_x, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_x[idx] = grad_c[idx] * rp_sinh(x[idx]);
    }
}

int backwards_cosh_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_cosh_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (const float*)x, (float*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_cosh_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)x, (double*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_cosh_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (const __half*)x, (__half*)grad_x, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_cosh_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (const __nv_bfloat16*)x, (__nv_bfloat16*)grad_x, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

template<typename T>
__global__ void backwards_gelu_kernel(const T* grad_c, const T* x, T* grad_x, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T sqrt_2_over_pi = T(0.7978845608);
        T coeff = T(0.044715);
        T coeff3 = T(0.134145);

        T x_val = x[idx];
        T x_sq = x_val * x_val;
        T x_cubed = x_sq * x_val;
        T inner = sqrt_2_over_pi * (x_val + coeff * x_cubed);
        T tanh_inner = rp_tanh(inner);
        T sech2_inner = T(1) - tanh_inner * tanh_inner;
        T d_inner = sqrt_2_over_pi * (T(1) + coeff3 * x_sq);
        T gelu_grad = T(0.5) * (T(1) + tanh_inner + x_val * sech2_inner * d_inner);
        grad_x[idx] = grad_c[idx] * gelu_grad;
    }
}

int backwards_gelu_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_gelu_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (const float*)x, (float*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_gelu_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)x, (double*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_gelu_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (const __half*)x, (__half*)grad_x, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_gelu_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (const __nv_bfloat16*)x, (__nv_bfloat16*)grad_x, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

template<typename T>
__global__ void backwards_silu_kernel(const T* grad_c, const T* x, T* grad_x, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T x_val = x[idx];
        T sigmoid = T(1) / (T(1) + rp_exp(-x_val));
        T silu_grad = sigmoid * (T(1) + x_val * (T(1) - sigmoid));
        grad_x[idx] = grad_c[idx] * silu_grad;
    }
}

int backwards_silu_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_silu_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (const float*)x, (float*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_silu_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)x, (double*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_silu_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (const __half*)x, (__half*)grad_x, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_silu_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (const __nv_bfloat16*)x, (__nv_bfloat16*)grad_x, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

template<typename T>
__global__ void backwards_leaky_relu_kernel(const T* grad_c, const T* x, T alpha, T* grad_x, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_x[idx] = grad_c[idx] * ((x[idx] > T(0)) ? T(1) : alpha);
    }
}

int backwards_leaky_relu_device(const void* grad_c, const void* x, const void* alpha, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !alpha || !grad_x) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        float alpha_val = *(const float*)alpha;
        backwards_leaky_relu_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (const float*)x, alpha_val, (float*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        double alpha_val = *(const double*)alpha;
        backwards_leaky_relu_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)x, alpha_val, (double*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT16) {
        __half alpha_val = __float2half(*(const float*)alpha);
        backwards_leaky_relu_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (const __half*)x, alpha_val, (__half*)grad_x, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        __nv_bfloat16 alpha_val = __float2bfloat16(*(const float*)alpha);
        backwards_leaky_relu_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (const __nv_bfloat16*)x, alpha_val, (__nv_bfloat16*)grad_x, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

template<typename T>
__global__ void backwards_rsqrt_kernel(const T* grad_c, const T* x, T* grad_x, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T x_val = x[idx];
        grad_x[idx] = grad_c[idx] * T(-0.5) / (x_val * rp_sqrt(x_val));
    }
}

int backwards_rsqrt_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_rsqrt_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (const float*)x, (float*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_rsqrt_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (const double*)x, (double*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_rsqrt_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (const __half*)x, (__half*)grad_x, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_rsqrt_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (const __nv_bfloat16*)x, (__nv_bfloat16*)grad_x, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

template<typename T>
__global__ void backwards_sum_all_kernel(const T* grad_c, T* grad_x, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_x[idx] = grad_c[0];
    }
}

int backwards_sum_all_device(const void* grad_c, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !grad_x) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_sum_all_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (float*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_sum_all_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (double*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_sum_all_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (__half*)grad_x, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_sum_all_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (__nv_bfloat16*)grad_x, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}

template<typename T>
__global__ void backwards_mean_all_kernel(const T* grad_c, T* grad_x, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_x[idx] = grad_c[0] / T(size);
    }
}

int backwards_mean_all_device(const void* grad_c, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !grad_x) return -1;

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    if (dtype == DTYPE_FLOAT32) {
        backwards_mean_all_kernel<float><<<num_blocks, block_size>>>(
            (const float*)grad_c, (float*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        backwards_mean_all_kernel<double><<<num_blocks, block_size>>>(
            (const double*)grad_c, (double*)grad_x, size);
    } else if (dtype == DTYPE_FLOAT16) {
        backwards_mean_all_kernel<__half><<<num_blocks, block_size>>>(
            (const __half*)grad_c, (__half*)grad_x, size);
    } else if (dtype == DTYPE_BFLOAT16) {
        backwards_mean_all_kernel<__nv_bfloat16><<<num_blocks, block_size>>>(
            (const __nv_bfloat16*)grad_c, (__nv_bfloat16*)grad_x, size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}
