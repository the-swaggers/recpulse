#include "functional.h"
#include "../core/cuda_helpers.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

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
        out[idx] = pow(a[idx], b[idx]);
    }
}

template<typename T>
__global__ void logb_kernel(T* out, const T* a, const T* b, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = log(a[idx]) / log(b[idx]);
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
        out[idx] = pow(a[idx], scalar);
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
        out[idx] = pow(scalar, a[idx]);
    }
}

template<typename T>
__global__ void logb_scalar_kernel(T* out, const T* a, T scalar, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = log(a[idx]) / log(scalar);
    }
}

template<typename T>
__global__ void rlogb_scalar_kernel(T* out, T scalar, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = log(scalar) / log(a[idx]);
    }
}

int add_kernel_device(void* out, const void* x1, const void* x2, size_t size, DType dtype) {
    if (!out || !x1 || !x2 || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        add_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x1, (const float*)x2, size);
    } else if (dtype == DTYPE_FLOAT64) {
        add_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x1, (const double*)x2, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int sub_kernel_device(void* out, const void* x1, const void* x2, size_t size, DType dtype) {
    if (!out || !x1 || !x2 || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        sub_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x1, (const float*)x2, size);
    } else if (dtype == DTYPE_FLOAT64) {
        sub_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x1, (const double*)x2, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int mul_kernel_device(void* out, const void* x1, const void* x2, size_t size, DType dtype) {
    if (!out || !x1 || !x2 || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        mul_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x1, (const float*)x2, size);
    } else if (dtype == DTYPE_FLOAT64) {
        mul_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x1, (const double*)x2, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int div_kernel_device(void* out, const void* x1, const void* x2, size_t size, DType dtype) {
    if (!out || !x1 || !x2 || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        div_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x1, (const float*)x2, size);
    } else if (dtype == DTYPE_FLOAT64) {
        div_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x1, (const double*)x2, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int pow_kernel_device(void* out, const void* x1, const void* x2, size_t size, DType dtype) {
    if (!out || !x1 || !x2 || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        pow_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x1, (const float*)x2, size);
    } else if (dtype == DTYPE_FLOAT64) {
        pow_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x1, (const double*)x2, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int logb_kernel_device(void* out, const void* x1, const void* x2, size_t size, DType dtype) {
    if (!out || !x1 || !x2 || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        logb_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x1, (const float*)x2, size);
    } else if (dtype == DTYPE_FLOAT64) {
        logb_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x1, (const double*)x2, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int add_scalar_kernel_device(void* out, const void* x, const void* scalar, size_t size, DType dtype) {
    if (!out || !x || !scalar || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        float scalar_val = *(const float*)scalar;
        add_scalar_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x, scalar_val, size);
    } else if (dtype == DTYPE_FLOAT64) {
        double scalar_val = *(const double*)scalar;
        add_scalar_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x, scalar_val, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int sub_scalar_kernel_device(void* out, const void* x, const void* scalar, size_t size, DType dtype) {
    if (!out || !x || !scalar || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        float scalar_val = *(const float*)scalar;
        sub_scalar_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x, scalar_val, size);
    } else if (dtype == DTYPE_FLOAT64) {
        double scalar_val = *(const double*)scalar;
        sub_scalar_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x, scalar_val, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int mul_scalar_kernel_device(void* out, const void* x, const void* scalar, size_t size, DType dtype) {
    if (!out || !x || !scalar || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        float scalar_val = *(const float*)scalar;
        mul_scalar_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x, scalar_val, size);
    } else if (dtype == DTYPE_FLOAT64) {
        double scalar_val = *(const double*)scalar;
        mul_scalar_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x, scalar_val, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int div_scalar_kernel_device(void* out, const void* x, const void* scalar, size_t size, DType dtype) {
    if (!out || !x || !scalar || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        float scalar_val = *(const float*)scalar;
        div_scalar_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x, scalar_val, size);
    } else if (dtype == DTYPE_FLOAT64) {
        double scalar_val = *(const double*)scalar;
        div_scalar_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x, scalar_val, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int pow_scalar_kernel_device(void* out, const void* x, const void* scalar, size_t size, DType dtype) {
    if (!out || !x || !scalar || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        float scalar_val = *(const float*)scalar;
        pow_scalar_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x, scalar_val, size);
    } else if (dtype == DTYPE_FLOAT64) {
        double scalar_val = *(const double*)scalar;
        pow_scalar_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x, scalar_val, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int rsub_scalar_kernel_device(void* out, const void* scalar, const void* x, size_t size, DType dtype) {
    if (!out || !x || !scalar || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        float scalar_val = *(const float*)scalar;
        rsub_scalar_kernel<float><<<blocks, threads>>>((float*)out, scalar_val, (const float*)x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        double scalar_val = *(const double*)scalar;
        rsub_scalar_kernel<double><<<blocks, threads>>>((double*)out, scalar_val, (const double*)x, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int rdiv_scalar_kernel_device(void* out, const void* scalar, const void* x, size_t size, DType dtype) {
    if (!out || !x || !scalar || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        float scalar_val = *(const float*)scalar;
        rdiv_scalar_kernel<float><<<blocks, threads>>>((float*)out, scalar_val, (const float*)x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        double scalar_val = *(const double*)scalar;
        rdiv_scalar_kernel<double><<<blocks, threads>>>((double*)out, scalar_val, (const double*)x, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int rpow_scalar_kernel_device(void* out, const void* scalar, const void* x, size_t size, DType dtype) {
    if (!out || !x || !scalar || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        float scalar_val = *(const float*)scalar;
        rpow_scalar_kernel<float><<<blocks, threads>>>((float*)out, scalar_val, (const float*)x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        double scalar_val = *(const double*)scalar;
        rpow_scalar_kernel<double><<<blocks, threads>>>((double*)out, scalar_val, (const double*)x, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int logb_scalar_kernel_device(void* out, const void* x, const void* scalar, size_t size, DType dtype) {
    if (!out || !x || !scalar || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        float scalar_val = *(const float*)scalar;
        logb_scalar_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x, scalar_val, size);
    } else if (dtype == DTYPE_FLOAT64) {
        double scalar_val = *(const double*)scalar;
        logb_scalar_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x, scalar_val, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int rlogb_scalar_kernel_device(void* out, const void* scalar, const void* x, size_t size, DType dtype) {
    if (!out || !x || !scalar || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        float scalar_val = *(const float*)scalar;
        rlogb_scalar_kernel<float><<<blocks, threads>>>((float*)out, scalar_val, (const float*)x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        double scalar_val = *(const double*)scalar;
        rlogb_scalar_kernel<double><<<blocks, threads>>>((double*)out, scalar_val, (const double*)x, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

template<typename T>
__global__ void exp_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = exp(a[idx]);
    }
}

template<typename T>
__global__ void log_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = log(a[idx]);
    }
}

template<typename T>
__global__ void sqrt_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = sqrt(a[idx]);
    }
}

template<typename T>
__global__ void abs_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = fabs(a[idx]);
    }
}

template<typename T>
__global__ void sin_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = sin(a[idx]);
    }
}

template<typename T>
__global__ void cos_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = cos(a[idx]);
    }
}

template<typename T>
__global__ void tan_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = tan(a[idx]);
    }
}

template<typename T>
__global__ void asin_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = asin(a[idx]);
    }
}

template<typename T>
__global__ void acos_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = acos(a[idx]);
    }
}

template<typename T>
__global__ void atan_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = atan(a[idx]);
    }
}

template<typename T>
__global__ void sinh_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = sinh(a[idx]);
    }
}

template<typename T>
__global__ void cosh_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = cosh(a[idx]);
    }
}

template<typename T>
__global__ void tanh_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = tanh(a[idx]);
    }
}

template<typename T>
__global__ void relu_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = (a[idx] > (T)0) ? a[idx] : (T)0;
    }
}

template<typename T>
__global__ void sigmoid_kernel(T* out, const T* a, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = (T)1 / ((T)1 + exp(-a[idx]));
    }
}

template<typename T>
__global__ void leaky_relu_kernel(T* out, const T* a, T alpha, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = (a[idx] > (T)0) ? a[idx] : alpha * a[idx];
    }
}

int exp_kernel_device(void* out, const void* x, size_t size, DType dtype) {
    if (!out || !x || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        exp_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        exp_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int log_kernel_device(void* out, const void* x, size_t size, DType dtype) {
    if (!out || !x || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        log_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        log_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int sqrt_kernel_device(void* out, const void* x, size_t size, DType dtype) {
    if (!out || !x || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        sqrt_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        sqrt_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int abs_kernel_device(void* out, const void* x, size_t size, DType dtype) {
    if (!out || !x || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        abs_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        abs_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int sin_kernel_device(void* out, const void* x, size_t size, DType dtype) {
    if (!out || !x || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        sin_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        sin_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int cos_kernel_device(void* out, const void* x, size_t size, DType dtype) {
    if (!out || !x || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        cos_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        cos_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int tan_kernel_device(void* out, const void* x, size_t size, DType dtype) {
    if (!out || !x || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        tan_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        tan_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int asin_kernel_device(void* out, const void* x, size_t size, DType dtype) {
    if (!out || !x || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        asin_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        asin_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int acos_kernel_device(void* out, const void* x, size_t size, DType dtype) {
    if (!out || !x || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        acos_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        acos_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int atan_kernel_device(void* out, const void* x, size_t size, DType dtype) {
    if (!out || !x || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        atan_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        atan_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int sinh_kernel_device(void* out, const void* x, size_t size, DType dtype) {
    if (!out || !x || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        sinh_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        sinh_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int cosh_kernel_device(void* out, const void* x, size_t size, DType dtype) {
    if (!out || !x || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        cosh_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        cosh_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int tanh_kernel_device(void* out, const void* x, size_t size, DType dtype) {
    if (!out || !x || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        tanh_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        tanh_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int relu_kernel_device(void* out, const void* x, size_t size, DType dtype) {
    if (!out || !x || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        relu_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        relu_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int sigmoid_kernel_device(void* out, const void* x, size_t size, DType dtype) {
    if (!out || !x || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        sigmoid_kernel<float><<<blocks, threads>>>((float*)out, (const float*)x, size);
    } else if (dtype == DTYPE_FLOAT64) {
        sigmoid_kernel<double><<<blocks, threads>>>((double*)out, (const double*)x, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

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

int sum_all_kernel_device(void* out, const void* x, size_t size, DType dtype) {
    fprintf(stderr, "sum_all_kernel_device called: size=%zu, dtype=%d\n", size, dtype);
    fflush(stderr);

    if (!out || !x || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;
    size_t shared_mem_size = threads * (dtype == DTYPE_FLOAT32 ? sizeof(float) : sizeof(double));

    fprintf(stderr, "Launching kernel: blocks=%zu, threads=%zu, shared_mem=%zu\n", blocks, threads, shared_mem_size);
    fflush(stderr);

    if (dtype == DTYPE_FLOAT32) {
        float* d_partial_sums;
        fprintf(stderr, "Allocating %zu bytes for partial sums\n", blocks * sizeof(float));
        fflush(stderr);
        cudaMalloc(&d_partial_sums, blocks * sizeof(float));

        sum_reduction_kernel<float><<<blocks, threads, shared_mem_size>>>(
            (const float*)x, d_partial_sums, size
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error in first reduction: %s\n", cudaGetErrorString(err));
            fflush(stderr);
            cudaFree(d_partial_sums);
            return -1;
        }

        if (blocks > 1) {
            size_t threads2 = (blocks < threads) ? blocks : threads;
            size_t shared_mem_size2 = threads2 * sizeof(float);
            fprintf(stderr, "Second reduction: blocks=%zu, threads2=%zu, shared_mem=%zu\n", blocks, threads2, shared_mem_size2);
            fflush(stderr);
            sum_reduction_kernel<float><<<1, threads2, shared_mem_size2>>>(
                d_partial_sums, (float*)out, blocks
            );
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA error in second reduction: %s\n", cudaGetErrorString(err));
                fflush(stderr);
                cudaFree(d_partial_sums);
                return -1;
            }
        } else {
            fprintf(stderr, "Using single block optimization\n");
            fflush(stderr);
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
            printf("CUDA error in first reduction: %s\n", cudaGetErrorString(err));
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
                printf("CUDA error in second reduction: %s\n", cudaGetErrorString(err));
                cudaFree(d_partial_sums);
                return -1;
            }
        } else {
            cudaMemcpy(out, d_partial_sums, sizeof(double), cudaMemcpyDeviceToDevice);
        }

        cudaFree(d_partial_sums);
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
