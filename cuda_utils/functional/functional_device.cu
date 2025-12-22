#include "functional.h"
#include "../core/cuda_helpers.h"
#include <cuda_runtime.h>

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

int add_kernel_device(void* out, const void* a, const void* b, size_t size, DType dtype) {
    if (!out || !a || !b || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        add_kernel<float><<<blocks, threads>>>((float*)out, (const float*)a, (const float*)b, size);
    } else if (dtype == DTYPE_FLOAT64) {
        add_kernel<double><<<blocks, threads>>>((double*)out, (const double*)a, (const double*)b, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int sub_kernel_device(void* out, const void* a, const void* b, size_t size, DType dtype) {
    if (!out || !a || !b || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        sub_kernel<float><<<blocks, threads>>>((float*)out, (const float*)a, (const float*)b, size);
    } else if (dtype == DTYPE_FLOAT64) {
        sub_kernel<double><<<blocks, threads>>>((double*)out, (const double*)a, (const double*)b, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int mul_kernel_device(void* out, const void* a, const void* b, size_t size, DType dtype) {
    if (!out || !a || !b || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        mul_kernel<float><<<blocks, threads>>>((float*)out, (const float*)a, (const float*)b, size);
    } else if (dtype == DTYPE_FLOAT64) {
        mul_kernel<double><<<blocks, threads>>>((double*)out, (const double*)a, (const double*)b, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int div_kernel_device(void* out, const void* a, const void* b, size_t size, DType dtype) {
    if (!out || !a || !b || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        div_kernel<float><<<blocks, threads>>>((float*)out, (const float*)a, (const float*)b, size);
    } else if (dtype == DTYPE_FLOAT64) {
        div_kernel<double><<<blocks, threads>>>((double*)out, (const double*)a, (const double*)b, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int pow_kernel_device(void* out, const void* a, const void* b, size_t size, DType dtype) {
    if (!out || !a || !b || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        pow_kernel<float><<<blocks, threads>>>((float*)out, (const float*)a, (const float*)b, size);
    } else if (dtype == DTYPE_FLOAT64) {
        pow_kernel<double><<<blocks, threads>>>((double*)out, (const double*)a, (const double*)b, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int logb_kernel_device(void* out, const void* a, const void* b, size_t size, DType dtype) {
    if (!out || !a || !b || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        logb_kernel<float><<<blocks, threads>>>((float*)out, (const float*)a, (const float*)b, size);
    } else if (dtype == DTYPE_FLOAT64) {
        logb_kernel<double><<<blocks, threads>>>((double*)out, (const double*)a, (const double*)b, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int add_scalar_kernel_device(void* out, const void* a, const void* scalar, size_t size, DType dtype) {
    if (!out || !a || !scalar || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        float scalar_val = *(const float*)scalar;
        add_scalar_kernel<float><<<blocks, threads>>>((float*)out, (const float*)a, scalar_val, size);
    } else if (dtype == DTYPE_FLOAT64) {
        double scalar_val = *(const double*)scalar;
        add_scalar_kernel<double><<<blocks, threads>>>((double*)out, (const double*)a, scalar_val, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int sub_scalar_kernel_device(void* out, const void* a, const void* scalar, size_t size, DType dtype) {
    if (!out || !a || !scalar || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        float scalar_val = *(const float*)scalar;
        sub_scalar_kernel<float><<<blocks, threads>>>((float*)out, (const float*)a, scalar_val, size);
    } else if (dtype == DTYPE_FLOAT64) {
        double scalar_val = *(const double*)scalar;
        sub_scalar_kernel<double><<<blocks, threads>>>((double*)out, (const double*)a, scalar_val, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int mul_scalar_kernel_device(void* out, const void* a, const void* scalar, size_t size, DType dtype) {
    if (!out || !a || !scalar || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        float scalar_val = *(const float*)scalar;
        mul_scalar_kernel<float><<<blocks, threads>>>((float*)out, (const float*)a, scalar_val, size);
    } else if (dtype == DTYPE_FLOAT64) {
        double scalar_val = *(const double*)scalar;
        mul_scalar_kernel<double><<<blocks, threads>>>((double*)out, (const double*)a, scalar_val, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int div_scalar_kernel_device(void* out, const void* a, const void* scalar, size_t size, DType dtype) {
    if (!out || !a || !scalar || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        float scalar_val = *(const float*)scalar;
        div_scalar_kernel<float><<<blocks, threads>>>((float*)out, (const float*)a, scalar_val, size);
    } else if (dtype == DTYPE_FLOAT64) {
        double scalar_val = *(const double*)scalar;
        div_scalar_kernel<double><<<blocks, threads>>>((double*)out, (const double*)a, scalar_val, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int pow_scalar_kernel_device(void* out, const void* a, const void* scalar, size_t size, DType dtype) {
    if (!out || !a || !scalar || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        float scalar_val = *(const float*)scalar;
        pow_scalar_kernel<float><<<blocks, threads>>>((float*)out, (const float*)a, scalar_val, size);
    } else if (dtype == DTYPE_FLOAT64) {
        double scalar_val = *(const double*)scalar;
        pow_scalar_kernel<double><<<blocks, threads>>>((double*)out, (const double*)a, scalar_val, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int rsub_scalar_kernel_device(void* out, const void* scalar, const void* a, size_t size, DType dtype) {
    if (!out || !a || !scalar || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        float scalar_val = *(const float*)scalar;
        rsub_scalar_kernel<float><<<blocks, threads>>>((float*)out, scalar_val, (const float*)a, size);
    } else if (dtype == DTYPE_FLOAT64) {
        double scalar_val = *(const double*)scalar;
        rsub_scalar_kernel<double><<<blocks, threads>>>((double*)out, scalar_val, (const double*)a, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int rdiv_scalar_kernel_device(void* out, const void* scalar, const void* a, size_t size, DType dtype) {
    if (!out || !a || !scalar || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        float scalar_val = *(const float*)scalar;
        rdiv_scalar_kernel<float><<<blocks, threads>>>((float*)out, scalar_val, (const float*)a, size);
    } else if (dtype == DTYPE_FLOAT64) {
        double scalar_val = *(const double*)scalar;
        rdiv_scalar_kernel<double><<<blocks, threads>>>((double*)out, scalar_val, (const double*)a, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int rpow_scalar_kernel_device(void* out, const void* scalar, const void* a, size_t size, DType dtype) {
    if (!out || !a || !scalar || size == 0) return -1;

    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    if (dtype == DTYPE_FLOAT32) {
        float scalar_val = *(const float*)scalar;
        rpow_scalar_kernel<float><<<blocks, threads>>>((float*)out, scalar_val, (const float*)a, size);
    } else if (dtype == DTYPE_FLOAT64) {
        double scalar_val = *(const double*)scalar;
        rpow_scalar_kernel<double><<<blocks, threads>>>((double*)out, scalar_val, (const double*)a, size);
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}
