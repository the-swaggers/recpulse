#include "functional.h"
#include "../core/cuda_helpers.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

int rp_add(void* out, const void* x1, const void* x2, size_t size, DType dtype, int device_id) {
    if (!out || !x1 || !x2 || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return add_kernel_host_f32((float*)out, (const float*)x1, (const float*)x2, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return add_kernel_host_f64((double*)out, (const double*)x1, (const double*)x2, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return add_kernel_device(out, x1, x2, size, dtype);
}

int rp_sub(void* out, const void* x1, const void* x2, size_t size, DType dtype, int device_id) {
    if (!out || !x1 || !x2 || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return sub_kernel_host_f32((float*)out, (const float*)x1, (const float*)x2, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return sub_kernel_host_f64((double*)out, (const double*)x1, (const double*)x2, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return sub_kernel_device(out, x1, x2, size, dtype);
}

int rp_mul(void* out, const void* x1, const void* x2, size_t size, DType dtype, int device_id) {
    if (!out || !x1 || !x2 || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return mul_kernel_host_f32((float*)out, (const float*)x1, (const float*)x2, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return mul_kernel_host_f64((double*)out, (const double*)x1, (const double*)x2, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return mul_kernel_device(out, x1, x2, size, dtype);
}

int rp_divide(void* out, const void* x1, const void* x2, size_t size, DType dtype, int device_id) {
    if (!out || !x1 || !x2 || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return div_kernel_host_f32((float*)out, (const float*)x1, (const float*)x2, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return div_kernel_host_f64((double*)out, (const double*)x1, (const double*)x2, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return div_kernel_device(out, x1, x2, size, dtype);
}

int rp_power(void* out, const void* x1, const void* x2, size_t size, DType dtype, int device_id) {
    if (!out || !x1 || !x2 || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return pow_kernel_host_f32((float*)out, (const float*)x1, (const float*)x2, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return pow_kernel_host_f64((double*)out, (const double*)x1, (const double*)x2, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return pow_kernel_device(out, x1, x2, size, dtype);
}

int rp_logb(void* out, const void* x1, const void* x2, size_t size, DType dtype, int device_id) {
    if (!out || !x1 || !x2 || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return logb_kernel_host_f32((float*)out, (const float*)x1, (const float*)x2, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return logb_kernel_host_f64((double*)out, (const double*)x1, (const double*)x2, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return logb_kernel_device(out, x1, x2, size, dtype);
}

#define STRIDED_BINARY_DISPATCHER(name, host_f32, host_f64, device_fn) \
int name(void* out, const void* x1, const void* x2, int ndim, const int* out_shape, \
         const int* x1_strides, const int* x2_strides, size_t out_size, DType dtype, int device_id) { \
    if (!out || !x1 || !x2 || out_size == 0) return -1; \
    if (device_id == -1) { \
        if (dtype == DTYPE_FLOAT32) { \
            return host_f32((float*)out, (const float*)x1, (const float*)x2, ndim, out_shape, x1_strides, x2_strides, out_size); \
        } else if (dtype == DTYPE_FLOAT64) { \
            return host_f64((double*)out, (const double*)x1, (const double*)x2, ndim, out_shape, x1_strides, x2_strides, out_size); \
        } \
        return -1; \
    } \
    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1; \
    return device_fn(out, x1, x2, ndim, out_shape, x1_strides, x2_strides, out_size, dtype); \
}

STRIDED_BINARY_DISPATCHER(rp_add_strided, add_strided_kernel_host_f32, add_strided_kernel_host_f64, add_strided_kernel_device)
STRIDED_BINARY_DISPATCHER(rp_sub_strided, sub_strided_kernel_host_f32, sub_strided_kernel_host_f64, sub_strided_kernel_device)
STRIDED_BINARY_DISPATCHER(rp_mul_strided, mul_strided_kernel_host_f32, mul_strided_kernel_host_f64, mul_strided_kernel_device)
STRIDED_BINARY_DISPATCHER(rp_divide_strided, div_strided_kernel_host_f32, div_strided_kernel_host_f64, div_strided_kernel_device)
STRIDED_BINARY_DISPATCHER(rp_power_strided, pow_strided_kernel_host_f32, pow_strided_kernel_host_f64, pow_strided_kernel_device)
STRIDED_BINARY_DISPATCHER(rp_logb_strided, logb_strided_kernel_host_f32, logb_strided_kernel_host_f64, logb_strided_kernel_device)

int rp_add_scalar(void* out, const void* x, const void* scalar, size_t size, DType dtype, int device_id) {
    if (!out || !x || !scalar || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return add_scalar_kernel_host_f32((float*)out, (const float*)x, *(const float*)scalar, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return add_scalar_kernel_host_f64((double*)out, (const double*)x, *(const double*)scalar, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return add_scalar_kernel_device(out, x, scalar, size, dtype);
}

int rp_sub_scalar(void* out, const void* x, const void* scalar, size_t size, DType dtype, int device_id) {
    if (!out || !x || !scalar || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return sub_scalar_kernel_host_f32((float*)out, (const float*)x, *(const float*)scalar, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return sub_scalar_kernel_host_f64((double*)out, (const double*)x, *(const double*)scalar, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return sub_scalar_kernel_device(out, x, scalar, size, dtype);
}

int rp_mul_scalar(void* out, const void* x, const void* scalar, size_t size, DType dtype, int device_id) {
    if (!out || !x || !scalar || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return mul_scalar_kernel_host_f32((float*)out, (const float*)x, *(const float*)scalar, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return mul_scalar_kernel_host_f64((double*)out, (const double*)x, *(const double*)scalar, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return mul_scalar_kernel_device(out, x, scalar, size, dtype);
}

int rp_div_scalar(void* out, const void* x, const void* scalar, size_t size, DType dtype, int device_id) {
    if (!out || !x || !scalar || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return div_scalar_kernel_host_f32((float*)out, (const float*)x, *(const float*)scalar, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return div_scalar_kernel_host_f64((double*)out, (const double*)x, *(const double*)scalar, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return div_scalar_kernel_device(out, x, scalar, size, dtype);
}

int rp_pow_scalar(void* out, const void* x, const void* scalar, size_t size, DType dtype, int device_id) {
    if (!out || !x || !scalar || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return pow_scalar_kernel_host_f32((float*)out, (const float*)x, *(const float*)scalar, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return pow_scalar_kernel_host_f64((double*)out, (const double*)x, *(const double*)scalar, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return pow_scalar_kernel_device(out, x, scalar, size, dtype);
}

int rp_rsub_scalar(void* out, const void* scalar, const void* x, size_t size, DType dtype, int device_id) {
    if (!out || !x || !scalar || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return rsub_scalar_kernel_host_f32((float*)out, *(const float*)scalar, (const float*)x, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return rsub_scalar_kernel_host_f64((double*)out, *(const double*)scalar, (const double*)x, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return rsub_scalar_kernel_device(out, scalar, x, size, dtype);
}

int rp_rdiv_scalar(void* out, const void* scalar, const void* x, size_t size, DType dtype, int device_id) {
    if (!out || !x || !scalar || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return rdiv_scalar_kernel_host_f32((float*)out, *(const float*)scalar, (const float*)x, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return rdiv_scalar_kernel_host_f64((double*)out, *(const double*)scalar, (const double*)x, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return rdiv_scalar_kernel_device(out, scalar, x, size, dtype);
}

int rp_rpow_scalar(void* out, const void* scalar, const void* x, size_t size, DType dtype, int device_id) {
    if (!out || !x || !scalar || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return rpow_scalar_kernel_host_f32((float*)out, *(const float*)scalar, (const float*)x, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return rpow_scalar_kernel_host_f64((double*)out, *(const double*)scalar, (const double*)x, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return rpow_scalar_kernel_device(out, scalar, x, size, dtype);
}

int rp_logb_scalar(void* out, const void* x, const void* scalar, size_t size, DType dtype, int device_id) {
    if (!out || !x || !scalar || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return logb_scalar_kernel_host_f32((float*)out, (const float*)x, *(const float*)scalar, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return logb_scalar_kernel_host_f64((double*)out, (const double*)x, *(const double*)scalar, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return logb_scalar_kernel_device(out, x, scalar, size, dtype);
}

int rp_rlogb_scalar(void* out, const void* scalar, const void* x, size_t size, DType dtype, int device_id) {
    if (!out || !x || !scalar || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return rlogb_scalar_kernel_host_f32((float*)out, *(const float*)scalar, (const float*)x, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return rlogb_scalar_kernel_host_f64((double*)out, *(const double*)scalar, (const double*)x, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return rlogb_scalar_kernel_device(out, scalar, x, size, dtype);
}

int rp_exp(void* out, const void* x, size_t size, DType dtype, int device_id) {
    if (!out || !x || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return exp_kernel_host_f32((float*)out, (const float*)x, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return exp_kernel_host_f64((double*)out, (const double*)x, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return exp_kernel_device(out, x, size, dtype);
}

int rp_log(void* out, const void* x, size_t size, DType dtype, int device_id) {
    if (!out || !x || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return log_kernel_host_f32((float*)out, (const float*)x, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return log_kernel_host_f64((double*)out, (const double*)x, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return log_kernel_device(out, x, size, dtype);
}

int rp_sqrt(void* out, const void* x, size_t size, DType dtype, int device_id) {
    if (!out || !x || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return sqrt_kernel_host_f32((float*)out, (const float*)x, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return sqrt_kernel_host_f64((double*)out, (const double*)x, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return sqrt_kernel_device(out, x, size, dtype);
}

int rp_abs(void* out, const void* x, size_t size, DType dtype, int device_id) {
    if (!out || !x || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return abs_kernel_host_f32((float*)out, (const float*)x, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return abs_kernel_host_f64((double*)out, (const double*)x, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return abs_kernel_device(out, x, size, dtype);
}

int rp_sin(void* out, const void* x, size_t size, DType dtype, int device_id) {
    if (!out || !x || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return sin_kernel_host_f32((float*)out, (const float*)x, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return sin_kernel_host_f64((double*)out, (const double*)x, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return sin_kernel_device(out, x, size, dtype);
}

int rp_cos(void* out, const void* x, size_t size, DType dtype, int device_id) {
    if (!out || !x || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return cos_kernel_host_f32((float*)out, (const float*)x, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return cos_kernel_host_f64((double*)out, (const double*)x, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return cos_kernel_device(out, x, size, dtype);
}

int rp_tan(void* out, const void* x, size_t size, DType dtype, int device_id) {
    if (!out || !x || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return tan_kernel_host_f32((float*)out, (const float*)x, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return tan_kernel_host_f64((double*)out, (const double*)x, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return tan_kernel_device(out, x, size, dtype);
}

int rp_asin(void* out, const void* x, size_t size, DType dtype, int device_id) {
    if (!out || !x || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return asin_kernel_host_f32((float*)out, (const float*)x, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return asin_kernel_host_f64((double*)out, (const double*)x, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return asin_kernel_device(out, x, size, dtype);
}

int rp_acos(void* out, const void* x, size_t size, DType dtype, int device_id) {
    if (!out || !x || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return acos_kernel_host_f32((float*)out, (const float*)x, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return acos_kernel_host_f64((double*)out, (const double*)x, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return acos_kernel_device(out, x, size, dtype);
}

int rp_atan(void* out, const void* x, size_t size, DType dtype, int device_id) {
    if (!out || !x || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return atan_kernel_host_f32((float*)out, (const float*)x, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return atan_kernel_host_f64((double*)out, (const double*)x, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return atan_kernel_device(out, x, size, dtype);
}

int rp_sinh(void* out, const void* x, size_t size, DType dtype, int device_id) {
    if (!out || !x || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return sinh_kernel_host_f32((float*)out, (const float*)x, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return sinh_kernel_host_f64((double*)out, (const double*)x, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return sinh_kernel_device(out, x, size, dtype);
}

int rp_cosh(void* out, const void* x, size_t size, DType dtype, int device_id) {
    if (!out || !x || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return cosh_kernel_host_f32((float*)out, (const float*)x, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return cosh_kernel_host_f64((double*)out, (const double*)x, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return cosh_kernel_device(out, x, size, dtype);
}

int rp_tanh(void* out, const void* x, size_t size, DType dtype, int device_id) {
    if (!out || !x || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return tanh_kernel_host_f32((float*)out, (const float*)x, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return tanh_kernel_host_f64((double*)out, (const double*)x, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return tanh_kernel_device(out, x, size, dtype);
}

int rp_relu(void* out, const void* x, size_t size, DType dtype, int device_id) {
    if (!out || !x || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return relu_kernel_host_f32((float*)out, (const float*)x, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return relu_kernel_host_f64((double*)out, (const double*)x, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return relu_kernel_device(out, x, size, dtype);
}

int rp_sigmoid(void* out, const void* x, size_t size, DType dtype, int device_id) {
    if (!out || !x || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return sigmoid_kernel_host_f32((float*)out, (const float*)x, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return sigmoid_kernel_host_f64((double*)out, (const double*)x, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return sigmoid_kernel_device(out, x, size, dtype);
}

int rp_leaky_relu(void* out, const void* x, const void* alpha, size_t size, DType dtype, int device_id) {
    if (!out || !x || !alpha || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return leaky_relu_kernel_host_f32((float*)out, (const float*)x, *(const float*)alpha, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return leaky_relu_kernel_host_f64((double*)out, (const double*)x, *(const double*)alpha, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return leaky_relu_kernel_device(out, x, alpha, size, dtype);
}

int rp_square(void* out, const void* x, size_t size, DType dtype, int device_id) {
    if (!out || !x || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return square_kernel_host_f32((float*)out, (const float*)x, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return square_kernel_host_f64((double*)out, (const double*)x, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return square_kernel_device(out, x, size, dtype);
}

int rp_ceil(void* out, const void* x, size_t size, DType dtype, int device_id) {
    if (!out || !x || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return ceil_kernel_host_f32((float*)out, (const float*)x, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return ceil_kernel_host_f64((double*)out, (const double*)x, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return ceil_kernel_device(out, x, size, dtype);
}

int rp_floor(void* out, const void* x, size_t size, DType dtype, int device_id) {
    if (!out || !x || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return floor_kernel_host_f32((float*)out, (const float*)x, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return floor_kernel_host_f64((double*)out, (const double*)x, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return floor_kernel_device(out, x, size, dtype);
}

int rp_round(void* out, const void* x, size_t size, DType dtype, int device_id) {
    if (!out || !x || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return round_kernel_host_f32((float*)out, (const float*)x, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return round_kernel_host_f64((double*)out, (const double*)x, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return round_kernel_device(out, x, size, dtype);
}

int rp_trunc(void* out, const void* x, size_t size, DType dtype, int device_id) {
    if (!out || !x || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return trunc_kernel_host_f32((float*)out, (const float*)x, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return trunc_kernel_host_f64((double*)out, (const double*)x, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return trunc_kernel_device(out, x, size, dtype);
}

int rp_rsqrt(void* out, const void* x, size_t size, DType dtype, int device_id) {
    if (!out || !x || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return rsqrt_kernel_host_f32((float*)out, (const float*)x, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return rsqrt_kernel_host_f64((double*)out, (const double*)x, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return rsqrt_kernel_device(out, x, size, dtype);
}

int rp_gelu(void* out, const void* x, size_t size, DType dtype, int device_id) {
    if (!out || !x || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return gelu_kernel_host_f32((float*)out, (const float*)x, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return gelu_kernel_host_f64((double*)out, (const double*)x, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return gelu_kernel_device(out, x, size, dtype);
}

int rp_silu(void* out, const void* x, size_t size, DType dtype, int device_id) {
    if (!out || !x || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return silu_kernel_host_f32((float*)out, (const float*)x, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return silu_kernel_host_f64((double*)out, (const double*)x, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return silu_kernel_device(out, x, size, dtype);
}

int rp_sum_all(void* out, const void* x, size_t size, DType dtype, int device_id) {
    if (!out || !x || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            float result = sum_all_kernel_host_f32((const float*)x, size);
            *((float*)out) = result;
            return 0;
        } else if (dtype == DTYPE_FLOAT64) {
            double result = sum_all_kernel_host_f64((const double*)x, size);
            *((double*)out) = result;
            return 0;
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return sum_all_kernel_device(out, x, size, dtype);
}

int rp_mean_all(void* out, const void* x, size_t size, DType dtype, int device_id) {
    if (!out || !x || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            float result = mean_all_kernel_host_f32((const float*)x, size);
            *((float*)out) = result;
            return 0;
        } else if (dtype == DTYPE_FLOAT64) {
            double result = mean_all_kernel_host_f64((const double*)x, size);
            *((double*)out) = result;
            return 0;
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return mean_all_kernel_device(out, x, size, dtype);
}

int rp_matmul(void* C, const void* A, const void* B, int m, int k, int n, DType dtype, int device_id) {
    if (!C || !A || !B || m <= 0 || k <= 0 || n <= 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return matmul_kernel_host_f32((float*)C, (const float*)A, (const float*)B, m, k, n);
        } else if (dtype == DTYPE_FLOAT64) {
            return matmul_kernel_host_f64((double*)C, (const double*)A, (const double*)B, m, k, n);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return matmul_kernel_device(C, A, B, m, k, n, dtype);
}

Tensor* rp_cat(Tensor** tensors, int num_tensors, int dim) {
    if (!tensors || num_tensors <= 0) return NULL;

    if (num_tensors == 1) {
        return tensor_copy(tensors[0]);
    }

    Tensor* first = tensors[0];
    if (!first) return NULL;

    if (dim < 0 || dim >= first->ndim) return NULL;

    int device_id = first->device_id;
    DType dtype = first->dtype;
    int ndim = first->ndim;

    for (int i = 1; i < num_tensors; i++) {
        if (!tensors[i]) return NULL;
        if (tensors[i]->device_id != device_id) return NULL;
        if (tensors[i]->dtype != dtype) return NULL;
        if (tensors[i]->ndim != ndim) return NULL;

        for (int d = 0; d < ndim; d++) {
            if (d != dim && tensors[i]->shape[d] != first->shape[d]) {
                return NULL;
            }
        }
    }

    if (device_id == -1) {
        return cat_kernel_host(tensors, num_tensors, dim);
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return NULL;
    return cat_kernel_device(tensors, num_tensors, dim);
}

Tensor* rp_slice(Tensor* src, int* start, int* stop, int* step) {
    if (!src) return NULL;

    int ndim = src->ndim;
    int* new_shape = (int*)malloc(ndim * sizeof(int));
    int* new_strides = (int*)malloc(ndim * sizeof(int));

    if (!new_shape || !new_strides) {
        if (new_shape) free(new_shape);
        if (new_strides) free(new_strides);
        return NULL;
    }

    size_t offset = 0;
    size_t elem_size = (src->dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);

    for (int i = 0; i < ndim; i++) {
        int dim_size = src->shape[i];

        int s = (start && start[i] != INT_MIN) ? start[i] : 0;
        int e = (stop && stop[i] != INT_MIN) ? stop[i] : dim_size;
        int st = (step && step[i] != 0) ? step[i] : 1;

        if (s < 0) s += dim_size;
        if (e < 0) e += dim_size;

        if (s < 0) s = 0;
        if (s > dim_size) s = dim_size;
        if (e < 0) e = 0;
        if (e > dim_size) e = dim_size;

        if (s > e) {
            int tmp = s;
            s = e;
            e = tmp;
        }

        new_shape[i] = (e - s + st - 1) / st;
        new_strides[i] = src->strides[i] * st;

        offset += s * src->strides[i];
    }

    size_t new_size = 1;
    for (int i = 0; i < ndim; i++) {
        new_size *= new_shape[i];
    }

    Tensor* view = (Tensor*)malloc(sizeof(Tensor));
    if (!view) {
        free(new_shape);
        free(new_strides);
        return NULL;
    }

    view->dtype = src->dtype;
    view->data = (char*)src->data + offset * elem_size;
    view->ndim = ndim;
    view->size = new_size;
    view->shape = new_shape;
    view->strides = new_strides;
    view->device_id = src->device_id;
    view->owns_data = false;
    view->base_tensor = src->base_tensor ? src->base_tensor : src;
    view->data_offset = offset * elem_size;
    view->metadata = NULL;

    return view;
}

bool rp_is_contiguous(Tensor* tensor) {
    if (!tensor) return false;

    int expected_stride = 1;
    for (int i = tensor->ndim - 1; i >= 0; i--) {
        if (tensor->strides[i] != expected_stride) {
            return false;
        }
        expected_stride *= tensor->shape[i];
    }
    return true;
}

Tensor* rp_contiguous(Tensor* src) {
    if (!src) return NULL;

    if (rp_is_contiguous(src)) {
        int* new_shape = (int*)malloc(src->ndim * sizeof(int));
        int* new_strides = (int*)malloc(src->ndim * sizeof(int));
        if (!new_shape || !new_strides) {
            if (new_shape) free(new_shape);
            if (new_strides) free(new_strides);
            return NULL;
        }

        for (int i = 0; i < src->ndim; i++) {
            new_shape[i] = src->shape[i];
            new_strides[i] = src->strides[i];
        }

        Tensor* view = (Tensor*)malloc(sizeof(Tensor));
        if (!view) {
            free(new_shape);
            free(new_strides);
            return NULL;
        }

        view->dtype = src->dtype;
        view->data = src->data;
        view->ndim = src->ndim;
        view->size = src->size;
        view->shape = new_shape;
        view->strides = new_strides;
        view->device_id = src->device_id;
        view->owns_data = false;
        view->base_tensor = src->base_tensor ? src->base_tensor : src;
        view->data_offset = src->data_offset;
        view->metadata = NULL;

        return view;
    }

    Tensor* contiguous = (src->device_id == -1)
        ? zeros_host_tensor(src->dtype, src->ndim, src->shape, NULL)
        : zeros_device_tensor(src->dtype, src->device_id, src->ndim, src->shape, NULL);

    if (!contiguous) return NULL;

    if (src->device_id == -1) {
        if (src->dtype == DTYPE_FLOAT32) {
            if (contiguous_copy_kernel_host_f32((float*)contiguous->data, (const float*)src->data,
                                               src->ndim, src->shape, src->strides) != 0) {
                free_tensor(contiguous);
                return NULL;
            }
        } else if (src->dtype == DTYPE_FLOAT64) {
            if (contiguous_copy_kernel_host_f64((double*)contiguous->data, (const double*)src->data,
                                               src->ndim, src->shape, src->strides) != 0) {
                free_tensor(contiguous);
                return NULL;
            }
        }
    } else {
        if (!check_cuda_call(cudaSetDevice(src->device_id), "cudaSetDevice")) {
            free_tensor(contiguous);
            return NULL;
        }
        if (contiguous_copy_kernel_device(contiguous->data, src->data,
                                         src->ndim, src->shape, src->strides,
                                         src->size, src->dtype) != 0) {
            free_tensor(contiguous);
            return NULL;
        }
    }

    return contiguous;
}

Tensor* rp_view(Tensor* src, int ndim, int* new_shape) {
    if (!src || !new_shape) return NULL;

    int infer_dim = -1;
    size_t explicit_size = 1;

    for (int i = 0; i < ndim; i++) {
        if (new_shape[i] == -1) {
            if (infer_dim != -1) {
                fprintf(stderr, "Error: Can only infer one dimension in reshape\n");
                return NULL;
            }
            infer_dim = i;
        } else if (new_shape[i] <= 0) {
            fprintf(stderr, "Error: Invalid shape dimension %d at index %d\n", new_shape[i], i);
            return NULL;
        } else {
            explicit_size *= new_shape[i];
        }
    }

    int* final_shape = (int*)malloc(ndim * sizeof(int));
    if (!final_shape) return NULL;

    for (int i = 0; i < ndim; i++) {
        if (i == infer_dim) {
            if (src->size % explicit_size != 0) {
                fprintf(stderr, "Error: Cannot infer dimension, size mismatch\n");
                free(final_shape);
                return NULL;
            }
            final_shape[i] = src->size / explicit_size;
        } else {
            final_shape[i] = new_shape[i];
        }
    }

    size_t final_size = 1;
    for (int i = 0; i < ndim; i++) {
        final_size *= final_shape[i];
    }

    if (final_size != src->size) {
        fprintf(stderr, "Error: Reshape size mismatch. Source: %zu, Target: %zu\n",
                src->size, final_size);
        free(final_shape);
        return NULL;
    }

    if (!rp_is_contiguous(src)) {
        fprintf(stderr, "Error: view requires contiguous tensor\n");
        free(final_shape);
        return NULL;
    }

    int* new_strides = (int*)malloc(ndim * sizeof(int));
    if (!new_strides) {
        free(final_shape);
        return NULL;
    }

    new_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        new_strides[i] = new_strides[i + 1] * final_shape[i + 1];
    }

    Tensor* view = (Tensor*)malloc(sizeof(Tensor));
    if (!view) {
        free(final_shape);
        free(new_strides);
        return NULL;
    }

    view->dtype = src->dtype;
    view->data = src->data;
    view->ndim = ndim;
    view->size = final_size;
    view->shape = final_shape;
    view->strides = new_strides;
    view->device_id = src->device_id;
    view->owns_data = false;
    view->base_tensor = src->base_tensor ? src->base_tensor : src;
    view->data_offset = src->data_offset;
    view->metadata = NULL;

    return view;
}

Tensor* rp_reshape(Tensor* src, int ndim, int* new_shape) {
    if (!src || !new_shape) return NULL;

    Tensor* result = rp_view(src, ndim, new_shape);
    if (result) {
        return result;
    }

    Tensor* contiguous = rp_contiguous(src);
    if (!contiguous) return NULL;

    result = rp_view(contiguous, ndim, new_shape);

    if (contiguous->owns_data) {
        if (result) {
            result->owns_data = true;
            result->base_tensor = NULL;
            free(contiguous->shape);
            free(contiguous->strides);
            free(contiguous);
        } else {
            free_tensor(contiguous);
        }
    } else {
        free_tensor(contiguous);
    }

    return result;
}

Tensor* rp_transpose(Tensor* src, int dim0, int dim1) {
    if (!src) return NULL;

    if (src->ndim < 2) {
        fprintf(stderr, "Error: transpose requires at least 2 dimensions\n");
        return NULL;
    }

    if (dim0 < 0) dim0 += src->ndim;
    if (dim1 < 0) dim1 += src->ndim;

    if (dim0 < 0 || dim0 >= src->ndim || dim1 < 0 || dim1 >= src->ndim) {
        fprintf(stderr, "Error: dimension out of bounds for transpose\n");
        return NULL;
    }

    if (dim0 == dim1) {
        int* new_shape = (int*)malloc(src->ndim * sizeof(int));
        int* new_strides = (int*)malloc(src->ndim * sizeof(int));
        if (!new_shape || !new_strides) {
            if (new_shape) free(new_shape);
            if (new_strides) free(new_strides);
            return NULL;
        }

        for (int i = 0; i < src->ndim; i++) {
            new_shape[i] = src->shape[i];
            new_strides[i] = src->strides[i];
        }

        Tensor* view = (Tensor*)malloc(sizeof(Tensor));
        if (!view) {
            free(new_shape);
            free(new_strides);
            return NULL;
        }

        view->dtype = src->dtype;
        view->data = src->data;
        view->ndim = src->ndim;
        view->size = src->size;
        view->shape = new_shape;
        view->strides = new_strides;
        view->device_id = src->device_id;
        view->owns_data = false;
        view->base_tensor = src->base_tensor ? src->base_tensor : src;
        view->data_offset = src->data_offset;
        view->metadata = NULL;

        return view;
    }

    int* new_shape = (int*)malloc(src->ndim * sizeof(int));
    int* new_strides = (int*)malloc(src->ndim * sizeof(int));
    if (!new_shape || !new_strides) {
        if (new_shape) free(new_shape);
        if (new_strides) free(new_strides);
        return NULL;
    }

    for (int i = 0; i < src->ndim; i++) {
        new_shape[i] = src->shape[i];
        new_strides[i] = src->strides[i];
    }

    int tmp_shape = new_shape[dim0];
    new_shape[dim0] = new_shape[dim1];
    new_shape[dim1] = tmp_shape;

    int tmp_stride = new_strides[dim0];
    new_strides[dim0] = new_strides[dim1];
    new_strides[dim1] = tmp_stride;

    Tensor* view = (Tensor*)malloc(sizeof(Tensor));
    if (!view) {
        free(new_shape);
        free(new_strides);
        return NULL;
    }

    view->dtype = src->dtype;
    view->data = src->data;
    view->ndim = src->ndim;
    view->size = src->size;
    view->shape = new_shape;
    view->strides = new_strides;
    view->device_id = src->device_id;
    view->owns_data = false;
    view->base_tensor = src->base_tensor ? src->base_tensor : src;
    view->data_offset = src->data_offset;
    view->metadata = NULL;

    return view;
}

Tensor* rp_squeeze(Tensor* src, int dim) {
    if (!src) return NULL;

    if (dim < -1 || dim >= src->ndim) {
        fprintf(stderr, "Error: dimension %d out of bounds for tensor with %d dimensions\n", dim, src->ndim);
        return NULL;
    }

    if (dim >= 0 && dim < src->ndim) {
        if (dim < 0) dim += src->ndim;

        if (src->shape[dim] != 1) {
            fprintf(stderr, "Error: cannot squeeze dimension %d with size %d (only size 1 can be squeezed)\n", dim, src->shape[dim]);
            return NULL;
        }

        int new_ndim = src->ndim - 1;
        if (new_ndim == 0) {
            fprintf(stderr, "Error: cannot squeeze to 0 dimensions\n");
            return NULL;
        }

        int* new_shape = (int*)malloc(new_ndim * sizeof(int));
        int* new_strides = (int*)malloc(new_ndim * sizeof(int));
        if (!new_shape || !new_strides) {
            if (new_shape) free(new_shape);
            if (new_strides) free(new_strides);
            return NULL;
        }

        int j = 0;
        for (int i = 0; i < src->ndim; i++) {
            if (i != dim) {
                new_shape[j] = src->shape[i];
                new_strides[j] = src->strides[i];
                j++;
            }
        }

        Tensor* view = (Tensor*)malloc(sizeof(Tensor));
        if (!view) {
            free(new_shape);
            free(new_strides);
            return NULL;
        }

        view->dtype = src->dtype;
        view->data = src->data;
        view->ndim = new_ndim;
        view->size = src->size;
        view->shape = new_shape;
        view->strides = new_strides;
        view->device_id = src->device_id;
        view->owns_data = false;
        view->base_tensor = src->base_tensor ? src->base_tensor : src;
        view->data_offset = src->data_offset;
        view->metadata = NULL;

        return view;
    }

    int count_ones = 0;
    for (int i = 0; i < src->ndim; i++) {
        if (src->shape[i] == 1) {
            count_ones++;
        }
    }

    if (count_ones == 0) {
        int* new_shape = (int*)malloc(src->ndim * sizeof(int));
        int* new_strides = (int*)malloc(src->ndim * sizeof(int));
        if (!new_shape || !new_strides) {
            if (new_shape) free(new_shape);
            if (new_strides) free(new_strides);
            return NULL;
        }

        for (int i = 0; i < src->ndim; i++) {
            new_shape[i] = src->shape[i];
            new_strides[i] = src->strides[i];
        }

        Tensor* view = (Tensor*)malloc(sizeof(Tensor));
        if (!view) {
            free(new_shape);
            free(new_strides);
            return NULL;
        }

        view->dtype = src->dtype;
        view->data = src->data;
        view->ndim = src->ndim;
        view->size = src->size;
        view->shape = new_shape;
        view->strides = new_strides;
        view->device_id = src->device_id;
        view->owns_data = false;
        view->base_tensor = src->base_tensor ? src->base_tensor : src;
        view->data_offset = src->data_offset;
        view->metadata = NULL;

        return view;
    }

    int new_ndim = src->ndim - count_ones;
    if (new_ndim == 0) {
        fprintf(stderr, "Error: cannot squeeze to 0 dimensions\n");
        return NULL;
    }

    int* new_shape = (int*)malloc(new_ndim * sizeof(int));
    int* new_strides = (int*)malloc(new_ndim * sizeof(int));
    if (!new_shape || !new_strides) {
        if (new_shape) free(new_shape);
        if (new_strides) free(new_strides);
        return NULL;
    }

    int j = 0;
    for (int i = 0; i < src->ndim; i++) {
        if (src->shape[i] != 1) {
            new_shape[j] = src->shape[i];
            new_strides[j] = src->strides[i];
            j++;
        }
    }

    Tensor* view = (Tensor*)malloc(sizeof(Tensor));
    if (!view) {
        free(new_shape);
        free(new_strides);
        return NULL;
    }

    view->dtype = src->dtype;
    view->data = src->data;
    view->ndim = new_ndim;
    view->size = src->size;
    view->shape = new_shape;
    view->strides = new_strides;
    view->device_id = src->device_id;
    view->owns_data = false;
    view->base_tensor = src->base_tensor ? src->base_tensor : src;
    view->data_offset = src->data_offset;
    view->metadata = NULL;

    return view;
}

Tensor* rp_unsqueeze(Tensor* src, int dim) {
    if (!src) return NULL;

    int new_ndim = src->ndim + 1;

    if (dim < 0) dim += new_ndim;

    if (dim < 0 || dim >= new_ndim) {
        fprintf(stderr, "Error: dimension %d out of bounds for unsqueeze (valid range: [-%d, %d))\n",
                dim, new_ndim, new_ndim);
        return NULL;
    }

    int* new_shape = (int*)malloc(new_ndim * sizeof(int));
    int* new_strides = (int*)malloc(new_ndim * sizeof(int));
    if (!new_shape || !new_strides) {
        if (new_shape) free(new_shape);
        if (new_strides) free(new_strides);
        return NULL;
    }

    int stride_at_dim;
    if (dim == new_ndim - 1) {
        stride_at_dim = 1;
    } else if (dim == 0) {
        stride_at_dim = src->strides[0] * src->shape[0];
    } else {
        stride_at_dim = src->strides[dim];
    }

    int src_idx = 0;
    for (int i = 0; i < new_ndim; i++) {
        if (i == dim) {
            new_shape[i] = 1;
            new_strides[i] = stride_at_dim;
        } else {
            new_shape[i] = src->shape[src_idx];
            new_strides[i] = src->strides[src_idx];
            src_idx++;
        }
    }

    Tensor* view = (Tensor*)malloc(sizeof(Tensor));
    if (!view) {
        free(new_shape);
        free(new_strides);
        return NULL;
    }

    view->dtype = src->dtype;
    view->data = src->data;
    view->ndim = new_ndim;
    view->size = src->size;
    view->shape = new_shape;
    view->strides = new_strides;
    view->device_id = src->device_id;
    view->owns_data = false;
    view->base_tensor = src->base_tensor ? src->base_tensor : src;
    view->data_offset = src->data_offset;
    view->metadata = NULL;

    return view;
}

Tensor** rp_split(Tensor* src, int* sizes, int num_splits, int dim) {
    if (!src || !sizes || num_splits <= 0) return NULL;

    if (dim < 0) dim += src->ndim;

    if (dim < 0 || dim >= src->ndim) {
        fprintf(stderr, "Error: dimension %d out of bounds for split (valid range: [-%d, %d))\n",
                dim, src->ndim, src->ndim);
        return NULL;
    }

    int total_size = 0;
    for (int i = 0; i < num_splits; i++) {
        if (sizes[i] <= 0) {
            fprintf(stderr, "Error: split size must be positive, got %d at index %d\n", sizes[i], i);
            return NULL;
        }
        total_size += sizes[i];
    }

    if (total_size != src->shape[dim]) {
        fprintf(stderr, "Error: sum of split sizes (%d) must equal dimension size (%d)\n",
                total_size, src->shape[dim]);
        return NULL;
    }

    Tensor** result = (Tensor**)malloc(num_splits * sizeof(Tensor*));
    if (!result) return NULL;

    int current_start = 0;
    for (int i = 0; i < num_splits; i++) {
        int* start = (int*)malloc(src->ndim * sizeof(int));
        int* stop = (int*)malloc(src->ndim * sizeof(int));
        int* step = (int*)malloc(src->ndim * sizeof(int));

        if (!start || !stop || !step) {
            if (start) free(start);
            if (stop) free(stop);
            if (step) free(step);
            for (int j = 0; j < i; j++) {
                free_tensor(result[j]);
            }
            free(result);
            return NULL;
        }

        for (int d = 0; d < src->ndim; d++) {
            if (d == dim) {
                start[d] = current_start;
                stop[d] = current_start + sizes[i];
                step[d] = 1;
            } else {
                start[d] = 0;
                stop[d] = src->shape[d];
                step[d] = 1;
            }
        }

        result[i] = rp_slice(src, start, stop, step);

        free(start);
        free(stop);
        free(step);

        if (!result[i]) {
            for (int j = 0; j < i; j++) {
                free_tensor(result[j]);
            }
            free(result);
            return NULL;
        }

        current_start += sizes[i];
    }

    return result;
}

Tensor** rp_split_equal(Tensor* src, int num_splits, int dim) {
    if (!src || num_splits <= 0) return NULL;

    if (dim < 0) dim += src->ndim;

    if (dim < 0 || dim >= src->ndim) {
        fprintf(stderr, "Error: dimension %d out of bounds for split_equal (valid range: [-%d, %d))\n",
                dim, src->ndim, src->ndim);
        return NULL;
    }

    int dim_size = src->shape[dim];
    if (dim_size % num_splits != 0) {
        fprintf(stderr, "Error: dimension size %d is not divisible by num_splits %d\n",
                dim_size, num_splits);
        return NULL;
    }

    int split_size = dim_size / num_splits;

    int* sizes = (int*)malloc(num_splits * sizeof(int));
    if (!sizes) return NULL;

    for (int i = 0; i < num_splits; i++) {
        sizes[i] = split_size;
    }

    Tensor** result = rp_split(src, sizes, num_splits, dim);
    free(sizes);

    return result;
}

Tensor* rp_flatten(Tensor* src, int start_dim, int end_dim) {
    if (!src) return NULL;

    if (start_dim < 0) start_dim += src->ndim;
    if (end_dim < 0) end_dim += src->ndim;

    if (start_dim < 0 || start_dim >= src->ndim) {
        fprintf(stderr, "Error: start_dim %d out of bounds for tensor with %d dimensions\n", start_dim, src->ndim);
        return NULL;
    }

    if (end_dim < 0 || end_dim >= src->ndim) {
        fprintf(stderr, "Error: end_dim %d out of bounds for tensor with %d dimensions\n", end_dim, src->ndim);
        return NULL;
    }

    if (start_dim > end_dim) {
        fprintf(stderr, "Error: start_dim %d must be <= end_dim %d\n", start_dim, end_dim);
        return NULL;
    }

    int num_dims_to_flatten = end_dim - start_dim + 1;
    int new_ndim = src->ndim - num_dims_to_flatten + 1;

    int* new_shape = (int*)malloc(new_ndim * sizeof(int));
    if (!new_shape) return NULL;

    int flattened_size = 1;
    for (int i = start_dim; i <= end_dim; i++) {
        flattened_size *= src->shape[i];
    }

    int idx = 0;
    for (int i = 0; i < start_dim; i++) {
        new_shape[idx++] = src->shape[i];
    }
    new_shape[idx++] = flattened_size;
    for (int i = end_dim + 1; i < src->ndim; i++) {
        new_shape[idx++] = src->shape[i];
    }

    Tensor* result = rp_reshape(src, new_ndim, new_shape);
    free(new_shape);

    return result;
}

Tensor* rp_permute(Tensor* src, int* dims) {
    if (!src || !dims) return NULL;

    int ndim = src->ndim;

    bool* seen = (bool*)calloc(ndim, sizeof(bool));
    if (!seen) return NULL;

    for (int i = 0; i < ndim; i++) {
        int d = dims[i];
        if (d < 0) d += ndim;

        if (d < 0 || d >= ndim) {
            fprintf(stderr, "Error: dimension %d out of bounds for tensor with %d dimensions\n", dims[i], ndim);
            free(seen);
            return NULL;
        }

        if (seen[d]) {
            fprintf(stderr, "Error: dimension %d appears more than once in permutation\n", d);
            free(seen);
            return NULL;
        }
        seen[d] = true;
    }
    free(seen);

    int* new_shape = (int*)malloc(ndim * sizeof(int));
    int* new_strides = (int*)malloc(ndim * sizeof(int));
    if (!new_shape || !new_strides) {
        if (new_shape) free(new_shape);
        if (new_strides) free(new_strides);
        return NULL;
    }

    for (int i = 0; i < ndim; i++) {
        int d = dims[i];
        if (d < 0) d += ndim;
        new_shape[i] = src->shape[d];
        new_strides[i] = src->strides[d];
    }

    Tensor* view = (Tensor*)malloc(sizeof(Tensor));
    if (!view) {
        free(new_shape);
        free(new_strides);
        return NULL;
    }

    view->dtype = src->dtype;
    view->data = src->data;
    view->ndim = ndim;
    view->size = src->size;
    view->shape = new_shape;
    view->strides = new_strides;
    view->device_id = src->device_id;
    view->owns_data = false;
    view->base_tensor = src->base_tensor ? src->base_tensor : src;
    view->data_offset = src->data_offset;
    view->metadata = NULL;

    return view;
}

Tensor** rp_chunk(Tensor* src, int chunks, int dim) {
    if (!src || chunks <= 0) return NULL;

    if (dim < 0) dim += src->ndim;

    if (dim < 0 || dim >= src->ndim) {
        fprintf(stderr, "Error: dimension %d out of bounds for chunk (valid range: [-%d, %d))\n",
                dim, src->ndim, src->ndim);
        return NULL;
    }

    int dim_size = src->shape[dim];

    if (chunks > dim_size) {
        chunks = dim_size;
    }

    int chunk_size = (dim_size + chunks - 1) / chunks;

    int* sizes = (int*)malloc(chunks * sizeof(int));
    if (!sizes) return NULL;

    int remaining = dim_size;
    int actual_chunks = 0;
    for (int i = 0; i < chunks; i++) {
        if (remaining <= 0) break;
        sizes[i] = (remaining < chunk_size) ? remaining : chunk_size;
        remaining -= sizes[i];
        actual_chunks++;
    }

    Tensor** split_result = rp_split(src, sizes, actual_chunks, dim);
    free(sizes);

    if (!split_result) return NULL;

    Tensor** result = (Tensor**)malloc((actual_chunks + 1) * sizeof(Tensor*));
    if (!result) {
        for (int i = 0; i < actual_chunks; i++) {
            free_tensor(split_result[i]);
        }
        free(split_result);
        return NULL;
    }

    for (int i = 0; i < actual_chunks; i++) {
        result[i] = split_result[i];
    }
    result[actual_chunks] = NULL;

    free(split_result);
    return result;
}

Tensor* rp_expand(Tensor* src, int ndim, int* shape) {
    if (!src || !shape || ndim < src->ndim) {
        if (ndim < src->ndim) {
            fprintf(stderr, "Error: target ndim (%d) must be >= source ndim (%d)\n", ndim, src->ndim);
        }
        return NULL;
    }

    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) {
            fprintf(stderr, "Error: invalid target shape dimension %d at index %d\n", shape[i], i);
            return NULL;
        }
    }

    int* new_shape = (int*)malloc(ndim * sizeof(int));
    int* new_strides = (int*)malloc(ndim * sizeof(int));
    if (!new_shape || !new_strides) {
        if (new_shape) free(new_shape);
        if (new_strides) free(new_strides);
        return NULL;
    }

    int src_offset = ndim - src->ndim;

    for (int i = 0; i < ndim; i++) {
        new_shape[i] = shape[i];

        if (i < src_offset) {
            new_strides[i] = 0;
        } else {
            int src_idx = i - src_offset;
            if (src->shape[src_idx] == 1) {
                new_strides[i] = 0;
            } else if (src->shape[src_idx] == shape[i]) {
                new_strides[i] = src->strides[src_idx];
            } else {
                fprintf(stderr, "Error: cannot expand dimension %d from size %d to %d (must be 1 or same size)\n",
                        src_idx, src->shape[src_idx], shape[i]);
                free(new_shape);
                free(new_strides);
                return NULL;
            }
        }
    }

    size_t new_size = 1;
    for (int i = 0; i < ndim; i++) {
        new_size *= new_shape[i];
    }

    Tensor* view = (Tensor*)malloc(sizeof(Tensor));
    if (!view) {
        free(new_shape);
        free(new_strides);
        return NULL;
    }

    view->dtype = src->dtype;
    view->data = src->data;
    view->ndim = ndim;
    view->size = new_size;
    view->shape = new_shape;
    view->strides = new_strides;
    view->device_id = src->device_id;
    view->owns_data = false;
    view->base_tensor = src->base_tensor ? src->base_tensor : src;
    view->data_offset = src->data_offset;
    view->metadata = NULL;

    return view;
}

Tensor* rp_repeat(Tensor* src, int* repeats) {
    if (!src || !repeats) {
        return NULL;
    }

    for (int i = 0; i < src->ndim; i++) {
        if (repeats[i] <= 0) {
            fprintf(stderr, "Error: repeat count must be positive at dimension %d (got %d)\n", i, repeats[i]);
            return NULL;
        }
    }

    int* new_shape = (int*)malloc(src->ndim * sizeof(int));
    if (!new_shape) {
        return NULL;
    }

    size_t new_size = 1;
    for (int i = 0; i < src->ndim; i++) {
        new_shape[i] = src->shape[i] * repeats[i];
        new_size *= new_shape[i];
    }

    Tensor* out = (src->device_id == -1)
        ? zeros_host_tensor(src->dtype, src->ndim, new_shape, NULL)
        : zeros_device_tensor(src->dtype, src->device_id, src->ndim, new_shape, NULL);

    free(new_shape);

    if (!out) {
        return NULL;
    }

    int result;
    if (src->device_id == -1) {
        if (src->dtype == DTYPE_FLOAT32) {
            result = repeat_kernel_host_f32((float*)out->data, (const float*)src->data,
                                           src->ndim, src->shape, repeats);
        } else if (src->dtype == DTYPE_FLOAT64) {
            result = repeat_kernel_host_f64((double*)out->data, (const double*)src->data,
                                            src->ndim, src->shape, repeats);
        } else {
            free_tensor(out);
            return NULL;
        }
    } else {
        cudaSetDevice(src->device_id);
        result = repeat_kernel_device(out->data, src->data, src->ndim, src->shape, repeats,
                                      new_size, src->dtype);
    }

    if (result != 0) {
        free_tensor(out);
        return NULL;
    }

    return out;
}
