#include "functional.h"
#include "../core/cuda_helpers.h"
#include <cuda_runtime.h>

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
