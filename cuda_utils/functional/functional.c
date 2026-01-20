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

Tensor* rp_reshape(Tensor* src, int ndim, int* new_shape) {
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
        fprintf(stderr, "Error: reshape requires contiguous tensor\n");
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
