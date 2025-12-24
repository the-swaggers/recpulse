#include "functional.h"
#include "../core/cuda_helpers.h"
#include <cuda_runtime.h>

int rp_add(void* out, const void* a, const void* b, size_t size, DType dtype, int device_id) {
    if (!out || !a || !b || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return add_kernel_host_f32((float*)out, (const float*)a, (const float*)b, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return add_kernel_host_f64((double*)out, (const double*)a, (const double*)b, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return add_kernel_device(out, a, b, size, dtype);
}

int rp_sub(void* out, const void* a, const void* b, size_t size, DType dtype, int device_id) {
    if (!out || !a || !b || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return sub_kernel_host_f32((float*)out, (const float*)a, (const float*)b, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return sub_kernel_host_f64((double*)out, (const double*)a, (const double*)b, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return sub_kernel_device(out, a, b, size, dtype);
}

int rp_mul(void* out, const void* a, const void* b, size_t size, DType dtype, int device_id) {
    if (!out || !a || !b || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return mul_kernel_host_f32((float*)out, (const float*)a, (const float*)b, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return mul_kernel_host_f64((double*)out, (const double*)a, (const double*)b, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return mul_kernel_device(out, a, b, size, dtype);
}

int rp_divide(void* out, const void* a, const void* b, size_t size, DType dtype, int device_id) {
    if (!out || !a || !b || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return div_kernel_host_f32((float*)out, (const float*)a, (const float*)b, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return div_kernel_host_f64((double*)out, (const double*)a, (const double*)b, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return div_kernel_device(out, a, b, size, dtype);
}

int rp_power(void* out, const void* a, const void* b, size_t size, DType dtype, int device_id) {
    if (!out || !a || !b || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return pow_kernel_host_f32((float*)out, (const float*)a, (const float*)b, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return pow_kernel_host_f64((double*)out, (const double*)a, (const double*)b, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return pow_kernel_device(out, a, b, size, dtype);
}

int rp_logb(void* out, const void* a, const void* b, size_t size, DType dtype, int device_id) {
    if (!out || !a || !b || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return logb_kernel_host_f32((float*)out, (const float*)a, (const float*)b, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return logb_kernel_host_f64((double*)out, (const double*)a, (const double*)b, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return logb_kernel_device(out, a, b, size, dtype);
}

int rp_add_scalar(void* out, const void* a, const void* scalar, size_t size, DType dtype, int device_id) {
    if (!out || !a || !scalar || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return add_scalar_kernel_host_f32((float*)out, (const float*)a, *(const float*)scalar, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return add_scalar_kernel_host_f64((double*)out, (const double*)a, *(const double*)scalar, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return add_scalar_kernel_device(out, a, scalar, size, dtype);
}

int rp_sub_scalar(void* out, const void* a, const void* scalar, size_t size, DType dtype, int device_id) {
    if (!out || !a || !scalar || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return sub_scalar_kernel_host_f32((float*)out, (const float*)a, *(const float*)scalar, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return sub_scalar_kernel_host_f64((double*)out, (const double*)a, *(const double*)scalar, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return sub_scalar_kernel_device(out, a, scalar, size, dtype);
}

int rp_mul_scalar(void* out, const void* a, const void* scalar, size_t size, DType dtype, int device_id) {
    if (!out || !a || !scalar || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return mul_scalar_kernel_host_f32((float*)out, (const float*)a, *(const float*)scalar, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return mul_scalar_kernel_host_f64((double*)out, (const double*)a, *(const double*)scalar, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return mul_scalar_kernel_device(out, a, scalar, size, dtype);
}

int rp_div_scalar(void* out, const void* a, const void* scalar, size_t size, DType dtype, int device_id) {
    if (!out || !a || !scalar || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return div_scalar_kernel_host_f32((float*)out, (const float*)a, *(const float*)scalar, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return div_scalar_kernel_host_f64((double*)out, (const double*)a, *(const double*)scalar, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return div_scalar_kernel_device(out, a, scalar, size, dtype);
}

int rp_pow_scalar(void* out, const void* a, const void* scalar, size_t size, DType dtype, int device_id) {
    if (!out || !a || !scalar || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return pow_scalar_kernel_host_f32((float*)out, (const float*)a, *(const float*)scalar, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return pow_scalar_kernel_host_f64((double*)out, (const double*)a, *(const double*)scalar, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return pow_scalar_kernel_device(out, a, scalar, size, dtype);
}

int rp_rsub_scalar(void* out, const void* scalar, const void* a, size_t size, DType dtype, int device_id) {
    if (!out || !a || !scalar || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return rsub_scalar_kernel_host_f32((float*)out, *(const float*)scalar, (const float*)a, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return rsub_scalar_kernel_host_f64((double*)out, *(const double*)scalar, (const double*)a, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return rsub_scalar_kernel_device(out, scalar, a, size, dtype);
}

int rp_rdiv_scalar(void* out, const void* scalar, const void* a, size_t size, DType dtype, int device_id) {
    if (!out || !a || !scalar || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return rdiv_scalar_kernel_host_f32((float*)out, *(const float*)scalar, (const float*)a, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return rdiv_scalar_kernel_host_f64((double*)out, *(const double*)scalar, (const double*)a, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return rdiv_scalar_kernel_device(out, scalar, a, size, dtype);
}

int rp_rpow_scalar(void* out, const void* scalar, const void* a, size_t size, DType dtype, int device_id) {
    if (!out || !a || !scalar || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return rpow_scalar_kernel_host_f32((float*)out, *(const float*)scalar, (const float*)a, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return rpow_scalar_kernel_host_f64((double*)out, *(const double*)scalar, (const double*)a, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return rpow_scalar_kernel_device(out, scalar, a, size, dtype);
}

int rp_logb_scalar(void* out, const void* a, const void* scalar, size_t size, DType dtype, int device_id) {
    if (!out || !a || !scalar || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return logb_scalar_kernel_host_f32((float*)out, (const float*)a, *(const float*)scalar, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return logb_scalar_kernel_host_f64((double*)out, (const double*)a, *(const double*)scalar, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return logb_scalar_kernel_device(out, a, scalar, size, dtype);
}

int rp_rlogb_scalar(void* out, const void* scalar, const void* a, size_t size, DType dtype, int device_id) {
    if (!out || !a || !scalar || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return rlogb_scalar_kernel_host_f32((float*)out, *(const float*)scalar, (const float*)a, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return rlogb_scalar_kernel_host_f64((double*)out, *(const double*)scalar, (const double*)a, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return rlogb_scalar_kernel_device(out, scalar, a, size, dtype);
}

int rp_exp(void* out, const void* a, size_t size, DType dtype, int device_id) {
    if (!out || !a || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return exp_kernel_host_f32((float*)out, (const float*)a, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return exp_kernel_host_f64((double*)out, (const double*)a, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return exp_kernel_device(out, a, size, dtype);
}

int rp_log(void* out, const void* a, size_t size, DType dtype, int device_id) {
    if (!out || !a || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return log_kernel_host_f32((float*)out, (const float*)a, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return log_kernel_host_f64((double*)out, (const double*)a, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return log_kernel_device(out, a, size, dtype);
}

int rp_sqrt(void* out, const void* a, size_t size, DType dtype, int device_id) {
    if (!out || !a || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return sqrt_kernel_host_f32((float*)out, (const float*)a, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return sqrt_kernel_host_f64((double*)out, (const double*)a, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return sqrt_kernel_device(out, a, size, dtype);
}

int rp_abs(void* out, const void* a, size_t size, DType dtype, int device_id) {
    if (!out || !a || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return abs_kernel_host_f32((float*)out, (const float*)a, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return abs_kernel_host_f64((double*)out, (const double*)a, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return abs_kernel_device(out, a, size, dtype);
}

int rp_sin(void* out, const void* a, size_t size, DType dtype, int device_id) {
    if (!out || !a || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return sin_kernel_host_f32((float*)out, (const float*)a, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return sin_kernel_host_f64((double*)out, (const double*)a, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return sin_kernel_device(out, a, size, dtype);
}

int rp_cos(void* out, const void* a, size_t size, DType dtype, int device_id) {
    if (!out || !a || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return cos_kernel_host_f32((float*)out, (const float*)a, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return cos_kernel_host_f64((double*)out, (const double*)a, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return cos_kernel_device(out, a, size, dtype);
}

int rp_tan(void* out, const void* a, size_t size, DType dtype, int device_id) {
    if (!out || !a || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return tan_kernel_host_f32((float*)out, (const float*)a, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return tan_kernel_host_f64((double*)out, (const double*)a, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return tan_kernel_device(out, a, size, dtype);
}

int rp_asin(void* out, const void* a, size_t size, DType dtype, int device_id) {
    if (!out || !a || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return asin_kernel_host_f32((float*)out, (const float*)a, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return asin_kernel_host_f64((double*)out, (const double*)a, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return asin_kernel_device(out, a, size, dtype);
}

int rp_acos(void* out, const void* a, size_t size, DType dtype, int device_id) {
    if (!out || !a || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return acos_kernel_host_f32((float*)out, (const float*)a, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return acos_kernel_host_f64((double*)out, (const double*)a, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return acos_kernel_device(out, a, size, dtype);
}

int rp_atan(void* out, const void* a, size_t size, DType dtype, int device_id) {
    if (!out || !a || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return atan_kernel_host_f32((float*)out, (const float*)a, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return atan_kernel_host_f64((double*)out, (const double*)a, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return atan_kernel_device(out, a, size, dtype);
}

int rp_sinh(void* out, const void* a, size_t size, DType dtype, int device_id) {
    if (!out || !a || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return sinh_kernel_host_f32((float*)out, (const float*)a, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return sinh_kernel_host_f64((double*)out, (const double*)a, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return sinh_kernel_device(out, a, size, dtype);
}

int rp_cosh(void* out, const void* a, size_t size, DType dtype, int device_id) {
    if (!out || !a || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return cosh_kernel_host_f32((float*)out, (const float*)a, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return cosh_kernel_host_f64((double*)out, (const double*)a, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return cosh_kernel_device(out, a, size, dtype);
}

int rp_tanh(void* out, const void* a, size_t size, DType dtype, int device_id) {
    if (!out || !a || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return tanh_kernel_host_f32((float*)out, (const float*)a, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return tanh_kernel_host_f64((double*)out, (const double*)a, size);
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return tanh_kernel_device(out, a, size, dtype);
}

int rp_sum_all(void* out, const void* a, size_t size, DType dtype, int device_id) {
    if (!out || !a || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            float result = sum_all_kernel_host_f32((const float*)a, size);
            *((float*)out) = result;
            return 0;
        } else if (dtype == DTYPE_FLOAT64) {
            double result = sum_all_kernel_host_f64((const double*)a, size);
            *((double*)out) = result;
            return 0;
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return sum_all_kernel_device(out, a, size, dtype);
}

int rp_mean_all(void* out, const void* a, size_t size, DType dtype, int device_id) {
    if (!out || !a || size == 0) return -1;

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            float result = mean_all_kernel_host_f32((const float*)a, size);
            *((float*)out) = result;
            return 0;
        } else if (dtype == DTYPE_FLOAT64) {
            double result = mean_all_kernel_host_f64((const double*)a, size);
            *((double*)out) = result;
            return 0;
        }
        return -1;
    }

    if (!check_cuda_call(cudaSetDevice(device_id), "cudaSetDevice")) return -1;
    return mean_all_kernel_device(out, a, size, dtype);
}
