#include "functional.h"
#include "../core/cuda_helpers.h"
#include <cuda_runtime.h>

int add(void* out, const void* a, const void* b, size_t size, DType dtype, int device_id) {
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

int sub(void* out, const void* a, const void* b, size_t size, DType dtype, int device_id) {
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

int mul(void* out, const void* a, const void* b, size_t size, DType dtype, int device_id) {
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

int divide(void* out, const void* a, const void* b, size_t size, DType dtype, int device_id) {
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

int power(void* out, const void* a, const void* b, size_t size, DType dtype, int device_id) {
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

int add_scalar(void* out, const void* a, const void* scalar, size_t size, DType dtype, int device_id) {
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

int sub_scalar(void* out, const void* a, const void* scalar, size_t size, DType dtype, int device_id) {
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

int mul_scalar(void* out, const void* a, const void* scalar, size_t size, DType dtype, int device_id) {
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

int div_scalar(void* out, const void* a, const void* scalar, size_t size, DType dtype, int device_id) {
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

int pow_scalar(void* out, const void* a, const void* scalar, size_t size, DType dtype, int device_id) {
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

int rsub_scalar(void* out, const void* scalar, const void* a, size_t size, DType dtype, int device_id) {
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

int rdiv_scalar(void* out, const void* scalar, const void* a, size_t size, DType dtype, int device_id) {
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

int rpow_scalar(void* out, const void* scalar, const void* a, size_t size, DType dtype, int device_id) {
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
