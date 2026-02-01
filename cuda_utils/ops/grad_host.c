#include "ops.h"
#include <string.h>
#include <math.h>

int backwards_add_x1_host(const void* grad_c, void* grad_x1, size_t size, DType dtype) {
    if (!grad_c || !grad_x1) return -1;

    size_t element_size = (dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);
    size_t nbytes = size * element_size;

    memcpy(grad_x1, grad_c, nbytes);
    return 0;
}

int backwards_add_x2_host(const void* grad_c, void* grad_x2, size_t size, DType dtype) {
    if (!grad_c || !grad_x2) return -1;

    size_t element_size = (dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);
    size_t nbytes = size * element_size;

    memcpy(grad_x2, grad_c, nbytes);
    return 0;
}

int backwards_mul_x1_host(const void* grad_c, const void* x2, void* grad_x1, size_t size, DType dtype) {
    if (!grad_c || !x2 || !grad_x1) return -1;

    if (dtype == DTYPE_FLOAT32) {
        const float* grad_c_f32 = (const float*)grad_c;
        const float* x2_f32 = (const float*)x2;
        float* grad_x1_f32 = (float*)grad_x1;

        for (size_t i = 0; i < size; i++) {
            grad_x1_f32[i] = grad_c_f32[i] * x2_f32[i];
        }
    } else {
        const double* grad_c_f64 = (const double*)grad_c;
        const double* x2_f64 = (const double*)x2;
        double* grad_x1_f64 = (double*)grad_x1;

        for (size_t i = 0; i < size; i++) {
            grad_x1_f64[i] = grad_c_f64[i] * x2_f64[i];
        }
    }

    return 0;
}

int backwards_mul_x2_host(const void* grad_c, const void* x1, void* grad_x2, size_t size, DType dtype) {
    if (!grad_c || !x1 || !grad_x2) return -1;

    if (dtype == DTYPE_FLOAT32) {
        const float* grad_c_f32 = (const float*)grad_c;
        const float* x1_f32 = (const float*)x1;
        float* grad_x2_f32 = (float*)grad_x2;

        for (size_t i = 0; i < size; i++) {
            grad_x2_f32[i] = grad_c_f32[i] * x1_f32[i];
        }
    } else {
        const double* grad_c_f64 = (const double*)grad_c;
        const double* x1_f64 = (const double*)x1;
        double* grad_x2_f64 = (double*)grad_x2;

        for (size_t i = 0; i < size; i++) {
            grad_x2_f64[i] = grad_c_f64[i] * x1_f64[i];
        }
    }

    return 0;
}

int backwards_sub_x1_host(const void* grad_c, void* grad_x1, size_t size, DType dtype) {
    if (!grad_c || !grad_x1) return -1;

    size_t element_size = (dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);
    size_t nbytes = size * element_size;

    memcpy(grad_x1, grad_c, nbytes);
    return 0;
}

int backwards_sub_x2_host(const void* grad_c, void* grad_x2, size_t size, DType dtype) {
    if (!grad_c || !grad_x2) return -1;

    if (dtype == DTYPE_FLOAT32) {
        const float* grad_c_f32 = (const float*)grad_c;
        float* grad_x2_f32 = (float*)grad_x2;

        for (size_t i = 0; i < size; i++) {
            grad_x2_f32[i] = -grad_c_f32[i];
        }
    } else {
        const double* grad_c_f64 = (const double*)grad_c;
        double* grad_x2_f64 = (double*)grad_x2;

        for (size_t i = 0; i < size; i++) {
            grad_x2_f64[i] = -grad_c_f64[i];
        }
    }

    return 0;
}

int backwards_div_x1_host(const void* grad_c, const void* x2, void* grad_x1, size_t size, DType dtype) {
    if (!grad_c || !x2 || !grad_x1) return -1;

    if (dtype == DTYPE_FLOAT32) {
        const float* grad_c_f32 = (const float*)grad_c;
        const float* x2_f32 = (const float*)x2;
        float* grad_x1_f32 = (float*)grad_x1;

        for (size_t i = 0; i < size; i++) {
            grad_x1_f32[i] = grad_c_f32[i] / x2_f32[i];
        }
    } else {
        const double* grad_c_f64 = (const double*)grad_c;
        const double* x2_f64 = (const double*)x2;
        double* grad_x1_f64 = (double*)grad_x1;

        for (size_t i = 0; i < size; i++) {
            grad_x1_f64[i] = grad_c_f64[i] / x2_f64[i];
        }
    }

    return 0;
}

int backwards_div_x2_host(const void* grad_c, const void* x1, const void* x2, void* grad_x2, size_t size, DType dtype) {
    if (!grad_c || !x1 || !x2 || !grad_x2) return -1;

    if (dtype == DTYPE_FLOAT32) {
        const float* grad_c_f32 = (const float*)grad_c;
        const float* x1_f32 = (const float*)x1;
        const float* x2_f32 = (const float*)x2;
        float* grad_x2_f32 = (float*)grad_x2;

        for (size_t i = 0; i < size; i++) {
            grad_x2_f32[i] = -grad_c_f32[i] * x1_f32[i] / (x2_f32[i] * x2_f32[i]);
        }
    } else {
        const double* grad_c_f64 = (const double*)grad_c;
        const double* x1_f64 = (const double*)x1;
        const double* x2_f64 = (const double*)x2;
        double* grad_x2_f64 = (double*)grad_x2;

        for (size_t i = 0; i < size; i++) {
            grad_x2_f64[i] = -grad_c_f64[i] * x1_f64[i] / (x2_f64[i] * x2_f64[i]);
        }
    }

    return 0;
}

int backwards_exp_host(const void* grad_c, const void* fn_output, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !fn_output || !grad_x) return -1;

    if (dtype == DTYPE_FLOAT32) {
        const float* grad_c_f32 = (const float*)grad_c;
        const float* fn_output_f32 = (const float*)fn_output;
        float* grad_x_f32 = (float*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f32[i] = grad_c_f32[i] * fn_output_f32[i];
        }
    } else {
        const double* grad_c_f64 = (const double*)grad_c;
        const double* fn_output_f64 = (const double*)fn_output;
        double* grad_x_f64 = (double*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f64[i] = grad_c_f64[i] * fn_output_f64[i];
        }
    }

    return 0;
}

int backwards_log_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    if (dtype == DTYPE_FLOAT32) {
        const float* grad_c_f32 = (const float*)grad_c;
        const float* x_f32 = (const float*)x;
        float* grad_x_f32 = (float*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f32[i] = grad_c_f32[i] / x_f32[i];
        }
    } else {
        const double* grad_c_f64 = (const double*)grad_c;
        const double* x_f64 = (const double*)x;
        double* grad_x_f64 = (double*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f64[i] = grad_c_f64[i] / x_f64[i];
        }
    }

    return 0;
}

int backwards_sqrt_host(const void* grad_c, const void* fn_output, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !fn_output || !grad_x) return -1;

    if (dtype == DTYPE_FLOAT32) {
        const float* grad_c_f32 = (const float*)grad_c;
        const float* fn_output_f32 = (const float*)fn_output;
        float* grad_x_f32 = (float*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f32[i] = 0.5 * grad_c_f32[i] / fn_output_f32[i];
        }
    } else {
        const double* grad_c_f64 = (const double*)grad_c;
        const double* fn_output_f64 = (const double*)fn_output;
        double* grad_x_f64 = (double*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f64[i] = 0.5 * grad_c_f64[i] / fn_output_f64[i];
        }
    }

    return 0;
}

int backwards_tanh_host(const void* grad_c, const void* fn_output, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !fn_output || !grad_x) return -1;

    if (dtype == DTYPE_FLOAT32) {
        const float* grad_c_f32 = (const float*)grad_c;
        const float* fn_output_f32 = (const float*)fn_output;
        float* grad_x_f32 = (float*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f32[i] = grad_c_f32[i] * (1 - fn_output_f32[i] * fn_output_f32[i]);
        }
    } else {
        const double* grad_c_f64 = (const double*)grad_c;
        const double* fn_output_f64 = (const double*)fn_output;
        double* grad_x_f64 = (double*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f64[i] = grad_c_f64[i] * (1 - fn_output_f64[i] * fn_output_f64[i]);
        }
    }

    return 0;
}

int backwards_power_x1_host(const void* grad_c, const void* x1, const void* x2, const void* out, void* grad_x1, size_t size, DType dtype) {
    if (!grad_c || !x1 || !x2 || !out || !grad_x1) return -1;

    if (dtype == DTYPE_FLOAT32) {
        const float* grad_c_f32 = (const float*)grad_c;
        const float* x1_f32 = (const float*)x1;
        const float* x2_f32 = (const float*)x2;
        const float* out_f32 = (const float*)out;
        float* grad_x1_f32 = (float*)grad_x1;

        for (size_t i = 0; i < size; i++) {
            grad_x1_f32[i] = grad_c_f32[i] * x2_f32[i] * out_f32[i] / x1_f32[i];
        }
    } else {
        const double* grad_c_f64 = (const double*)grad_c;
        const double* x1_f64 = (const double*)x1;
        const double* x2_f64 = (const double*)x2;
        const double* out_f64 = (const double*)out;
        double* grad_x1_f64 = (double*)grad_x1;

        for (size_t i = 0; i < size; i++) {
            grad_x1_f64[i] = grad_c_f64[i] * x2_f64[i] * out_f64[i] / x1_f64[i];
        }
    }

    return 0;
}

int backwards_power_x2_host(const void* grad_c, const void* x1, const void* out, void* grad_x2, size_t size, DType dtype) {
    if (!grad_c || !x1 || !out || !grad_x2) return -1;

    if (dtype == DTYPE_FLOAT32) {
        const float* grad_c_f32 = (const float*)grad_c;
        const float* x1_f32 = (const float*)x1;
        const float* out_f32 = (const float*)out;
        float* grad_x2_f32 = (float*)grad_x2;

        for (size_t i = 0; i < size; i++) {
            grad_x2_f32[i] = grad_c_f32[i] * out_f32[i] * logf(x1_f32[i]);
        }
    } else {
        const double* grad_c_f64 = (const double*)grad_c;
        const double* x1_f64 = (const double*)x1;
        const double* out_f64 = (const double*)out;
        double* grad_x2_f64 = (double*)grad_x2;

        for (size_t i = 0; i < size; i++) {
            grad_x2_f64[i] = grad_c_f64[i] * out_f64[i] * log(x1_f64[i]);
        }
    }

    return 0;
}

int backwards_relu_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    if (dtype == DTYPE_FLOAT32) {
        const float* grad_c_f32 = (const float*)grad_c;
        const float* x_f32 = (const float*)x;
        float* grad_x_f32 = (float*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f32[i] = (x_f32[i] > 0.0f) ? grad_c_f32[i] : 0.0f;
        }
    } else {
        const double* grad_c_f64 = (const double*)grad_c;
        const double* x_f64 = (const double*)x;
        double* grad_x_f64 = (double*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f64[i] = (x_f64[i] > 0.0) ? grad_c_f64[i] : 0.0;
        }
    }

    return 0;
}

int backwards_sigmoid_host(const void* grad_c, const void* fn_output, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !fn_output || !grad_x) return -1;

    if (dtype == DTYPE_FLOAT32) {
        const float* grad_c_f32 = (const float*)grad_c;
        const float* fn_output_f32 = (const float*)fn_output;
        float* grad_x_f32 = (float*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f32[i] = grad_c_f32[i] * fn_output_f32[i] * (1.0f - fn_output_f32[i]);
        }
    } else {
        const double* grad_c_f64 = (const double*)grad_c;
        const double* fn_output_f64 = (const double*)fn_output;
        double* grad_x_f64 = (double*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f64[i] = grad_c_f64[i] * fn_output_f64[i] * (1.0 - fn_output_f64[i]);
        }
    }

    return 0;
}

int backwards_square_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    if (dtype == DTYPE_FLOAT32) {
        const float* grad_c_f32 = (const float*)grad_c;
        const float* x_f32 = (const float*)x;
        float* grad_x_f32 = (float*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f32[i] = grad_c_f32[i] * 2.0f * x_f32[i];
        }
    } else {
        const double* grad_c_f64 = (const double*)grad_c;
        const double* x_f64 = (const double*)x;
        double* grad_x_f64 = (double*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f64[i] = grad_c_f64[i] * 2.0 * x_f64[i];
        }
    }

    return 0;
}

int backwards_abs_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    if (dtype == DTYPE_FLOAT32) {
        const float* grad_c_f32 = (const float*)grad_c;
        const float* x_f32 = (const float*)x;
        float* grad_x_f32 = (float*)grad_x;

        for (size_t i = 0; i < size; i++) {
            if (x_f32[i] > 0.0f) {
                grad_x_f32[i] = grad_c_f32[i];
            } else if (x_f32[i] < 0.0f) {
                grad_x_f32[i] = -grad_c_f32[i];
            } else {
                grad_x_f32[i] = 0.0f;
            }
        }
    } else {
        const double* grad_c_f64 = (const double*)grad_c;
        const double* x_f64 = (const double*)x;
        double* grad_x_f64 = (double*)grad_x;

        for (size_t i = 0; i < size; i++) {
            if (x_f64[i] > 0.0) {
                grad_x_f64[i] = grad_c_f64[i];
            } else if (x_f64[i] < 0.0) {
                grad_x_f64[i] = -grad_c_f64[i];
            } else {
                grad_x_f64[i] = 0.0;
            }
        }
    }

    return 0;
}

int backwards_sin_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    if (dtype == DTYPE_FLOAT32) {
        const float* grad_c_f32 = (const float*)grad_c;
        const float* x_f32 = (const float*)x;
        float* grad_x_f32 = (float*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f32[i] = grad_c_f32[i] * cosf(x_f32[i]);
        }
    } else {
        const double* grad_c_f64 = (const double*)grad_c;
        const double* x_f64 = (const double*)x;
        double* grad_x_f64 = (double*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f64[i] = grad_c_f64[i] * cos(x_f64[i]);
        }
    }

    return 0;
}

int backwards_cos_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    if (dtype == DTYPE_FLOAT32) {
        const float* grad_c_f32 = (const float*)grad_c;
        const float* x_f32 = (const float*)x;
        float* grad_x_f32 = (float*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f32[i] = -grad_c_f32[i] * sinf(x_f32[i]);
        }
    } else {
        const double* grad_c_f64 = (const double*)grad_c;
        const double* x_f64 = (const double*)x;
        double* grad_x_f64 = (double*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f64[i] = -grad_c_f64[i] * sin(x_f64[i]);
        }
    }

    return 0;
}

int backwards_tan_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    if (dtype == DTYPE_FLOAT32) {
        const float* grad_c_f32 = (const float*)grad_c;
        const float* x_f32 = (const float*)x;
        float* grad_x_f32 = (float*)grad_x;

        for (size_t i = 0; i < size; i++) {
            float cos_x = cosf(x_f32[i]);
            grad_x_f32[i] = grad_c_f32[i] / (cos_x * cos_x);
        }
    } else {
        const double* grad_c_f64 = (const double*)grad_c;
        const double* x_f64 = (const double*)x;
        double* grad_x_f64 = (double*)grad_x;

        for (size_t i = 0; i < size; i++) {
            double cos_x = cos(x_f64[i]);
            grad_x_f64[i] = grad_c_f64[i] / (cos_x * cos_x);
        }
    }

    return 0;
}
