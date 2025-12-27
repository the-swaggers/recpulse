#include "ops.h"
#include <string.h>

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
