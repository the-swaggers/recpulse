#include "ops.h"
#include "../core/half_precision.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

#define HALF_CHECK (dtype == DTYPE_FLOAT16 || dtype == DTYPE_BFLOAT16)

#define HALF_GRAD_MEMCPY(grad_c, grad_x, size, dtype) do { \
    size_t _nbytes = (size) * dtype_size(dtype); \
    memcpy(grad_x, grad_c, _nbytes); \
    return 0; \
} while(0)

#define HALF_GRAD_1IN(body, grad_c, x, grad_x, size, dtype) do { \
    float* _gc = (float*)malloc((size) * sizeof(float)); \
    float* _x = (float*)malloc((size) * sizeof(float)); \
    float* _gx = (float*)malloc((size) * sizeof(float)); \
    if (!_gc || !_x || !_gx) { free(_gc); free(_x); free(_gx); return -1; } \
    half_to_fp32_array(grad_c, _gc, size, dtype); \
    half_to_fp32_array(x, _x, size, dtype); \
    { body } \
    fp32_to_half_array(_gx, (void*)(grad_x), size, dtype); \
    free(_gc); free(_x); free(_gx); \
    return 0; \
} while(0)

#define HALF_GRAD_1OUT(body, grad_c, fn_output, grad_x, size, dtype) do { \
    float* _gc = (float*)malloc((size) * sizeof(float)); \
    float* _fo = (float*)malloc((size) * sizeof(float)); \
    float* _gx = (float*)malloc((size) * sizeof(float)); \
    if (!_gc || !_fo || !_gx) { free(_gc); free(_fo); free(_gx); return -1; } \
    half_to_fp32_array(grad_c, _gc, size, dtype); \
    half_to_fp32_array(fn_output, _fo, size, dtype); \
    { body } \
    fp32_to_half_array(_gx, (void*)(grad_x), size, dtype); \
    free(_gc); free(_fo); free(_gx); \
    return 0; \
} while(0)

int backwards_add_x1_host(const void* grad_c, void* grad_x1, size_t size, DType dtype) {
    if (!grad_c || !grad_x1) return -1;

    size_t element_size = dtype_size(dtype);
    size_t nbytes = size * element_size;

    memcpy(grad_x1, grad_c, nbytes);
    return 0;
}

int backwards_add_x2_host(const void* grad_c, void* grad_x2, size_t size, DType dtype) {
    if (!grad_c || !grad_x2) return -1;

    size_t element_size = dtype_size(dtype);
    size_t nbytes = size * element_size;

    memcpy(grad_x2, grad_c, nbytes);
    return 0;
}

int backwards_mul_x1_host(const void* grad_c, const void* x2, void* grad_x1, size_t size, DType dtype) {
    if (!grad_c || !x2 || !grad_x1) return -1;

    if (HALF_CHECK) {
        float* gc_f32 = (float*)malloc(size * sizeof(float));
        float* x2_f32 = (float*)malloc(size * sizeof(float));
        float* gx1_f32 = (float*)malloc(size * sizeof(float));
        if (!gc_f32 || !x2_f32 || !gx1_f32) { free(gc_f32); free(x2_f32); free(gx1_f32); return -1; }
        half_to_fp32_array(grad_c, gc_f32, size, dtype);
        half_to_fp32_array(x2, x2_f32, size, dtype);
        for (size_t i = 0; i < size; i++) {
            gx1_f32[i] = gc_f32[i] * x2_f32[i];
        }
        fp32_to_half_array(gx1_f32, grad_x1, size, dtype);
        free(gc_f32); free(x2_f32); free(gx1_f32);
        return 0;
    }

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

    if (HALF_CHECK) {
        float* gc_f32 = (float*)malloc(size * sizeof(float));
        float* x1_f32 = (float*)malloc(size * sizeof(float));
        float* gx2_f32 = (float*)malloc(size * sizeof(float));
        if (!gc_f32 || !x1_f32 || !gx2_f32) { free(gc_f32); free(x1_f32); free(gx2_f32); return -1; }
        half_to_fp32_array(grad_c, gc_f32, size, dtype);
        half_to_fp32_array(x1, x1_f32, size, dtype);
        for (size_t i = 0; i < size; i++) {
            gx2_f32[i] = gc_f32[i] * x1_f32[i];
        }
        fp32_to_half_array(gx2_f32, grad_x2, size, dtype);
        free(gc_f32); free(x1_f32); free(gx2_f32);
        return 0;
    }

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

    size_t element_size = dtype_size(dtype);
    size_t nbytes = size * element_size;

    memcpy(grad_x1, grad_c, nbytes);
    return 0;
}

int backwards_sub_x2_host(const void* grad_c, void* grad_x2, size_t size, DType dtype) {
    if (!grad_c || !grad_x2) return -1;

    if (HALF_CHECK) {
        float* gc_f32 = (float*)malloc(size * sizeof(float));
        float* gx2_f32 = (float*)malloc(size * sizeof(float));
        if (!gc_f32 || !gx2_f32) { free(gc_f32); free(gx2_f32); return -1; }
        half_to_fp32_array(grad_c, gc_f32, size, dtype);
        for (size_t i = 0; i < size; i++) {
            gx2_f32[i] = -gc_f32[i];
        }
        fp32_to_half_array(gx2_f32, grad_x2, size, dtype);
        free(gc_f32); free(gx2_f32);
        return 0;
    }

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

    if (HALF_CHECK) {
        float* gc_f32 = (float*)malloc(size * sizeof(float));
        float* x2_f32 = (float*)malloc(size * sizeof(float));
        float* gx1_f32 = (float*)malloc(size * sizeof(float));
        if (!gc_f32 || !x2_f32 || !gx1_f32) { free(gc_f32); free(x2_f32); free(gx1_f32); return -1; }
        half_to_fp32_array(grad_c, gc_f32, size, dtype);
        half_to_fp32_array(x2, x2_f32, size, dtype);
        for (size_t i = 0; i < size; i++) {
            gx1_f32[i] = gc_f32[i] / x2_f32[i];
        }
        fp32_to_half_array(gx1_f32, grad_x1, size, dtype);
        free(gc_f32); free(x2_f32); free(gx1_f32);
        return 0;
    }

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

    if (HALF_CHECK) {
        float* gc_f32 = (float*)malloc(size * sizeof(float));
        float* x1_f32 = (float*)malloc(size * sizeof(float));
        float* x2_f32 = (float*)malloc(size * sizeof(float));
        float* gx2_f32 = (float*)malloc(size * sizeof(float));
        if (!gc_f32 || !x1_f32 || !x2_f32 || !gx2_f32) { free(gc_f32); free(x1_f32); free(x2_f32); free(gx2_f32); return -1; }
        half_to_fp32_array(grad_c, gc_f32, size, dtype);
        half_to_fp32_array(x1, x1_f32, size, dtype);
        half_to_fp32_array(x2, x2_f32, size, dtype);
        for (size_t i = 0; i < size; i++) {
            gx2_f32[i] = -gc_f32[i] * x1_f32[i] / (x2_f32[i] * x2_f32[i]);
        }
        fp32_to_half_array(gx2_f32, grad_x2, size, dtype);
        free(gc_f32); free(x1_f32); free(x2_f32); free(gx2_f32);
        return 0;
    }

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

    if (HALF_CHECK) {
        HALF_GRAD_1OUT(
            for (size_t i = 0; i < size; i++) {
                _gx[i] = _gc[i] * _fo[i];
            },
            grad_c, fn_output, grad_x, size, dtype
        );
    }

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

    if (HALF_CHECK) {
        HALF_GRAD_1IN(
            for (size_t i = 0; i < size; i++) {
                _gx[i] = _gc[i] / _x[i];
            },
            grad_c, x, grad_x, size, dtype
        );
    }

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

    if (HALF_CHECK) {
        HALF_GRAD_1OUT(
            for (size_t i = 0; i < size; i++) {
                _gx[i] = 0.5f * _gc[i] / _fo[i];
            },
            grad_c, fn_output, grad_x, size, dtype
        );
    }

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

    if (HALF_CHECK) {
        HALF_GRAD_1OUT(
            for (size_t i = 0; i < size; i++) {
                _gx[i] = _gc[i] * (1.0f - _fo[i] * _fo[i]);
            },
            grad_c, fn_output, grad_x, size, dtype
        );
    }

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

    if (HALF_CHECK) {
        float* gc_f32 = (float*)malloc(size * sizeof(float));
        float* x1_f32 = (float*)malloc(size * sizeof(float));
        float* x2_f32 = (float*)malloc(size * sizeof(float));
        float* out_f32 = (float*)malloc(size * sizeof(float));
        float* gx1_f32 = (float*)malloc(size * sizeof(float));
        if (!gc_f32 || !x1_f32 || !x2_f32 || !out_f32 || !gx1_f32) {
            free(gc_f32); free(x1_f32); free(x2_f32); free(out_f32); free(gx1_f32);
            return -1;
        }
        half_to_fp32_array(grad_c, gc_f32, size, dtype);
        half_to_fp32_array(x1, x1_f32, size, dtype);
        half_to_fp32_array(x2, x2_f32, size, dtype);
        half_to_fp32_array(out, out_f32, size, dtype);
        for (size_t i = 0; i < size; i++) {
            gx1_f32[i] = gc_f32[i] * x2_f32[i] * out_f32[i] / x1_f32[i];
        }
        fp32_to_half_array(gx1_f32, grad_x1, size, dtype);
        free(gc_f32); free(x1_f32); free(x2_f32); free(out_f32); free(gx1_f32);
        return 0;
    }

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

    if (HALF_CHECK) {
        float* gc_f32 = (float*)malloc(size * sizeof(float));
        float* x1_f32 = (float*)malloc(size * sizeof(float));
        float* out_f32 = (float*)malloc(size * sizeof(float));
        float* gx2_f32 = (float*)malloc(size * sizeof(float));
        if (!gc_f32 || !x1_f32 || !out_f32 || !gx2_f32) {
            free(gc_f32); free(x1_f32); free(out_f32); free(gx2_f32);
            return -1;
        }
        half_to_fp32_array(grad_c, gc_f32, size, dtype);
        half_to_fp32_array(x1, x1_f32, size, dtype);
        half_to_fp32_array(out, out_f32, size, dtype);
        for (size_t i = 0; i < size; i++) {
            gx2_f32[i] = gc_f32[i] * out_f32[i] * logf(x1_f32[i]);
        }
        fp32_to_half_array(gx2_f32, grad_x2, size, dtype);
        free(gc_f32); free(x1_f32); free(out_f32); free(gx2_f32);
        return 0;
    }

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

int backwards_logb_x1_host(const void* grad_c, const void* x1, const void* x2, void* grad_x1, size_t size, DType dtype) {
    if (!grad_c || !x1 || !x2 || !grad_x1) return -1;

    if (HALF_CHECK) {
        float* gc_f32 = (float*)malloc(size * sizeof(float));
        float* x1_f32 = (float*)malloc(size * sizeof(float));
        float* x2_f32 = (float*)malloc(size * sizeof(float));
        float* gx1_f32 = (float*)malloc(size * sizeof(float));
        if (!gc_f32 || !x1_f32 || !x2_f32 || !gx1_f32) {
            free(gc_f32); free(x1_f32); free(x2_f32); free(gx1_f32);
            return -1;
        }
        half_to_fp32_array(grad_c, gc_f32, size, dtype);
        half_to_fp32_array(x1, x1_f32, size, dtype);
        half_to_fp32_array(x2, x2_f32, size, dtype);
        for (size_t i = 0; i < size; i++) {
            gx1_f32[i] = gc_f32[i] / (x1_f32[i] * logf(x2_f32[i]));
        }
        fp32_to_half_array(gx1_f32, grad_x1, size, dtype);
        free(gc_f32); free(x1_f32); free(x2_f32); free(gx1_f32);
        return 0;
    }

    if (dtype == DTYPE_FLOAT32) {
        const float* grad_c_f32 = (const float*)grad_c;
        const float* x1_f32 = (const float*)x1;
        const float* x2_f32 = (const float*)x2;
        float* grad_x1_f32 = (float*)grad_x1;

        for (size_t i = 0; i < size; i++) {
            grad_x1_f32[i] = grad_c_f32[i] / (x1_f32[i] * logf(x2_f32[i]));
        }
    } else {
        const double* grad_c_f64 = (const double*)grad_c;
        const double* x1_f64 = (const double*)x1;
        const double* x2_f64 = (const double*)x2;
        double* grad_x1_f64 = (double*)grad_x1;

        for (size_t i = 0; i < size; i++) {
            grad_x1_f64[i] = grad_c_f64[i] / (x1_f64[i] * log(x2_f64[i]));
        }
    }

    return 0;
}

int backwards_logb_x2_host(const void* grad_c, const void* x1, const void* x2, void* grad_x2, size_t size, DType dtype) {
    if (!grad_c || !x1 || !x2 || !grad_x2) return -1;

    if (HALF_CHECK) {
        float* gc_f32 = (float*)malloc(size * sizeof(float));
        float* x1_f32 = (float*)malloc(size * sizeof(float));
        float* x2_f32 = (float*)malloc(size * sizeof(float));
        float* gx2_f32 = (float*)malloc(size * sizeof(float));
        if (!gc_f32 || !x1_f32 || !x2_f32 || !gx2_f32) {
            free(gc_f32); free(x1_f32); free(x2_f32); free(gx2_f32);
            return -1;
        }
        half_to_fp32_array(grad_c, gc_f32, size, dtype);
        half_to_fp32_array(x1, x1_f32, size, dtype);
        half_to_fp32_array(x2, x2_f32, size, dtype);
        for (size_t i = 0; i < size; i++) {
            float log_x2 = logf(x2_f32[i]);
            gx2_f32[i] = -gc_f32[i] * logf(x1_f32[i]) / (x2_f32[i] * log_x2 * log_x2);
        }
        fp32_to_half_array(gx2_f32, grad_x2, size, dtype);
        free(gc_f32); free(x1_f32); free(x2_f32); free(gx2_f32);
        return 0;
    }

    if (dtype == DTYPE_FLOAT32) {
        const float* grad_c_f32 = (const float*)grad_c;
        const float* x1_f32 = (const float*)x1;
        const float* x2_f32 = (const float*)x2;
        float* grad_x2_f32 = (float*)grad_x2;

        for (size_t i = 0; i < size; i++) {
            float log_x2 = logf(x2_f32[i]);
            grad_x2_f32[i] = -grad_c_f32[i] * logf(x1_f32[i]) / (x2_f32[i] * log_x2 * log_x2);
        }
    } else {
        const double* grad_c_f64 = (const double*)grad_c;
        const double* x1_f64 = (const double*)x1;
        const double* x2_f64 = (const double*)x2;
        double* grad_x2_f64 = (double*)grad_x2;

        for (size_t i = 0; i < size; i++) {
            double log_x2 = log(x2_f64[i]);
            grad_x2_f64[i] = -grad_c_f64[i] * log(x1_f64[i]) / (x2_f64[i] * log_x2 * log_x2);
        }
    }

    return 0;
}

int backwards_relu_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    if (HALF_CHECK) {
        HALF_GRAD_1IN(
            for (size_t i = 0; i < size; i++) {
                _gx[i] = (_x[i] > 0.0f) ? _gc[i] : 0.0f;
            },
            grad_c, x, grad_x, size, dtype
        );
    }

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

    if (HALF_CHECK) {
        HALF_GRAD_1OUT(
            for (size_t i = 0; i < size; i++) {
                _gx[i] = _gc[i] * _fo[i] * (1.0f - _fo[i]);
            },
            grad_c, fn_output, grad_x, size, dtype
        );
    }

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

    if (HALF_CHECK) {
        HALF_GRAD_1IN(
            for (size_t i = 0; i < size; i++) {
                _gx[i] = _gc[i] * 2.0f * _x[i];
            },
            grad_c, x, grad_x, size, dtype
        );
    }

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

    if (HALF_CHECK) {
        HALF_GRAD_1IN(
            for (size_t i = 0; i < size; i++) {
                if (_x[i] > 0.0f) {
                    _gx[i] = _gc[i];
                } else if (_x[i] < 0.0f) {
                    _gx[i] = -_gc[i];
                } else {
                    _gx[i] = 0.0f;
                }
            },
            grad_c, x, grad_x, size, dtype
        );
    }

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

    if (HALF_CHECK) {
        HALF_GRAD_1IN(
            for (size_t i = 0; i < size; i++) {
                _gx[i] = _gc[i] * cosf(_x[i]);
            },
            grad_c, x, grad_x, size, dtype
        );
    }

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

    if (HALF_CHECK) {
        HALF_GRAD_1IN(
            for (size_t i = 0; i < size; i++) {
                _gx[i] = -_gc[i] * sinf(_x[i]);
            },
            grad_c, x, grad_x, size, dtype
        );
    }

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

    if (HALF_CHECK) {
        HALF_GRAD_1IN(
            for (size_t i = 0; i < size; i++) {
                float cos_x = cosf(_x[i]);
                _gx[i] = _gc[i] / (cos_x * cos_x);
            },
            grad_c, x, grad_x, size, dtype
        );
    }

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

int backwards_asin_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    if (HALF_CHECK) {
        HALF_GRAD_1IN(
            for (size_t i = 0; i < size; i++) {
                _gx[i] = _gc[i] / sqrtf(1.0f - _x[i] * _x[i]);
            },
            grad_c, x, grad_x, size, dtype
        );
    }

    if (dtype == DTYPE_FLOAT32) {
        const float* grad_c_f32 = (const float*)grad_c;
        const float* x_f32 = (const float*)x;
        float* grad_x_f32 = (float*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f32[i] = grad_c_f32[i] / sqrtf(1.0f - x_f32[i] * x_f32[i]);
        }
    } else {
        const double* grad_c_f64 = (const double*)grad_c;
        const double* x_f64 = (const double*)x;
        double* grad_x_f64 = (double*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f64[i] = grad_c_f64[i] / sqrt(1.0 - x_f64[i] * x_f64[i]);
        }
    }

    return 0;
}

int backwards_acos_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    if (HALF_CHECK) {
        HALF_GRAD_1IN(
            for (size_t i = 0; i < size; i++) {
                _gx[i] = -_gc[i] / sqrtf(1.0f - _x[i] * _x[i]);
            },
            grad_c, x, grad_x, size, dtype
        );
    }

    if (dtype == DTYPE_FLOAT32) {
        const float* grad_c_f32 = (const float*)grad_c;
        const float* x_f32 = (const float*)x;
        float* grad_x_f32 = (float*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f32[i] = -grad_c_f32[i] / sqrtf(1.0f - x_f32[i] * x_f32[i]);
        }
    } else {
        const double* grad_c_f64 = (const double*)grad_c;
        const double* x_f64 = (const double*)x;
        double* grad_x_f64 = (double*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f64[i] = -grad_c_f64[i] / sqrt(1.0 - x_f64[i] * x_f64[i]);
        }
    }

    return 0;
}

int backwards_atan_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    if (HALF_CHECK) {
        HALF_GRAD_1IN(
            for (size_t i = 0; i < size; i++) {
                _gx[i] = _gc[i] / (1.0f + _x[i] * _x[i]);
            },
            grad_c, x, grad_x, size, dtype
        );
    }

    if (dtype == DTYPE_FLOAT32) {
        const float* grad_c_f32 = (const float*)grad_c;
        const float* x_f32 = (const float*)x;
        float* grad_x_f32 = (float*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f32[i] = grad_c_f32[i] / (1.0f + x_f32[i] * x_f32[i]);
        }
    } else {
        const double* grad_c_f64 = (const double*)grad_c;
        const double* x_f64 = (const double*)x;
        double* grad_x_f64 = (double*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f64[i] = grad_c_f64[i] / (1.0 + x_f64[i] * x_f64[i]);
        }
    }

    return 0;
}

int backwards_sinh_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    if (HALF_CHECK) {
        HALF_GRAD_1IN(
            for (size_t i = 0; i < size; i++) {
                _gx[i] = _gc[i] * coshf(_x[i]);
            },
            grad_c, x, grad_x, size, dtype
        );
    }

    if (dtype == DTYPE_FLOAT32) {
        const float* grad_c_f32 = (const float*)grad_c;
        const float* x_f32 = (const float*)x;
        float* grad_x_f32 = (float*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f32[i] = grad_c_f32[i] * coshf(x_f32[i]);
        }
    } else {
        const double* grad_c_f64 = (const double*)grad_c;
        const double* x_f64 = (const double*)x;
        double* grad_x_f64 = (double*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f64[i] = grad_c_f64[i] * cosh(x_f64[i]);
        }
    }

    return 0;
}

int backwards_cosh_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    if (HALF_CHECK) {
        HALF_GRAD_1IN(
            for (size_t i = 0; i < size; i++) {
                _gx[i] = _gc[i] * sinhf(_x[i]);
            },
            grad_c, x, grad_x, size, dtype
        );
    }

    if (dtype == DTYPE_FLOAT32) {
        const float* grad_c_f32 = (const float*)grad_c;
        const float* x_f32 = (const float*)x;
        float* grad_x_f32 = (float*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f32[i] = grad_c_f32[i] * sinhf(x_f32[i]);
        }
    } else {
        const double* grad_c_f64 = (const double*)grad_c;
        const double* x_f64 = (const double*)x;
        double* grad_x_f64 = (double*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f64[i] = grad_c_f64[i] * sinh(x_f64[i]);
        }
    }

    return 0;
}

int backwards_gelu_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    if (HALF_CHECK) {
        HALF_GRAD_1IN(
            const float sqrt_2_over_pi = 0.7978845608f;
            const float coeff = 0.044715f;
            const float coeff3 = 0.134145f;
            for (size_t i = 0; i < size; i++) {
                float x_val = _x[i];
                float x_sq = x_val * x_val;
                float x_cubed = x_sq * x_val;
                float inner = sqrt_2_over_pi * (x_val + coeff * x_cubed);
                float tanh_inner = tanhf(inner);
                float sech2_inner = 1.0f - tanh_inner * tanh_inner;
                float d_inner = sqrt_2_over_pi * (1.0f + coeff3 * x_sq);
                float gelu_grad = 0.5f * (1.0f + tanh_inner + x_val * sech2_inner * d_inner);
                _gx[i] = _gc[i] * gelu_grad;
            },
            grad_c, x, grad_x, size, dtype
        );
    }

    if (dtype == DTYPE_FLOAT32) {
        const float* grad_c_f32 = (const float*)grad_c;
        const float* x_f32 = (const float*)x;
        float* grad_x_f32 = (float*)grad_x;

        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;
        const float coeff3 = 0.134145f;

        for (size_t i = 0; i < size; i++) {
            float x_val = x_f32[i];
            float x_sq = x_val * x_val;
            float x_cubed = x_sq * x_val;
            float inner = sqrt_2_over_pi * (x_val + coeff * x_cubed);
            float tanh_inner = tanhf(inner);
            float sech2_inner = 1.0f - tanh_inner * tanh_inner;
            float d_inner = sqrt_2_over_pi * (1.0f + coeff3 * x_sq);
            float gelu_grad = 0.5f * (1.0f + tanh_inner + x_val * sech2_inner * d_inner);
            grad_x_f32[i] = grad_c_f32[i] * gelu_grad;
        }
    } else {
        const double* grad_c_f64 = (const double*)grad_c;
        const double* x_f64 = (const double*)x;
        double* grad_x_f64 = (double*)grad_x;

        const double sqrt_2_over_pi = 0.7978845608;
        const double coeff = 0.044715;
        const double coeff3 = 0.134145;

        for (size_t i = 0; i < size; i++) {
            double x_val = x_f64[i];
            double x_sq = x_val * x_val;
            double x_cubed = x_sq * x_val;
            double inner = sqrt_2_over_pi * (x_val + coeff * x_cubed);
            double tanh_inner = tanh(inner);
            double sech2_inner = 1.0 - tanh_inner * tanh_inner;
            double d_inner = sqrt_2_over_pi * (1.0 + coeff3 * x_sq);
            double gelu_grad = 0.5 * (1.0 + tanh_inner + x_val * sech2_inner * d_inner);
            grad_x_f64[i] = grad_c_f64[i] * gelu_grad;
        }
    }

    return 0;
}

int backwards_silu_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    if (HALF_CHECK) {
        HALF_GRAD_1IN(
            for (size_t i = 0; i < size; i++) {
                float x_val = _x[i];
                float sigmoid = 1.0f / (1.0f + expf(-x_val));
                float silu_grad = sigmoid * (1.0f + x_val * (1.0f - sigmoid));
                _gx[i] = _gc[i] * silu_grad;
            },
            grad_c, x, grad_x, size, dtype
        );
    }

    if (dtype == DTYPE_FLOAT32) {
        const float* grad_c_f32 = (const float*)grad_c;
        const float* x_f32 = (const float*)x;
        float* grad_x_f32 = (float*)grad_x;

        for (size_t i = 0; i < size; i++) {
            float x_val = x_f32[i];
            float sigmoid = 1.0f / (1.0f + expf(-x_val));
            float silu_grad = sigmoid * (1.0f + x_val * (1.0f - sigmoid));
            grad_x_f32[i] = grad_c_f32[i] * silu_grad;
        }
    } else {
        const double* grad_c_f64 = (const double*)grad_c;
        const double* x_f64 = (const double*)x;
        double* grad_x_f64 = (double*)grad_x;

        for (size_t i = 0; i < size; i++) {
            double x_val = x_f64[i];
            double sigmoid = 1.0 / (1.0 + exp(-x_val));
            double silu_grad = sigmoid * (1.0 + x_val * (1.0 - sigmoid));
            grad_x_f64[i] = grad_c_f64[i] * silu_grad;
        }
    }

    return 0;
}

int backwards_leaky_relu_host(const void* grad_c, const void* x, const void* alpha, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !alpha || !grad_x) return -1;

    if (HALF_CHECK) {
        float* gc_f32 = (float*)malloc(size * sizeof(float));
        float* x_f32 = (float*)malloc(size * sizeof(float));
        float* gx_f32 = (float*)malloc(size * sizeof(float));
        if (!gc_f32 || !x_f32 || !gx_f32) { free(gc_f32); free(x_f32); free(gx_f32); return -1; }
        half_to_fp32_array(grad_c, gc_f32, size, dtype);
        half_to_fp32_array(x, x_f32, size, dtype);
        float alpha_val = *(const float*)alpha;
        for (size_t i = 0; i < size; i++) {
            gx_f32[i] = gc_f32[i] * ((x_f32[i] > 0.0f) ? 1.0f : alpha_val);
        }
        fp32_to_half_array(gx_f32, grad_x, size, dtype);
        free(gc_f32); free(x_f32); free(gx_f32);
        return 0;
    }

    if (dtype == DTYPE_FLOAT32) {
        const float* grad_c_f32 = (const float*)grad_c;
        const float* x_f32 = (const float*)x;
        float alpha_val = *(const float*)alpha;
        float* grad_x_f32 = (float*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f32[i] = grad_c_f32[i] * ((x_f32[i] > 0.0f) ? 1.0f : alpha_val);
        }
    } else {
        const double* grad_c_f64 = (const double*)grad_c;
        const double* x_f64 = (const double*)x;
        double alpha_val = *(const double*)alpha;
        double* grad_x_f64 = (double*)grad_x;

        for (size_t i = 0; i < size; i++) {
            grad_x_f64[i] = grad_c_f64[i] * ((x_f64[i] > 0.0) ? 1.0 : alpha_val);
        }
    }

    return 0;
}

int backwards_rsqrt_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !x || !grad_x) return -1;

    if (HALF_CHECK) {
        HALF_GRAD_1IN(
            for (size_t i = 0; i < size; i++) {
                float x_val = _x[i];
                _gx[i] = _gc[i] * (-0.5f) / (x_val * sqrtf(x_val));
            },
            grad_c, x, grad_x, size, dtype
        );
    }

    if (dtype == DTYPE_FLOAT32) {
        const float* grad_c_f32 = (const float*)grad_c;
        const float* x_f32 = (const float*)x;
        float* grad_x_f32 = (float*)grad_x;

        for (size_t i = 0; i < size; i++) {
            float x_val = x_f32[i];
            grad_x_f32[i] = grad_c_f32[i] * (-0.5f) / (x_val * sqrtf(x_val));
        }
    } else {
        const double* grad_c_f64 = (const double*)grad_c;
        const double* x_f64 = (const double*)x;
        double* grad_x_f64 = (double*)grad_x;

        for (size_t i = 0; i < size; i++) {
            double x_val = x_f64[i];
            grad_x_f64[i] = grad_c_f64[i] * (-0.5) / (x_val * sqrt(x_val));
        }
    }

    return 0;
}

int backwards_sum_all_host(const void* grad_c, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !grad_x) return -1;

    if (HALF_CHECK) {
        float gc_f32;
        half_to_fp32_array(grad_c, &gc_f32, 1, dtype);
        float* gx_f32 = (float*)malloc(size * sizeof(float));
        if (!gx_f32) return -1;
        for (size_t i = 0; i < size; i++) {
            gx_f32[i] = gc_f32;
        }
        fp32_to_half_array(gx_f32, grad_x, size, dtype);
        free(gx_f32);
        return 0;
    }

    if (dtype == DTYPE_FLOAT32) {
        float val = *(const float*)grad_c;
        float* grad_x_f32 = (float*)grad_x;
        for (size_t i = 0; i < size; i++) {
            grad_x_f32[i] = val;
        }
    } else {
        double val = *(const double*)grad_c;
        double* grad_x_f64 = (double*)grad_x;
        for (size_t i = 0; i < size; i++) {
            grad_x_f64[i] = val;
        }
    }

    return 0;
}

int backwards_mean_all_host(const void* grad_c, void* grad_x, size_t size, DType dtype) {
    if (!grad_c || !grad_x) return -1;

    if (HALF_CHECK) {
        float gc_f32;
        half_to_fp32_array(grad_c, &gc_f32, 1, dtype);
        float val = gc_f32 / (float)size;
        float* gx_f32 = (float*)malloc(size * sizeof(float));
        if (!gx_f32) return -1;
        for (size_t i = 0; i < size; i++) {
            gx_f32[i] = val;
        }
        fp32_to_half_array(gx_f32, grad_x, size, dtype);
        free(gx_f32);
        return 0;
    }

    if (dtype == DTYPE_FLOAT32) {
        float val = *(const float*)grad_c / (float)size;
        float* grad_x_f32 = (float*)grad_x;
        for (size_t i = 0; i < size; i++) {
            grad_x_f32[i] = val;
        }
    } else {
        double val = *(const double*)grad_c / (double)size;
        double* grad_x_f64 = (double*)grad_x;
        for (size_t i = 0; i < size; i++) {
            grad_x_f64[i] = val;
        }
    }

    return 0;
}
