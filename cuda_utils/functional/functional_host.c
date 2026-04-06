#include "functional.h"
#include "../core/half_precision.h"
#include "../rand/rand.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <float.h>

int add_kernel_host_f32(float* out, const float* x1, const float* x2, size_t size) {
    if (!out || !x1 || !x2 || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = x1[i] + x2[i];
    }
    return 0;
}

int add_kernel_host_f64(double* out, const double* x1, const double* x2, size_t size) {
    if (!out || !x1 || !x2 || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = x1[i] + x2[i];
    }
    return 0;
}

int sub_kernel_host_f32(float* out, const float* x1, const float* x2, size_t size) {
    if (!out || !x1 || !x2 || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = x1[i] - x2[i];
    }
    return 0;
}

int sub_kernel_host_f64(double* out, const double* x1, const double* x2, size_t size) {
    if (!out || !x1 || !x2 || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = x1[i] - x2[i];
    }
    return 0;
}

int mul_kernel_host_f32(float* out, const float* x1, const float* x2, size_t size) {
    if (!out || !x1 || !x2 || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = x1[i] * x2[i];
    }
    return 0;
}

int mul_kernel_host_f64(double* out, const double* x1, const double* x2, size_t size) {
    if (!out || !x1 || !x2 || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = x1[i] * x2[i];
    }
    return 0;
}

int div_kernel_host_f32(float* out, const float* x1, const float* x2, size_t size) {
    if (!out || !x1 || !x2 || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = x1[i] / x2[i];
    }
    return 0;
}

int div_kernel_host_f64(double* out, const double* x1, const double* x2, size_t size) {
    if (!out || !x1 || !x2 || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = x1[i] / x2[i];
    }
    return 0;
}

int pow_kernel_host_f32(float* out, const float* x1, const float* x2, size_t size) {
    if (!out || !x1 || !x2 || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = powf(x1[i], x2[i]);
    }
    return 0;
}

int pow_kernel_host_f64(double* out, const double* x1, const double* x2, size_t size) {
    if (!out || !x1 || !x2 || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = pow(x1[i], x2[i]);
    }
    return 0;
}

int logb_kernel_host_f32(float* out, const float* x1, const float* x2, size_t size) {
    if (!out || !x1 || !x2 || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = logf(x1[i]) / logf(x2[i]);
    }
    return 0;
}

int logb_kernel_host_f64(double* out, const double* x1, const double* x2, size_t size) {
    if (!out || !x1 || !x2 || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = log(x1[i]) / log(x2[i]);
    }
    return 0;
}

#define STRIDED_BINARY_HOST(name, T, op_expr) \
int name(T* out, const T* x1, const T* x2, int ndim, const int* out_shape, \
         const int* x1_strides, const int* x2_strides, size_t out_size) { \
    if (!out || !x1 || !x2 || out_size == 0) return -1; \
    int* out_strides = (int*)malloc(ndim * sizeof(int)); \
    if (!out_strides) return -1; \
    out_strides[ndim - 1] = 1; \
    for (int d = ndim - 2; d >= 0; d--) { \
        out_strides[d] = out_strides[d + 1] * out_shape[d + 1]; \
    } \
    for (size_t i = 0; i < out_size; i++) { \
        size_t rem = i; \
        size_t idx1 = 0, idx2 = 0; \
        for (int d = 0; d < ndim; d++) { \
            int coord = rem / out_strides[d]; \
            rem %= out_strides[d]; \
            idx1 += coord * x1_strides[d]; \
            idx2 += coord * x2_strides[d]; \
        } \
        T a = x1[idx1]; \
        T b = x2[idx2]; \
        out[i] = op_expr; \
    } \
    free(out_strides); \
    return 0; \
}

STRIDED_BINARY_HOST(add_strided_kernel_host_f32, float, a + b)
STRIDED_BINARY_HOST(add_strided_kernel_host_f64, double, a + b)
STRIDED_BINARY_HOST(sub_strided_kernel_host_f32, float, a - b)
STRIDED_BINARY_HOST(sub_strided_kernel_host_f64, double, a - b)
STRIDED_BINARY_HOST(mul_strided_kernel_host_f32, float, a * b)
STRIDED_BINARY_HOST(mul_strided_kernel_host_f64, double, a * b)
STRIDED_BINARY_HOST(div_strided_kernel_host_f32, float, a / b)
STRIDED_BINARY_HOST(div_strided_kernel_host_f64, double, a / b)
STRIDED_BINARY_HOST(pow_strided_kernel_host_f32, float, powf(a, b))
STRIDED_BINARY_HOST(pow_strided_kernel_host_f64, double, pow(a, b))
STRIDED_BINARY_HOST(logb_strided_kernel_host_f32, float, logf(a) / logf(b))
STRIDED_BINARY_HOST(logb_strided_kernel_host_f64, double, log(a) / log(b))

int add_scalar_kernel_host_f32(float* out, const float* x, float scalar, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = x[i] + scalar;
    }
    return 0;
}

int add_scalar_kernel_host_f64(double* out, const double* x, double scalar, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = x[i] + scalar;
    }
    return 0;
}

int sub_scalar_kernel_host_f32(float* out, const float* x, float scalar, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = x[i] - scalar;
    }
    return 0;
}

int sub_scalar_kernel_host_f64(double* out, const double* x, double scalar, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = x[i] - scalar;
    }
    return 0;
}

int mul_scalar_kernel_host_f32(float* out, const float* x, float scalar, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = x[i] * scalar;
    }
    return 0;
}

int mul_scalar_kernel_host_f64(double* out, const double* x, double scalar, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = x[i] * scalar;
    }
    return 0;
}

int div_scalar_kernel_host_f32(float* out, const float* x, float scalar, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = x[i] / scalar;
    }
    return 0;
}

int div_scalar_kernel_host_f64(double* out, const double* x, double scalar, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = x[i] / scalar;
    }
    return 0;
}

int pow_scalar_kernel_host_f32(float* out, const float* x, float scalar, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = powf(x[i], scalar);
    }
    return 0;
}

int pow_scalar_kernel_host_f64(double* out, const double* x, double scalar, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = pow(x[i], scalar);
    }
    return 0;
}

int rsub_scalar_kernel_host_f32(float* out, float scalar, const float* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = scalar - x[i];
    }
    return 0;
}

int rsub_scalar_kernel_host_f64(double* out, double scalar, const double* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = scalar - x[i];
    }
    return 0;
}

int rdiv_scalar_kernel_host_f32(float* out, float scalar, const float* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = scalar / x[i];
    }
    return 0;
}

int rdiv_scalar_kernel_host_f64(double* out, double scalar, const double* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = scalar / x[i];
    }
    return 0;
}

int rpow_scalar_kernel_host_f32(float* out, float scalar, const float* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = powf(scalar, x[i]);
    }
    return 0;
}

int rpow_scalar_kernel_host_f64(double* out, double scalar, const double* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = pow(scalar, x[i]);
    }
    return 0;
}

int logb_scalar_kernel_host_f32(float* out, const float* x, float scalar, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = logf(x[i]) / logf(scalar);
    }
    return 0;
}

int logb_scalar_kernel_host_f64(double* out, const double* x, double scalar, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = log(x[i]) / log(scalar);
    }
    return 0;
}

int rlogb_scalar_kernel_host_f32(float* out, float scalar, const float* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = logf(scalar) / logf(x[i]);
    }
    return 0;
}

int rlogb_scalar_kernel_host_f64(double* out, double scalar, const double* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = log(scalar) / log(x[i]);
    }
    return 0;
}

int exp_kernel_host_f32(float* out, const float* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = expf(x[i]);
    }
    return 0;
}

int exp_kernel_host_f64(double* out, const double* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = exp(x[i]);
    }
    return 0;
}

int log_kernel_host_f32(float* out, const float* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = logf(x[i]);
    }
    return 0;
}

int log_kernel_host_f64(double* out, const double* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = log(x[i]);
    }
    return 0;
}

int sqrt_kernel_host_f32(float* out, const float* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = sqrtf(x[i]);
    }
    return 0;
}

int sqrt_kernel_host_f64(double* out, const double* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = sqrt(x[i]);
    }
    return 0;
}

int abs_kernel_host_f32(float* out, const float* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = fabsf(x[i]);
    }
    return 0;
}

int abs_kernel_host_f64(double* out, const double* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = fabs(x[i]);
    }
    return 0;
}

int sin_kernel_host_f32(float* out, const float* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = sinf(x[i]);
    }
    return 0;
}

int sin_kernel_host_f64(double* out, const double* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = sin(x[i]);
    }
    return 0;
}

int cos_kernel_host_f32(float* out, const float* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = cosf(x[i]);
    }
    return 0;
}

int cos_kernel_host_f64(double* out, const double* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = cos(x[i]);
    }
    return 0;
}

int tan_kernel_host_f32(float* out, const float* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = tanf(x[i]);
    }
    return 0;
}

int tan_kernel_host_f64(double* out, const double* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = tan(x[i]);
    }
    return 0;
}

int asin_kernel_host_f32(float* out, const float* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = asinf(x[i]);
    }
    return 0;
}

int asin_kernel_host_f64(double* out, const double* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = asin(x[i]);
    }
    return 0;
}

int acos_kernel_host_f32(float* out, const float* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = acosf(x[i]);
    }
    return 0;
}

int acos_kernel_host_f64(double* out, const double* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = acos(x[i]);
    }
    return 0;
}

int atan_kernel_host_f32(float* out, const float* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = atanf(x[i]);
    }
    return 0;
}

int atan_kernel_host_f64(double* out, const double* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = atan(x[i]);
    }
    return 0;
}

int sinh_kernel_host_f32(float* out, const float* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = sinhf(x[i]);
    }
    return 0;
}

int sinh_kernel_host_f64(double* out, const double* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = sinh(x[i]);
    }
    return 0;
}

int cosh_kernel_host_f32(float* out, const float* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = coshf(x[i]);
    }
    return 0;
}

int cosh_kernel_host_f64(double* out, const double* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = cosh(x[i]);
    }
    return 0;
}

int tanh_kernel_host_f32(float* out, const float* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = tanhf(x[i]);
    }
    return 0;
}

int tanh_kernel_host_f64(double* out, const double* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = tanh(x[i]);
    }
    return 0;
}

int relu_kernel_host_f32(float* out, const float* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
    }
    return 0;
}

int relu_kernel_host_f64(double* out, const double* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = (x[i] > 0.0) ? x[i] : 0.0;
    }
    return 0;
}

int sigmoid_kernel_host_f32(float* out, const float* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = 1.0f / (1.0f + expf(-x[i]));
    }
    return 0;
}

int sigmoid_kernel_host_f64(double* out, const double* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = 1.0 / (1.0 + exp(-x[i]));
    }
    return 0;
}

int leaky_relu_kernel_host_f32(float* out, const float* x, float alpha, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = (x[i] > 0.0f) ? x[i] : alpha * x[i];
    }
    return 0;
}

int leaky_relu_kernel_host_f64(double* out, const double* x, double alpha, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = (x[i] > 0.0) ? x[i] : alpha * x[i];
    }
    return 0;
}

int square_kernel_host_f32(float* out, const float* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = x[i] * x[i];
    }
    return 0;
}

int square_kernel_host_f64(double* out, const double* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = x[i] * x[i];
    }
    return 0;
}

int ceil_kernel_host_f32(float* out, const float* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = ceilf(x[i]);
    }
    return 0;
}

int ceil_kernel_host_f64(double* out, const double* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = ceil(x[i]);
    }
    return 0;
}

int floor_kernel_host_f32(float* out, const float* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = floorf(x[i]);
    }
    return 0;
}

int floor_kernel_host_f64(double* out, const double* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = floor(x[i]);
    }
    return 0;
}

int round_kernel_host_f32(float* out, const float* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = roundf(x[i]);
    }
    return 0;
}

int round_kernel_host_f64(double* out, const double* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = round(x[i]);
    }
    return 0;
}

int trunc_kernel_host_f32(float* out, const float* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = truncf(x[i]);
    }
    return 0;
}

int trunc_kernel_host_f64(double* out, const double* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = trunc(x[i]);
    }
    return 0;
}

int rsqrt_kernel_host_f32(float* out, const float* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = 1.0f / sqrtf(x[i]);
    }
    return 0;
}

int rsqrt_kernel_host_f64(double* out, const double* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = 1.0 / sqrt(x[i]);
    }
    return 0;
}

int gelu_kernel_host_f32(float* out, const float* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    for (size_t i = 0; i < size; i++) {
        float x_val = x[i];
        float x_cubed = x_val * x_val * x_val;
        float inner = sqrt_2_over_pi * (x_val + coeff * x_cubed);
        out[i] = 0.5f * x_val * (1.0f + tanhf(inner));
    }
    return 0;
}

int gelu_kernel_host_f64(double* out, const double* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    const double sqrt_2_over_pi = 0.7978845608;
    const double coeff = 0.044715;
    for (size_t i = 0; i < size; i++) {
        double x_val = x[i];
        double x_cubed = x_val * x_val * x_val;
        double inner = sqrt_2_over_pi * (x_val + coeff * x_cubed);
        out[i] = 0.5 * x_val * (1.0 + tanh(inner));
    }
    return 0;
}

int silu_kernel_host_f32(float* out, const float* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = x[i] / (1.0f + expf(-x[i]));
    }
    return 0;
}

int silu_kernel_host_f64(double* out, const double* x, size_t size) {
    if (!out || !x || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = x[i] / (1.0 + exp(-x[i]));
    }
    return 0;
}

float sum_all_kernel_host_f32(const float* x, size_t size) {
    if (!x || size == 0) return 0.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        sum += x[i];
    }
    return sum;
}

double sum_all_kernel_host_f64(const double* x, size_t size) {
    if (!x || size == 0) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < size; i++) {
        sum += x[i];
    }
    return sum;
}

float mean_all_kernel_host_f32(const float* x, size_t size) {
    if (!x || size == 0) return 0.0f;
    return sum_all_kernel_host_f32(x, size) / (float)size;
}

double mean_all_kernel_host_f64(const double* x, size_t size) {
    if (!x || size == 0) return 0.0;
    return sum_all_kernel_host_f64(x, size) / (double)size;
}

int sum_dim_kernel_host_f32(float* out, const float* x, size_t outer_size, size_t dim_size, size_t inner_size) {
    if (!out || !x) return -1;
    for (size_t o = 0; o < outer_size; o++) {
        for (size_t i = 0; i < inner_size; i++) {
            float sum = 0.0f;
            for (size_t d = 0; d < dim_size; d++) {
                sum += x[o * dim_size * inner_size + d * inner_size + i];
            }
            out[o * inner_size + i] = sum;
        }
    }
    return 0;
}

int sum_dim_kernel_host_f64(double* out, const double* x, size_t outer_size, size_t dim_size, size_t inner_size) {
    if (!out || !x) return -1;
    for (size_t o = 0; o < outer_size; o++) {
        for (size_t i = 0; i < inner_size; i++) {
            double sum = 0.0;
            for (size_t d = 0; d < dim_size; d++) {
                sum += x[o * dim_size * inner_size + d * inner_size + i];
            }
            out[o * inner_size + i] = sum;
        }
    }
    return 0;
}

int mean_dim_kernel_host_f32(float* out, const float* x, size_t outer_size, size_t dim_size, size_t inner_size) {
    if (!out || !x) return -1;
    int ret = sum_dim_kernel_host_f32(out, x, outer_size, dim_size, inner_size);
    if (ret != 0) return ret;
    size_t out_size = outer_size * inner_size;
    float divisor = (float)dim_size;
    for (size_t i = 0; i < out_size; i++) {
        out[i] /= divisor;
    }
    return 0;
}

int mean_dim_kernel_host_f64(double* out, const double* x, size_t outer_size, size_t dim_size, size_t inner_size) {
    if (!out || !x) return -1;
    int ret = sum_dim_kernel_host_f64(out, x, outer_size, dim_size, inner_size);
    if (ret != 0) return ret;
    size_t out_size = outer_size * inner_size;
    double divisor = (double)dim_size;
    for (size_t i = 0; i < out_size; i++) {
        out[i] /= divisor;
    }
    return 0;
}

int matmul_kernel_host_f32(float* C, const float* A, const float* B, int m, int k, int n) {
    if (!C || !A || !B || m <= 0 || k <= 0 || n <= 0) return -1;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0f;
            for (int l = 0; l < k; l++) {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
    return 0;
}

int matmul_kernel_host_f64(double* C, const double* A, const double* B, int m, int k, int n) {
    if (!C || !A || !B || m <= 0 || k <= 0 || n <= 0) return -1;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0;
            for (int l = 0; l < k; l++) {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
    return 0;
}

Tensor* cat_kernel_host(Tensor** tensors, int num_tensors, int dim) {
    if (!tensors || num_tensors <= 0 || !tensors[0]) return NULL;

    Tensor* first = tensors[0];
    int ndim = first->ndim;
    DType dtype = first->dtype;

    int* out_shape = (int*)malloc(ndim * sizeof(int));
    if (!out_shape) return NULL;

    for (int d = 0; d < ndim; d++) {
        out_shape[d] = first->shape[d];
    }

    for (int i = 1; i < num_tensors; i++) {
        out_shape[dim] += tensors[i]->shape[dim];
    }

    Tensor* out = zeros_tensor(dtype, -1, ndim, out_shape, NULL);
    if (!out) {
        free(out_shape);
        return NULL;
    }

    int offset_in_cat_dim = 0;

    for (int t = 0; t < num_tensors; t++) {
        Tensor* src = tensors[t];
        size_t elem_size = dtype_size(dtype);

        if (ndim == 1) {
            void* dst_ptr = (char*)out->data + offset_in_cat_dim * elem_size;
            memcpy(dst_ptr, src->data, src->size * elem_size);
        } else {
            int* indices = (int*)calloc(ndim, sizeof(int));
            if (!indices) {
                free_tensor(out);
                free(out_shape);
                return NULL;
            }

            for (size_t i = 0; i < src->size; i++) {
                size_t temp = i;
                for (int d = ndim - 1; d >= 0; d--) {
                    indices[d] = temp % src->shape[d];
                    temp /= src->shape[d];
                }

                indices[dim] += offset_in_cat_dim;

                size_t out_idx = 0;
                for (int d = 0; d < ndim; d++) {
                    out_idx = out_idx * out_shape[d] + indices[d];
                }

                indices[dim] -= offset_in_cat_dim;

                memcpy((char*)out->data + out_idx * elem_size,
                       (char*)src->data + i * elem_size,
                       elem_size);
            }

            free(indices);
        }

        offset_in_cat_dim += src->shape[dim];
    }

    free(out_shape);
    return out;
}

int contiguous_copy_kernel_host_f32(float* out, const float* in, int ndim, int* shape, int* strides) {
    if (!out || !in || !shape || !strides || ndim <= 0) return -1;

    size_t total_size = 1;
    for (int i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }

    int* indices = (int*)calloc(ndim, sizeof(int));
    if (!indices) return -1;

    for (size_t out_idx = 0; out_idx < total_size; out_idx++) {
        size_t in_idx = 0;
        for (int d = 0; d < ndim; d++) {
            in_idx += indices[d] * strides[d];
        }

        out[out_idx] = in[in_idx];

        for (int d = ndim - 1; d >= 0; d--) {
            indices[d]++;
            if (indices[d] < shape[d]) {
                break;
            }
            indices[d] = 0;
        }
    }

    free(indices);
    return 0;
}

int contiguous_copy_kernel_host_f64(double* out, const double* in, int ndim, int* shape, int* strides) {
    if (!out || !in || !shape || !strides || ndim <= 0) return -1;

    size_t total_size = 1;
    for (int i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }

    int* indices = (int*)calloc(ndim, sizeof(int));
    if (!indices) return -1;

    for (size_t out_idx = 0; out_idx < total_size; out_idx++) {
        size_t in_idx = 0;
        for (int d = 0; d < ndim; d++) {
            in_idx += indices[d] * strides[d];
        }

        out[out_idx] = in[in_idx];

        for (int d = ndim - 1; d >= 0; d--) {
            indices[d]++;
            if (indices[d] < shape[d]) {
                break;
            }
            indices[d] = 0;
        }
    }

    free(indices);
    return 0;
}

int repeat_kernel_host_f32(float* out, const float* in, int ndim, int* src_shape, int* repeats) {
    if (!out || !in || !src_shape || !repeats || ndim <= 0) return -1;

    size_t out_size = 1;
    int* out_shape = (int*)malloc(ndim * sizeof(int));
    int* src_strides = (int*)malloc(ndim * sizeof(int));
    if (!out_shape || !src_strides) {
        if (out_shape) free(out_shape);
        if (src_strides) free(src_strides);
        return -1;
    }

    for (int i = 0; i < ndim; i++) {
        out_shape[i] = src_shape[i] * repeats[i];
        out_size *= out_shape[i];
    }

    src_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
    }

    int* out_indices = (int*)calloc(ndim, sizeof(int));
    if (!out_indices) {
        free(out_shape);
        free(src_strides);
        return -1;
    }

    for (size_t out_idx = 0; out_idx < out_size; out_idx++) {
        size_t src_idx = 0;
        for (int d = 0; d < ndim; d++) {
            int src_coord = out_indices[d] % src_shape[d];
            src_idx += src_coord * src_strides[d];
        }

        out[out_idx] = in[src_idx];

        for (int d = ndim - 1; d >= 0; d--) {
            out_indices[d]++;
            if (out_indices[d] < out_shape[d]) {
                break;
            }
            out_indices[d] = 0;
        }
    }

    free(out_shape);
    free(src_strides);
    free(out_indices);
    return 0;
}

int repeat_kernel_host_f64(double* out, const double* in, int ndim, int* src_shape, int* repeats) {
    if (!out || !in || !src_shape || !repeats || ndim <= 0) return -1;

    size_t out_size = 1;
    int* out_shape = (int*)malloc(ndim * sizeof(int));
    int* src_strides = (int*)malloc(ndim * sizeof(int));
    if (!out_shape || !src_strides) {
        if (out_shape) free(out_shape);
        if (src_strides) free(src_strides);
        return -1;
    }

    for (int i = 0; i < ndim; i++) {
        out_shape[i] = src_shape[i] * repeats[i];
        out_size *= out_shape[i];
    }

    src_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
    }

    int* out_indices = (int*)calloc(ndim, sizeof(int));
    if (!out_indices) {
        free(out_shape);
        free(src_strides);
        return -1;
    }

    for (size_t out_idx = 0; out_idx < out_size; out_idx++) {
        size_t src_idx = 0;
        for (int d = 0; d < ndim; d++) {
            int src_coord = out_indices[d] % src_shape[d];
            src_idx += src_coord * src_strides[d];
        }

        out[out_idx] = in[src_idx];

        for (int d = ndim - 1; d >= 0; d--) {
            out_indices[d]++;
            if (out_indices[d] < out_shape[d]) {
                break;
            }
            out_indices[d] = 0;
        }
    }

    free(out_shape);
    free(src_strides);
    free(out_indices);
    return 0;
}

int softmax_kernel_host_f32(float* out, const float* x, size_t outer_size, size_t dim_size, size_t inner_size) {
    for (size_t o = 0; o < outer_size; o++) {
        for (size_t i = 0; i < inner_size; i++) {
            float max_val = -FLT_MAX;
            for (size_t d = 0; d < dim_size; d++) {
                size_t idx = o * dim_size * inner_size + d * inner_size + i;
                if (x[idx] > max_val) max_val = x[idx];
            }
            float sum_exp = 0.0f;
            for (size_t d = 0; d < dim_size; d++) {
                size_t idx = o * dim_size * inner_size + d * inner_size + i;
                out[idx] = expf(x[idx] - max_val);
                sum_exp += out[idx];
            }
            for (size_t d = 0; d < dim_size; d++) {
                size_t idx = o * dim_size * inner_size + d * inner_size + i;
                out[idx] /= sum_exp;
            }
        }
    }
    return 0;
}

int softmax_kernel_host_f64(double* out, const double* x, size_t outer_size, size_t dim_size, size_t inner_size) {
    for (size_t o = 0; o < outer_size; o++) {
        for (size_t i = 0; i < inner_size; i++) {
            double max_val = -DBL_MAX;
            for (size_t d = 0; d < dim_size; d++) {
                size_t idx = o * dim_size * inner_size + d * inner_size + i;
                if (x[idx] > max_val) max_val = x[idx];
            }
            double sum_exp = 0.0;
            for (size_t d = 0; d < dim_size; d++) {
                size_t idx = o * dim_size * inner_size + d * inner_size + i;
                out[idx] = exp(x[idx] - max_val);
                sum_exp += out[idx];
            }
            for (size_t d = 0; d < dim_size; d++) {
                size_t idx = o * dim_size * inner_size + d * inner_size + i;
                out[idx] /= sum_exp;
            }
        }
    }
    return 0;
}

int log_softmax_kernel_host_f32(float* out, const float* x, size_t outer_size, size_t dim_size, size_t inner_size) {
    for (size_t o = 0; o < outer_size; o++) {
        for (size_t i = 0; i < inner_size; i++) {
            float max_val = -FLT_MAX;
            for (size_t d = 0; d < dim_size; d++) {
                size_t idx = o * dim_size * inner_size + d * inner_size + i;
                if (x[idx] > max_val) max_val = x[idx];
            }
            float sum_exp = 0.0f;
            for (size_t d = 0; d < dim_size; d++) {
                size_t idx = o * dim_size * inner_size + d * inner_size + i;
                sum_exp += expf(x[idx] - max_val);
            }
            float log_sum_exp = logf(sum_exp);
            for (size_t d = 0; d < dim_size; d++) {
                size_t idx = o * dim_size * inner_size + d * inner_size + i;
                out[idx] = x[idx] - max_val - log_sum_exp;
            }
        }
    }
    return 0;
}

int log_softmax_kernel_host_f64(double* out, const double* x, size_t outer_size, size_t dim_size, size_t inner_size) {
    for (size_t o = 0; o < outer_size; o++) {
        for (size_t i = 0; i < inner_size; i++) {
            double max_val = -DBL_MAX;
            for (size_t d = 0; d < dim_size; d++) {
                size_t idx = o * dim_size * inner_size + d * inner_size + i;
                if (x[idx] > max_val) max_val = x[idx];
            }
            double sum_exp = 0.0;
            for (size_t d = 0; d < dim_size; d++) {
                size_t idx = o * dim_size * inner_size + d * inner_size + i;
                sum_exp += exp(x[idx] - max_val);
            }
            double log_sum_exp = log(sum_exp);
            for (size_t d = 0; d < dim_size; d++) {
                size_t idx = o * dim_size * inner_size + d * inner_size + i;
                out[idx] = x[idx] - max_val - log_sum_exp;
            }
        }
    }
    return 0;
}

static void linear_to_coords(size_t linear, const int* shape, int ndim, int* coords) {
    for (int d = ndim - 1; d >= 0; d--) {
        coords[d] = linear % shape[d];
        linear /= shape[d];
    }
}

static size_t coords_to_linear(const int* coords, const int* shape, int ndim) {
    size_t idx = 0;
    for (int d = 0; d < ndim; d++) {
        idx = idx * shape[d] + coords[d];
    }
    return idx;
}

int gather_kernel_host_f32(float* out, const float* input, const int* indices,
                           int ndim, const int* input_shape, const int* index_shape,
                           int dim, size_t index_size) {
    if (!out || !input || !indices) return -1;
    int* coords = (int*)malloc(ndim * sizeof(int));
    if (!coords) return -1;

    for (size_t i = 0; i < index_size; i++) {
        linear_to_coords(i, index_shape, ndim, coords);
        int saved = coords[dim];
        coords[dim] = indices[i];
        size_t src_idx = coords_to_linear(coords, input_shape, ndim);
        coords[dim] = saved;
        out[i] = input[src_idx];
    }

    free(coords);
    return 0;
}

int gather_kernel_host_f64(double* out, const double* input, const int* indices,
                           int ndim, const int* input_shape, const int* index_shape,
                           int dim, size_t index_size) {
    if (!out || !input || !indices) return -1;
    int* coords = (int*)malloc(ndim * sizeof(int));
    if (!coords) return -1;

    for (size_t i = 0; i < index_size; i++) {
        linear_to_coords(i, index_shape, ndim, coords);
        int saved = coords[dim];
        coords[dim] = indices[i];
        size_t src_idx = coords_to_linear(coords, input_shape, ndim);
        coords[dim] = saved;
        out[i] = input[src_idx];
    }

    free(coords);
    return 0;
}

int scatter_add_kernel_host_f32(float* out, const float* src, const int* indices,
                                int ndim, const int* out_shape, const int* index_shape,
                                int dim, size_t index_size) {
    if (!out || !src || !indices) return -1;
    int* coords = (int*)malloc(ndim * sizeof(int));
    if (!coords) return -1;

    for (size_t i = 0; i < index_size; i++) {
        linear_to_coords(i, index_shape, ndim, coords);
        int saved = coords[dim];
        coords[dim] = indices[i];
        size_t dst_idx = coords_to_linear(coords, out_shape, ndim);
        coords[dim] = saved;
        out[dst_idx] += src[i];
    }

    free(coords);
    return 0;
}

int scatter_add_kernel_host_f64(double* out, const double* src, const int* indices,
                                int ndim, const int* out_shape, const int* index_shape,
                                int dim, size_t index_size) {
    if (!out || !src || !indices) return -1;
    int* coords = (int*)malloc(ndim * sizeof(int));
    if (!coords) return -1;

    for (size_t i = 0; i < index_size; i++) {
        linear_to_coords(i, index_shape, ndim, coords);
        int saved = coords[dim];
        coords[dim] = indices[i];
        size_t dst_idx = coords_to_linear(coords, out_shape, ndim);
        coords[dim] = saved;
        out[dst_idx] += src[i];
    }

    free(coords);
    return 0;
}

int im2col_kernel_host_f32(float* col, const float* im, int C_in, int H, int W,
                           int kH, int kW, int stride_h, int stride_w,
                           int pad_h, int pad_w, int dilation_h, int dilation_w,
                           int out_H, int out_W) {
    if (!col || !im) return -1;
    int col_cols = out_H * out_W;
    for (int c = 0; c < C_in; c++) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                int row = (c * kH + kh) * kW + kw;
                for (int oh = 0; oh < out_H; oh++) {
                    for (int ow = 0; ow < out_W; ow++) {
                        int h_in = oh * stride_h + kh * dilation_h - pad_h;
                        int w_in = ow * stride_w + kw * dilation_w - pad_w;
                        int col_idx = row * col_cols + oh * out_W + ow;
                        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W)
                            col[col_idx] = im[c * H * W + h_in * W + w_in];
                        else
                            col[col_idx] = 0.0f;
                    }
                }
            }
        }
    }
    return 0;
}

int im2col_kernel_host_f64(double* col, const double* im, int C_in, int H, int W,
                           int kH, int kW, int stride_h, int stride_w,
                           int pad_h, int pad_w, int dilation_h, int dilation_w,
                           int out_H, int out_W) {
    if (!col || !im) return -1;
    int col_cols = out_H * out_W;
    for (int c = 0; c < C_in; c++) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                int row = (c * kH + kh) * kW + kw;
                for (int oh = 0; oh < out_H; oh++) {
                    for (int ow = 0; ow < out_W; ow++) {
                        int h_in = oh * stride_h + kh * dilation_h - pad_h;
                        int w_in = ow * stride_w + kw * dilation_w - pad_w;
                        int col_idx = row * col_cols + oh * out_W + ow;
                        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W)
                            col[col_idx] = im[c * H * W + h_in * W + w_in];
                        else
                            col[col_idx] = 0.0;
                    }
                }
            }
        }
    }
    return 0;
}

int col2im_kernel_host_f32(float* im, const float* col, int C_in, int H, int W,
                           int kH, int kW, int stride_h, int stride_w,
                           int pad_h, int pad_w, int dilation_h, int dilation_w,
                           int out_H, int out_W) {
    if (!im || !col) return -1;
    int col_cols = out_H * out_W;
    for (int c = 0; c < C_in; c++) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                int row = (c * kH + kh) * kW + kw;
                for (int oh = 0; oh < out_H; oh++) {
                    for (int ow = 0; ow < out_W; ow++) {
                        int h_in = oh * stride_h + kh * dilation_h - pad_h;
                        int w_in = ow * stride_w + kw * dilation_w - pad_w;
                        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W)
                            im[c * H * W + h_in * W + w_in] += col[row * col_cols + oh * out_W + ow];
                    }
                }
            }
        }
    }
    return 0;
}

int col2im_kernel_host_f64(double* im, const double* col, int C_in, int H, int W,
                           int kH, int kW, int stride_h, int stride_w,
                           int pad_h, int pad_w, int dilation_h, int dilation_w,
                           int out_H, int out_W) {
    if (!im || !col) return -1;
    int col_cols = out_H * out_W;
    for (int c = 0; c < C_in; c++) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                int row = (c * kH + kh) * kW + kw;
                for (int oh = 0; oh < out_H; oh++) {
                    for (int ow = 0; ow < out_W; ow++) {
                        int h_in = oh * stride_h + kh * dilation_h - pad_h;
                        int w_in = ow * stride_w + kw * dilation_w - pad_w;
                        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W)
                            im[c * H * W + h_in * W + w_in] += col[row * col_cols + oh * out_W + ow];
                    }
                }
            }
        }
    }
    return 0;
}

int maxpool2d_kernel_host_f32(float* out, int* max_indices, const float* input,
                              int N, int C, int H, int W,
                              int kH, int kW, int stride_h, int stride_w,
                              int pad_h, int pad_w, int out_H, int out_W) {
    if (!out || !max_indices || !input) return -1;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < out_H; oh++) {
                for (int ow = 0; ow < out_W; ow++) {
                    float max_val = -FLT_MAX;
                    int max_idx = -1;
                    for (int kh_i = 0; kh_i < kH; kh_i++) {
                        for (int kw_i = 0; kw_i < kW; kw_i++) {
                            int h_in = oh * stride_h + kh_i - pad_h;
                            int w_in = ow * stride_w + kw_i - pad_w;
                            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                int idx = ((n * C + c) * H + h_in) * W + w_in;
                                if (input[idx] > max_val) { max_val = input[idx]; max_idx = idx; }
                            }
                        }
                    }
                    int out_idx = ((n * C + c) * out_H + oh) * out_W + ow;
                    out[out_idx] = max_val;
                    max_indices[out_idx] = max_idx;
                }
            }
        }
    }
    return 0;
}

int maxpool2d_kernel_host_f64(double* out, int* max_indices, const double* input,
                              int N, int C, int H, int W,
                              int kH, int kW, int stride_h, int stride_w,
                              int pad_h, int pad_w, int out_H, int out_W) {
    if (!out || !max_indices || !input) return -1;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < out_H; oh++) {
                for (int ow = 0; ow < out_W; ow++) {
                    double max_val = -DBL_MAX;
                    int max_idx = -1;
                    for (int kh_i = 0; kh_i < kH; kh_i++) {
                        for (int kw_i = 0; kw_i < kW; kw_i++) {
                            int h_in = oh * stride_h + kh_i - pad_h;
                            int w_in = ow * stride_w + kw_i - pad_w;
                            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                int idx = ((n * C + c) * H + h_in) * W + w_in;
                                if (input[idx] > max_val) { max_val = input[idx]; max_idx = idx; }
                            }
                        }
                    }
                    int out_idx = ((n * C + c) * out_H + oh) * out_W + ow;
                    out[out_idx] = max_val;
                    max_indices[out_idx] = max_idx;
                }
            }
        }
    }
    return 0;
}

int avgpool2d_kernel_host_f32(float* out, const float* input,
                              int N, int C, int H, int W,
                              int kH, int kW, int stride_h, int stride_w,
                              int pad_h, int pad_w, int out_H, int out_W) {
    if (!out || !input) return -1;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < out_H; oh++) {
                for (int ow = 0; ow < out_W; ow++) {
                    float sum = 0.0f;
                    int count = 0;
                    for (int kh_i = 0; kh_i < kH; kh_i++) {
                        for (int kw_i = 0; kw_i < kW; kw_i++) {
                            int h_in = oh * stride_h + kh_i - pad_h;
                            int w_in = ow * stride_w + kw_i - pad_w;
                            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                sum += input[((n * C + c) * H + h_in) * W + w_in];
                                count++;
                            }
                        }
                    }
                    out[((n * C + c) * out_H + oh) * out_W + ow] = (count > 0) ? sum / (float)count : 0.0f;
                }
            }
        }
    }
    return 0;
}

int avgpool2d_kernel_host_f64(double* out, const double* input,
                              int N, int C, int H, int W,
                              int kH, int kW, int stride_h, int stride_w,
                              int pad_h, int pad_w, int out_H, int out_W) {
    if (!out || !input) return -1;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < out_H; oh++) {
                for (int ow = 0; ow < out_W; ow++) {
                    double sum = 0.0;
                    int count = 0;
                    for (int kh_i = 0; kh_i < kH; kh_i++) {
                        for (int kw_i = 0; kw_i < kW; kw_i++) {
                            int h_in = oh * stride_h + kh_i - pad_h;
                            int w_in = ow * stride_w + kw_i - pad_w;
                            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                sum += input[((n * C + c) * H + h_in) * W + w_in];
                                count++;
                            }
                        }
                    }
                    out[((n * C + c) * out_H + oh) * out_W + ow] = (count > 0) ? sum / (double)count : 0.0;
                }
            }
        }
    }
    return 0;
}

int dropout_kernel_host_f32(float* out, float* mask, const float* x, size_t size, float p) {
    if (!out || !mask || !x) return -1;
    float scale = 1.0f / (1.0f - p);
    rand_fill_host_f32(mask, size);
    for (size_t i = 0; i < size; i++) {
        if (mask[i] >= p) {
            mask[i] = 1.0f;
            out[i] = x[i] * scale;
        } else {
            mask[i] = 0.0f;
            out[i] = 0.0f;
        }
    }
    return 0;
}

int dropout_kernel_host_f64(double* out, double* mask, const double* x, size_t size, float p) {
    if (!out || !mask || !x) return -1;
    double scale = 1.0 / (1.0 - (double)p);
    rand_fill_host_f64(mask, size);
    for (size_t i = 0; i < size; i++) {
        if (mask[i] >= (double)p) {
            mask[i] = 1.0;
            out[i] = x[i] * scale;
        } else {
            mask[i] = 0.0;
            out[i] = 0.0;
        }
    }
    return 0;
}
