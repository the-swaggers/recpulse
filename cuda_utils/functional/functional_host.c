#include "functional.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

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
        size_t elem_size = (dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);

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

                if (dtype == DTYPE_FLOAT32) {
                    ((float*)out->data)[out_idx] = ((float*)src->data)[i];
                } else {
                    ((double*)out->data)[out_idx] = ((double*)src->data)[i];
                }
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
