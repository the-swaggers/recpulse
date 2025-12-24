#include "functional.h"
#include <math.h>

int add_kernel_host_f32(float* out, const float* a, const float* b, size_t size) {
    if (!out || !a || !b || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = a[i] + b[i];
    }
    return 0;
}

int add_kernel_host_f64(double* out, const double* a, const double* b, size_t size) {
    if (!out || !a || !b || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = a[i] + b[i];
    }
    return 0;
}

int sub_kernel_host_f32(float* out, const float* a, const float* b, size_t size) {
    if (!out || !a || !b || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = a[i] - b[i];
    }
    return 0;
}

int sub_kernel_host_f64(double* out, const double* a, const double* b, size_t size) {
    if (!out || !a || !b || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = a[i] - b[i];
    }
    return 0;
}

int mul_kernel_host_f32(float* out, const float* a, const float* b, size_t size) {
    if (!out || !a || !b || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = a[i] * b[i];
    }
    return 0;
}

int mul_kernel_host_f64(double* out, const double* a, const double* b, size_t size) {
    if (!out || !a || !b || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = a[i] * b[i];
    }
    return 0;
}

int div_kernel_host_f32(float* out, const float* a, const float* b, size_t size) {
    if (!out || !a || !b || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = a[i] / b[i];
    }
    return 0;
}

int div_kernel_host_f64(double* out, const double* a, const double* b, size_t size) {
    if (!out || !a || !b || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = a[i] / b[i];
    }
    return 0;
}

int pow_kernel_host_f32(float* out, const float* a, const float* b, size_t size) {
    if (!out || !a || !b || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = powf(a[i], b[i]);
    }
    return 0;
}

int pow_kernel_host_f64(double* out, const double* a, const double* b, size_t size) {
    if (!out || !a || !b || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = pow(a[i], b[i]);
    }
    return 0;
}

int logb_kernel_host_f32(float* out, const float* a, const float* b, size_t size) {
    if (!out || !a || !b || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = logf(a[i]) / logf(b[i]);
    }
    return 0;
}

int logb_kernel_host_f64(double* out, const double* a, const double* b, size_t size) {
    if (!out || !a || !b || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = log(a[i]) / log(b[i]);
    }
    return 0;
}

int add_scalar_kernel_host_f32(float* out, const float* a, float scalar, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = a[i] + scalar;
    }
    return 0;
}

int add_scalar_kernel_host_f64(double* out, const double* a, double scalar, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = a[i] + scalar;
    }
    return 0;
}

int sub_scalar_kernel_host_f32(float* out, const float* a, float scalar, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = a[i] - scalar;
    }
    return 0;
}

int sub_scalar_kernel_host_f64(double* out, const double* a, double scalar, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = a[i] - scalar;
    }
    return 0;
}

int mul_scalar_kernel_host_f32(float* out, const float* a, float scalar, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = a[i] * scalar;
    }
    return 0;
}

int mul_scalar_kernel_host_f64(double* out, const double* a, double scalar, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = a[i] * scalar;
    }
    return 0;
}

int div_scalar_kernel_host_f32(float* out, const float* a, float scalar, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = a[i] / scalar;
    }
    return 0;
}

int div_scalar_kernel_host_f64(double* out, const double* a, double scalar, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = a[i] / scalar;
    }
    return 0;
}

int pow_scalar_kernel_host_f32(float* out, const float* a, float scalar, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = powf(a[i], scalar);
    }
    return 0;
}

int pow_scalar_kernel_host_f64(double* out, const double* a, double scalar, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = pow(a[i], scalar);
    }
    return 0;
}

int rsub_scalar_kernel_host_f32(float* out, float scalar, const float* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = scalar - a[i];
    }
    return 0;
}

int rsub_scalar_kernel_host_f64(double* out, double scalar, const double* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = scalar - a[i];
    }
    return 0;
}

int rdiv_scalar_kernel_host_f32(float* out, float scalar, const float* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = scalar / a[i];
    }
    return 0;
}

int rdiv_scalar_kernel_host_f64(double* out, double scalar, const double* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = scalar / a[i];
    }
    return 0;
}

int rpow_scalar_kernel_host_f32(float* out, float scalar, const float* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = powf(scalar, a[i]);
    }
    return 0;
}

int rpow_scalar_kernel_host_f64(double* out, double scalar, const double* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = pow(scalar, a[i]);
    }
    return 0;
}

int logb_scalar_kernel_host_f32(float* out, const float* a, float scalar, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = logf(a[i]) / logf(scalar);
    }
    return 0;
}

int logb_scalar_kernel_host_f64(double* out, const double* a, double scalar, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = log(a[i]) / log(scalar);
    }
    return 0;
}

int rlogb_scalar_kernel_host_f32(float* out, float scalar, const float* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = logf(scalar) / logf(a[i]);
    }
    return 0;
}

int rlogb_scalar_kernel_host_f64(double* out, double scalar, const double* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = log(scalar) / log(a[i]);
    }
    return 0;
}

int exp_kernel_host_f32(float* out, const float* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = expf(a[i]);
    }
    return 0;
}

int exp_kernel_host_f64(double* out, const double* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = exp(a[i]);
    }
    return 0;
}

int log_kernel_host_f32(float* out, const float* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = logf(a[i]);
    }
    return 0;
}

int log_kernel_host_f64(double* out, const double* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = log(a[i]);
    }
    return 0;
}

int sqrt_kernel_host_f32(float* out, const float* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = sqrtf(a[i]);
    }
    return 0;
}

int sqrt_kernel_host_f64(double* out, const double* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = sqrt(a[i]);
    }
    return 0;
}

int abs_kernel_host_f32(float* out, const float* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = fabsf(a[i]);
    }
    return 0;
}

int abs_kernel_host_f64(double* out, const double* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = fabs(a[i]);
    }
    return 0;
}

int sin_kernel_host_f32(float* out, const float* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = sinf(a[i]);
    }
    return 0;
}

int sin_kernel_host_f64(double* out, const double* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = sin(a[i]);
    }
    return 0;
}

int cos_kernel_host_f32(float* out, const float* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = cosf(a[i]);
    }
    return 0;
}

int cos_kernel_host_f64(double* out, const double* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = cos(a[i]);
    }
    return 0;
}

int tan_kernel_host_f32(float* out, const float* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = tanf(a[i]);
    }
    return 0;
}

int tan_kernel_host_f64(double* out, const double* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = tan(a[i]);
    }
    return 0;
}

int asin_kernel_host_f32(float* out, const float* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = asinf(a[i]);
    }
    return 0;
}

int asin_kernel_host_f64(double* out, const double* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = asin(a[i]);
    }
    return 0;
}

int acos_kernel_host_f32(float* out, const float* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = acosf(a[i]);
    }
    return 0;
}

int acos_kernel_host_f64(double* out, const double* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = acos(a[i]);
    }
    return 0;
}

int atan_kernel_host_f32(float* out, const float* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = atanf(a[i]);
    }
    return 0;
}

int atan_kernel_host_f64(double* out, const double* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = atan(a[i]);
    }
    return 0;
}

int sinh_kernel_host_f32(float* out, const float* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = sinhf(a[i]);
    }
    return 0;
}

int sinh_kernel_host_f64(double* out, const double* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = sinh(a[i]);
    }
    return 0;
}

int cosh_kernel_host_f32(float* out, const float* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = coshf(a[i]);
    }
    return 0;
}

int cosh_kernel_host_f64(double* out, const double* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = cosh(a[i]);
    }
    return 0;
}

int tanh_kernel_host_f32(float* out, const float* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = tanhf(a[i]);
    }
    return 0;
}

int tanh_kernel_host_f64(double* out, const double* a, size_t size) {
    if (!out || !a || size == 0) return -1;
    for (size_t i = 0; i < size; i++) {
        out[i] = tanh(a[i]);
    }
    return 0;
}

float sum_all_kernel_host_f32(const float* a, size_t size) {
    if (!a || size == 0) return 0.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        sum += a[i];
    }
    return sum;
}

double sum_all_kernel_host_f64(const double* a, size_t size) {
    if (!a || size == 0) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < size; i++) {
        sum += a[i];
    }
    return sum;
}

float mean_all_kernel_host_f32(const float* a, size_t size) {
    if (!a || size == 0) return 0.0f;
    return sum_all_kernel_host_f32(a, size) / (float)size;
}

double mean_all_kernel_host_f64(const double* a, size_t size) {
    if (!a || size == 0) return 0.0;
    return sum_all_kernel_host_f64(a, size) / (double)size;
}
