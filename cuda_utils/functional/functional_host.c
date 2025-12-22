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
