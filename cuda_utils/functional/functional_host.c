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
