#ifndef FUNCTIONAL_H
#define FUNCTIONAL_H

#include <stddef.h>
#include "../core/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

int add(void* out, const void* a, const void* b, size_t size, DType dtype, int device_id);
int sub(void* out, const void* a, const void* b, size_t size, DType dtype, int device_id);
int mul(void* out, const void* a, const void* b, size_t size, DType dtype, int device_id);
int divide(void* out, const void* a, const void* b, size_t size, DType dtype, int device_id);
int power(void* out, const void* a, const void* b, size_t size, DType dtype, int device_id);

int add_kernel_cpu_f32(float* out, const float* a, const float* b, size_t size);
int sub_kernel_cpu_f32(float* out, const float* a, const float* b, size_t size);
int mul_kernel_cpu_f32(float* out, const float* a, const float* b, size_t size);
int div_kernel_cpu_f32(float* out, const float* a, const float* b, size_t size);
int pow_kernel_cpu_f32(float* out, const float* a, const float* b, size_t size);

int add_kernel_cpu_f64(double* out, const double* a, const double* b, size_t size);
int sub_kernel_cpu_f64(double* out, const double* a, const double* b, size_t size);
int mul_kernel_cpu_f64(double* out, const double* a, const double* b, size_t size);
int div_kernel_cpu_f64(double* out, const double* a, const double* b, size_t size);
int pow_kernel_cpu_f64(double* out, const double* a, const double* b, size_t size);

int add_kernel_cuda(void* out, const void* a, const void* b, size_t size, DType dtype);
int sub_kernel_cuda(void* out, const void* a, const void* b, size_t size, DType dtype);
int mul_kernel_cuda(void* out, const void* a, const void* b, size_t size, DType dtype);
int div_kernel_cuda(void* out, const void* a, const void* b, size_t size, DType dtype);
int pow_kernel_cuda(void* out, const void* a, const void* b, size_t size, DType dtype);

#ifdef __cplusplus
}
#endif

#endif 
