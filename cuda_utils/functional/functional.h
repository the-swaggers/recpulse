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

int add_scalar(void* out, const void* a, const void* scalar, size_t size, DType dtype, int device_id);
int sub_scalar(void* out, const void* a, const void* scalar, size_t size, DType dtype, int device_id);
int mul_scalar(void* out, const void* a, const void* scalar, size_t size, DType dtype, int device_id);
int div_scalar(void* out, const void* a, const void* scalar, size_t size, DType dtype, int device_id);
int pow_scalar(void* out, const void* a, const void* scalar, size_t size, DType dtype, int device_id);
int rsub_scalar(void* out, const void* scalar, const void* a, size_t size, DType dtype, int device_id);
int rdiv_scalar(void* out, const void* scalar, const void* a, size_t size, DType dtype, int device_id);
int rpow_scalar(void* out, const void* scalar, const void* a, size_t size, DType dtype, int device_id);

int add_kernel_host_f32(float* out, const float* a, const float* b, size_t size);
int sub_kernel_host_f32(float* out, const float* a, const float* b, size_t size);
int mul_kernel_host_f32(float* out, const float* a, const float* b, size_t size);
int div_kernel_host_f32(float* out, const float* a, const float* b, size_t size);
int pow_kernel_host_f32(float* out, const float* a, const float* b, size_t size);

int add_kernel_host_f64(double* out, const double* a, const double* b, size_t size);
int sub_kernel_host_f64(double* out, const double* a, const double* b, size_t size);
int mul_kernel_host_f64(double* out, const double* a, const double* b, size_t size);
int div_kernel_host_f64(double* out, const double* a, const double* b, size_t size);
int pow_kernel_host_f64(double* out, const double* a, const double* b, size_t size);

int add_scalar_kernel_host_f32(float* out, const float* a, float scalar, size_t size);
int sub_scalar_kernel_host_f32(float* out, const float* a, float scalar, size_t size);
int mul_scalar_kernel_host_f32(float* out, const float* a, float scalar, size_t size);
int div_scalar_kernel_host_f32(float* out, const float* a, float scalar, size_t size);
int pow_scalar_kernel_host_f32(float* out, const float* a, float scalar, size_t size);
int rsub_scalar_kernel_host_f32(float* out, float scalar, const float* a, size_t size);
int rdiv_scalar_kernel_host_f32(float* out, float scalar, const float* a, size_t size);
int rpow_scalar_kernel_host_f32(float* out, float scalar, const float* a, size_t size);

int add_scalar_kernel_host_f64(double* out, const double* a, double scalar, size_t size);
int sub_scalar_kernel_host_f64(double* out, const double* a, double scalar, size_t size);
int mul_scalar_kernel_host_f64(double* out, const double* a, double scalar, size_t size);
int div_scalar_kernel_host_f64(double* out, const double* a, double scalar, size_t size);
int pow_scalar_kernel_host_f64(double* out, const double* a, double scalar, size_t size);
int rsub_scalar_kernel_host_f64(double* out, double scalar, const double* a, size_t size);
int rdiv_scalar_kernel_host_f64(double* out, double scalar, const double* a, size_t size);
int rpow_scalar_kernel_host_f64(double* out, double scalar, const double* a, size_t size);

int add_kernel_device(void* out, const void* a, const void* b, size_t size, DType dtype);
int sub_kernel_device(void* out, const void* a, const void* b, size_t size, DType dtype);
int mul_kernel_device(void* out, const void* a, const void* b, size_t size, DType dtype);
int div_kernel_device(void* out, const void* a, const void* b, size_t size, DType dtype);
int pow_kernel_device(void* out, const void* a, const void* b, size_t size, DType dtype);

int add_scalar_kernel_device(void* out, const void* a, const void* scalar, size_t size, DType dtype);
int sub_scalar_kernel_device(void* out, const void* a, const void* scalar, size_t size, DType dtype);
int mul_scalar_kernel_device(void* out, const void* a, const void* scalar, size_t size, DType dtype);
int div_scalar_kernel_device(void* out, const void* a, const void* scalar, size_t size, DType dtype);
int pow_scalar_kernel_device(void* out, const void* a, const void* scalar, size_t size, DType dtype);
int rsub_scalar_kernel_device(void* out, const void* scalar, const void* a, size_t size, DType dtype);
int rdiv_scalar_kernel_device(void* out, const void* scalar, const void* a, size_t size, DType dtype);
int rpow_scalar_kernel_device(void* out, const void* scalar, const void* a, size_t size, DType dtype);

#ifdef __cplusplus
}
#endif

#endif 
