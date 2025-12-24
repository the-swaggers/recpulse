#ifndef FUNCTIONAL_H
#define FUNCTIONAL_H

#include <stddef.h>
#include "../core/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

int rp_add(void* out, const void* a, const void* b, size_t size, DType dtype, int device_id);
int rp_sub(void* out, const void* a, const void* b, size_t size, DType dtype, int device_id);
int rp_mul(void* out, const void* a, const void* b, size_t size, DType dtype, int device_id);
int rp_divide(void* out, const void* a, const void* b, size_t size, DType dtype, int device_id);
int rp_power(void* out, const void* a, const void* b, size_t size, DType dtype, int device_id);
int rp_logb(void* out, const void* a, const void* b, size_t size, DType dtype, int device_id); // out[i] = log_b(a) = log(a[i]) / log(b[i])

int rp_add_scalar(void* out, const void* a, const void* scalar, size_t size, DType dtype, int device_id);
int rp_sub_scalar(void* out, const void* a, const void* scalar, size_t size, DType dtype, int device_id);
int rp_mul_scalar(void* out, const void* a, const void* scalar, size_t size, DType dtype, int device_id);
int rp_div_scalar(void* out, const void* a, const void* scalar, size_t size, DType dtype, int device_id);
int rp_pow_scalar(void* out, const void* a, const void* scalar, size_t size, DType dtype, int device_id);
int rp_rsub_scalar(void* out, const void* scalar, const void* a, size_t size, DType dtype, int device_id);
int rp_rdiv_scalar(void* out, const void* scalar, const void* a, size_t size, DType dtype, int device_id);
int rp_rpow_scalar(void* out, const void* scalar, const void* a, size_t size, DType dtype, int device_id);
int rp_logb_scalar(void* out, const void* a, const void* scalar, size_t size, DType dtype, int device_id);
int rp_rlogb_scalar(void* out, const void* scalar, const void* a, size_t size, DType dtype, int device_id);

int rp_exp(void* out, const void* a, size_t size, DType dtype, int device_id);
int rp_log(void* out, const void* a, size_t size, DType dtype, int device_id);
int rp_sqrt(void* out, const void* a, size_t size, DType dtype, int device_id);
int rp_abs(void* out, const void* a, size_t size, DType dtype, int device_id);
int rp_sin(void* out, const void* a, size_t size, DType dtype, int device_id);
int rp_cos(void* out, const void* a, size_t size, DType dtype, int device_id);
int rp_tan(void* out, const void* a, size_t size, DType dtype, int device_id);
int rp_asin(void* out, const void* a, size_t size, DType dtype, int device_id);
int rp_acos(void* out, const void* a, size_t size, DType dtype, int device_id);
int rp_atan(void* out, const void* a, size_t size, DType dtype, int device_id);
int rp_sinh(void* out, const void* a, size_t size, DType dtype, int device_id);
int rp_cosh(void* out, const void* a, size_t size, DType dtype, int device_id);
int rp_tanh(void* out, const void* a, size_t size, DType dtype, int device_id);

int rp_sum_all(void* out, const void* a, size_t size, DType dtype, int device_id);
int rp_mean_all(void* out, const void* a, size_t size, DType dtype, int device_id);

int add_kernel_host_f32(float* out, const float* a, const float* b, size_t size);
int sub_kernel_host_f32(float* out, const float* a, const float* b, size_t size);
int mul_kernel_host_f32(float* out, const float* a, const float* b, size_t size);
int div_kernel_host_f32(float* out, const float* a, const float* b, size_t size);
int pow_kernel_host_f32(float* out, const float* a, const float* b, size_t size);
int logb_kernel_host_f32(float* out, const float* a, const float* b, size_t size);

int add_kernel_host_f64(double* out, const double* a, const double* b, size_t size);
int sub_kernel_host_f64(double* out, const double* a, const double* b, size_t size);
int mul_kernel_host_f64(double* out, const double* a, const double* b, size_t size);
int div_kernel_host_f64(double* out, const double* a, const double* b, size_t size);
int pow_kernel_host_f64(double* out, const double* a, const double* b, size_t size);
int logb_kernel_host_f64(double* out, const double* a, const double* b, size_t size);

int add_scalar_kernel_host_f32(float* out, const float* a, float scalar, size_t size);
int sub_scalar_kernel_host_f32(float* out, const float* a, float scalar, size_t size);
int mul_scalar_kernel_host_f32(float* out, const float* a, float scalar, size_t size);
int div_scalar_kernel_host_f32(float* out, const float* a, float scalar, size_t size);
int pow_scalar_kernel_host_f32(float* out, const float* a, float scalar, size_t size);
int rsub_scalar_kernel_host_f32(float* out, float scalar, const float* a, size_t size);
int rdiv_scalar_kernel_host_f32(float* out, float scalar, const float* a, size_t size);
int rpow_scalar_kernel_host_f32(float* out, float scalar, const float* a, size_t size);
int logb_scalar_kernel_host_f32(float* out, const float* a, float scalar, size_t size);
int rlogb_scalar_kernel_host_f32(float* out, float scalar, const float* a, size_t size);

int add_scalar_kernel_host_f64(double* out, const double* a, double scalar, size_t size);
int sub_scalar_kernel_host_f64(double* out, const double* a, double scalar, size_t size);
int mul_scalar_kernel_host_f64(double* out, const double* a, double scalar, size_t size);
int div_scalar_kernel_host_f64(double* out, const double* a, double scalar, size_t size);
int pow_scalar_kernel_host_f64(double* out, const double* a, double scalar, size_t size);
int rsub_scalar_kernel_host_f64(double* out, double scalar, const double* a, size_t size);
int rdiv_scalar_kernel_host_f64(double* out, double scalar, const double* a, size_t size);
int rpow_scalar_kernel_host_f64(double* out, double scalar, const double* a, size_t size);
int logb_scalar_kernel_host_f64(double* out, const double* a, double scalar, size_t size);
int rlogb_scalar_kernel_host_f64(double* out, double scalar, const double* a, size_t size);

int exp_kernel_host_f32(float* out, const float* a, size_t size);
int exp_kernel_host_f64(double* out, const double* a, size_t size);
int log_kernel_host_f32(float* out, const float* a, size_t size);
int log_kernel_host_f64(double* out, const double* a, size_t size);
int sqrt_kernel_host_f32(float* out, const float* a, size_t size);
int sqrt_kernel_host_f64(double* out, const double* a, size_t size);
int abs_kernel_host_f32(float* out, const float* a, size_t size);
int abs_kernel_host_f64(double* out, const double* a, size_t size);
int sin_kernel_host_f32(float* out, const float* a, size_t size);
int sin_kernel_host_f64(double* out, const double* a, size_t size);
int cos_kernel_host_f32(float* out, const float* a, size_t size);
int cos_kernel_host_f64(double* out, const double* a, size_t size);
int tan_kernel_host_f32(float* out, const float* a, size_t size);
int tan_kernel_host_f64(double* out, const double* a, size_t size);
int asin_kernel_host_f32(float* out, const float* a, size_t size);
int asin_kernel_host_f64(double* out, const double* a, size_t size);
int acos_kernel_host_f32(float* out, const float* a, size_t size);
int acos_kernel_host_f64(double* out, const double* a, size_t size);
int atan_kernel_host_f32(float* out, const float* a, size_t size);
int atan_kernel_host_f64(double* out, const double* a, size_t size);
int sinh_kernel_host_f32(float* out, const float* a, size_t size);
int sinh_kernel_host_f64(double* out, const double* a, size_t size);
int cosh_kernel_host_f32(float* out, const float* a, size_t size);
int cosh_kernel_host_f64(double* out, const double* a, size_t size);
int tanh_kernel_host_f32(float* out, const float* a, size_t size);
int tanh_kernel_host_f64(double* out, const double* a, size_t size);

float sum_all_kernel_host_f32(const float* a, size_t size);
double sum_all_kernel_host_f64(const double* a, size_t size);
float mean_all_kernel_host_f32(const float* a, size_t size);
double mean_all_kernel_host_f64(const double* a, size_t size);

int add_kernel_device(void* out, const void* a, const void* b, size_t size, DType dtype);
int sub_kernel_device(void* out, const void* a, const void* b, size_t size, DType dtype);
int mul_kernel_device(void* out, const void* a, const void* b, size_t size, DType dtype);
int div_kernel_device(void* out, const void* a, const void* b, size_t size, DType dtype);
int pow_kernel_device(void* out, const void* a, const void* b, size_t size, DType dtype);
int logb_kernel_device(void* out, const void* a, const void* b, size_t size, DType dtype);

int add_scalar_kernel_device(void* out, const void* a, const void* scalar, size_t size, DType dtype);
int sub_scalar_kernel_device(void* out, const void* a, const void* scalar, size_t size, DType dtype);
int mul_scalar_kernel_device(void* out, const void* a, const void* scalar, size_t size, DType dtype);
int div_scalar_kernel_device(void* out, const void* a, const void* scalar, size_t size, DType dtype);
int pow_scalar_kernel_device(void* out, const void* a, const void* scalar, size_t size, DType dtype);
int rsub_scalar_kernel_device(void* out, const void* scalar, const void* a, size_t size, DType dtype);
int rdiv_scalar_kernel_device(void* out, const void* scalar, const void* a, size_t size, DType dtype);
int rpow_scalar_kernel_device(void* out, const void* scalar, const void* a, size_t size, DType dtype);
int logb_scalar_kernel_device(void* out, const void* a, const void* scalar, size_t size, DType dtype);
int rlogb_scalar_kernel_device(void* out, const void* scalar, const void* a, size_t size, DType dtype);

int exp_kernel_device(void* out, const void* a, size_t size, DType dtype);
int log_kernel_device(void* out, const void* a, size_t size, DType dtype);
int sqrt_kernel_device(void* out, const void* a, size_t size, DType dtype);
int abs_kernel_device(void* out, const void* a, size_t size, DType dtype);
int sin_kernel_device(void* out, const void* a, size_t size, DType dtype);
int cos_kernel_device(void* out, const void* a, size_t size, DType dtype);
int tan_kernel_device(void* out, const void* a, size_t size, DType dtype);
int asin_kernel_device(void* out, const void* a, size_t size, DType dtype);
int acos_kernel_device(void* out, const void* a, size_t size, DType dtype);
int atan_kernel_device(void* out, const void* a, size_t size, DType dtype);
int sinh_kernel_device(void* out, const void* a, size_t size, DType dtype);
int cosh_kernel_device(void* out, const void* a, size_t size, DType dtype);
int tanh_kernel_device(void* out, const void* a, size_t size, DType dtype);

int sum_all_kernel_device(void* out, const void* a, size_t size, DType dtype);
int mean_all_kernel_device(void* out, const void* a, size_t size, DType dtype);

#ifdef __cplusplus
}
#endif

#endif 
