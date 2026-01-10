#ifndef FUNCTIONAL_H
#define FUNCTIONAL_H

#include <stddef.h>
#include "../core/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

int rp_add(void* out, const void* x1, const void* x2, size_t size, DType dtype, int device_id);
int rp_sub(void* out, const void* x1, const void* x2, size_t size, DType dtype, int device_id);
int rp_mul(void* out, const void* x1, const void* x2, size_t size, DType dtype, int device_id);
int rp_divide(void* out, const void* x1, const void* x2, size_t size, DType dtype, int device_id);
int rp_power(void* out, const void* x1, const void* x2, size_t size, DType dtype, int device_id);
int rp_logb(void* out, const void* x1, const void* x2, size_t size, DType dtype, int device_id); // out[i] = log_b(a) = log(a[i]) / log(b[i])

int rp_add_scalar(void* out, const void* x, const void* scalar, size_t size, DType dtype, int device_id);
int rp_sub_scalar(void* out, const void* x, const void* scalar, size_t size, DType dtype, int device_id);
int rp_mul_scalar(void* out, const void* x, const void* scalar, size_t size, DType dtype, int device_id);
int rp_div_scalar(void* out, const void* x, const void* scalar, size_t size, DType dtype, int device_id);
int rp_pow_scalar(void* out, const void* x, const void* scalar, size_t size, DType dtype, int device_id);
int rp_rsub_scalar(void* out, const void* scalar, const void* x, size_t size, DType dtype, int device_id);
int rp_rdiv_scalar(void* out, const void* scalar, const void* x, size_t size, DType dtype, int device_id);
int rp_rpow_scalar(void* out, const void* scalar, const void* x, size_t size, DType dtype, int device_id);
int rp_logb_scalar(void* out, const void* x, const void* scalar, size_t size, DType dtype, int device_id);
int rp_rlogb_scalar(void* out, const void* scalar, const void* x, size_t size, DType dtype, int device_id);

int rp_exp(void* out, const void* x, size_t size, DType dtype, int device_id);
int rp_log(void* out, const void* x, size_t size, DType dtype, int device_id);
int rp_sqrt(void* out, const void* x, size_t size, DType dtype, int device_id);
int rp_abs(void* out, const void* x, size_t size, DType dtype, int device_id);
int rp_sin(void* out, const void* x, size_t size, DType dtype, int device_id);
int rp_cos(void* out, const void* x, size_t size, DType dtype, int device_id);
int rp_tan(void* out, const void* x, size_t size, DType dtype, int device_id);
int rp_asin(void* out, const void* x, size_t size, DType dtype, int device_id);
int rp_acos(void* out, const void* x, size_t size, DType dtype, int device_id);
int rp_atan(void* out, const void* x, size_t size, DType dtype, int device_id);
int rp_sinh(void* out, const void* x, size_t size, DType dtype, int device_id);
int rp_cosh(void* out, const void* x, size_t size, DType dtype, int device_id);
int rp_tanh(void* out, const void* x, size_t size, DType dtype, int device_id);
int rp_relu(void* out, const void* x, size_t size, DType dtype, int device_id);
int rp_sigmoid(void* out, const void* x, size_t size, DType dtype, int device_id);
int rp_leaky_relu(void* out, const void* x, const void* alpha, size_t size, DType dtype, int device_id);
int rp_square(void* out, const void* x, size_t size, DType dtype, int device_id);
int rp_ceil(void* out, const void* x, size_t size, DType dtype, int device_id);
int rp_floor(void* out, const void* x, size_t size, DType dtype, int device_id);
int rp_round(void* out, const void* x, size_t size, DType dtype, int device_id);
int rp_trunc(void* out, const void* x, size_t size, DType dtype, int device_id);
int rp_rsqrt(void* out, const void* x, size_t size, DType dtype, int device_id);
int rp_gelu(void* out, const void* x, size_t size, DType dtype, int device_id);
int rp_silu(void* out, const void* x, size_t size, DType dtype, int device_id);

int rp_sum_all(void* out, const void* x, size_t size, DType dtype, int device_id);
int rp_mean_all(void* out, const void* x, size_t size, DType dtype, int device_id);

int rp_matmul(void* C, const void* A, const void* B, int m, int k, int n, DType dtype, int device_id);

Tensor* rp_cat(Tensor** tensors, int num_tensors, int dim);

int add_kernel_host_f32(float* out, const float* x1, const float* x2, size_t size);
int sub_kernel_host_f32(float* out, const float* x1, const float* x2, size_t size);
int mul_kernel_host_f32(float* out, const float* x1, const float* x2, size_t size);
int div_kernel_host_f32(float* out, const float* x1, const float* x2, size_t size);
int pow_kernel_host_f32(float* out, const float* x1, const float* x2, size_t size);
int logb_kernel_host_f32(float* out, const float* x1, const float* x2, size_t size);

int add_kernel_host_f64(double* out, const double* x1, const double* x2, size_t size);
int sub_kernel_host_f64(double* out, const double* x1, const double* x2, size_t size);
int mul_kernel_host_f64(double* out, const double* x1, const double* x2, size_t size);
int div_kernel_host_f64(double* out, const double* x1, const double* x2, size_t size);
int pow_kernel_host_f64(double* out, const double* x1, const double* x2, size_t size);
int logb_kernel_host_f64(double* out, const double* x1, const double* x2, size_t size);

int add_scalar_kernel_host_f32(float* out, const float* x, float scalar, size_t size);
int sub_scalar_kernel_host_f32(float* out, const float* x, float scalar, size_t size);
int mul_scalar_kernel_host_f32(float* out, const float* x, float scalar, size_t size);
int div_scalar_kernel_host_f32(float* out, const float* x, float scalar, size_t size);
int pow_scalar_kernel_host_f32(float* out, const float* x, float scalar, size_t size);
int rsub_scalar_kernel_host_f32(float* out, float scalar, const float* x, size_t size);
int rdiv_scalar_kernel_host_f32(float* out, float scalar, const float* x, size_t size);
int rpow_scalar_kernel_host_f32(float* out, float scalar, const float* x, size_t size);
int logb_scalar_kernel_host_f32(float* out, const float* x, float scalar, size_t size);
int rlogb_scalar_kernel_host_f32(float* out, float scalar, const float* x, size_t size);

int add_scalar_kernel_host_f64(double* out, const double* x, double scalar, size_t size);
int sub_scalar_kernel_host_f64(double* out, const double* x, double scalar, size_t size);
int mul_scalar_kernel_host_f64(double* out, const double* x, double scalar, size_t size);
int div_scalar_kernel_host_f64(double* out, const double* x, double scalar, size_t size);
int pow_scalar_kernel_host_f64(double* out, const double* x, double scalar, size_t size);
int rsub_scalar_kernel_host_f64(double* out, double scalar, const double* x, size_t size);
int rdiv_scalar_kernel_host_f64(double* out, double scalar, const double* x, size_t size);
int rpow_scalar_kernel_host_f64(double* out, double scalar, const double* x, size_t size);
int logb_scalar_kernel_host_f64(double* out, const double* x, double scalar, size_t size);
int rlogb_scalar_kernel_host_f64(double* out, double scalar, const double* x, size_t size);

int exp_kernel_host_f32(float* out, const float* x, size_t size);
int exp_kernel_host_f64(double* out, const double* x, size_t size);
int log_kernel_host_f32(float* out, const float* x, size_t size);
int log_kernel_host_f64(double* out, const double* x, size_t size);
int sqrt_kernel_host_f32(float* out, const float* x, size_t size);
int sqrt_kernel_host_f64(double* out, const double* x, size_t size);
int abs_kernel_host_f32(float* out, const float* x, size_t size);
int abs_kernel_host_f64(double* out, const double* x, size_t size);
int sin_kernel_host_f32(float* out, const float* x, size_t size);
int sin_kernel_host_f64(double* out, const double* x, size_t size);
int cos_kernel_host_f32(float* out, const float* x, size_t size);
int cos_kernel_host_f64(double* out, const double* x, size_t size);
int tan_kernel_host_f32(float* out, const float* x, size_t size);
int tan_kernel_host_f64(double* out, const double* x, size_t size);
int asin_kernel_host_f32(float* out, const float* x, size_t size);
int asin_kernel_host_f64(double* out, const double* x, size_t size);
int acos_kernel_host_f32(float* out, const float* x, size_t size);
int acos_kernel_host_f64(double* out, const double* x, size_t size);
int atan_kernel_host_f32(float* out, const float* x, size_t size);
int atan_kernel_host_f64(double* out, const double* x, size_t size);
int sinh_kernel_host_f32(float* out, const float* x, size_t size);
int sinh_kernel_host_f64(double* out, const double* x, size_t size);
int cosh_kernel_host_f32(float* out, const float* x, size_t size);
int cosh_kernel_host_f64(double* out, const double* x, size_t size);
int tanh_kernel_host_f32(float* out, const float* x, size_t size);
int tanh_kernel_host_f64(double* out, const double* x, size_t size);
int relu_kernel_host_f32(float* out, const float* x, size_t size);
int relu_kernel_host_f64(double* out, const double* x, size_t size);
int sigmoid_kernel_host_f32(float* out, const float* x, size_t size);
int sigmoid_kernel_host_f64(double* out, const double* x, size_t size);
int leaky_relu_kernel_host_f32(float* out, const float* x, float alpha, size_t size);
int leaky_relu_kernel_host_f64(double* out, const double* x, double alpha, size_t size);
int square_kernel_host_f32(float* out, const float* x, size_t size);
int square_kernel_host_f64(double* out, const double* x, size_t size);
int ceil_kernel_host_f32(float* out, const float* x, size_t size);
int ceil_kernel_host_f64(double* out, const double* x, size_t size);
int floor_kernel_host_f32(float* out, const float* x, size_t size);
int floor_kernel_host_f64(double* out, const double* x, size_t size);
int round_kernel_host_f32(float* out, const float* x, size_t size);
int round_kernel_host_f64(double* out, const double* x, size_t size);
int trunc_kernel_host_f32(float* out, const float* x, size_t size);
int trunc_kernel_host_f64(double* out, const double* x, size_t size);
int rsqrt_kernel_host_f32(float* out, const float* x, size_t size);
int rsqrt_kernel_host_f64(double* out, const double* x, size_t size);
int gelu_kernel_host_f32(float* out, const float* x, size_t size);
int gelu_kernel_host_f64(double* out, const double* x, size_t size);
int silu_kernel_host_f32(float* out, const float* x, size_t size);
int silu_kernel_host_f64(double* out, const double* x, size_t size);

float sum_all_kernel_host_f32(const float* x, size_t size);
double sum_all_kernel_host_f64(const double* x, size_t size);
float mean_all_kernel_host_f32(const float* x, size_t size);
double mean_all_kernel_host_f64(const double* x, size_t size);

int matmul_kernel_host_f32(float* C, const float* A, const float* B, int m, int k, int n);
int matmul_kernel_host_f64(double* C, const double* A, const double* B, int m, int k, int n);

Tensor* cat_kernel_host(Tensor** tensors, int num_tensors, int dim);

int add_kernel_device(void* out, const void* x1, const void* x2, size_t size, DType dtype);
int sub_kernel_device(void* out, const void* x1, const void* x2, size_t size, DType dtype);
int mul_kernel_device(void* out, const void* x1, const void* x2, size_t size, DType dtype);
int div_kernel_device(void* out, const void* x1, const void* x2, size_t size, DType dtype);
int pow_kernel_device(void* out, const void* x1, const void* x2, size_t size, DType dtype);
int logb_kernel_device(void* out, const void* x1, const void* x2, size_t size, DType dtype);

int add_scalar_kernel_device(void* out, const void* x, const void* scalar, size_t size, DType dtype);
int sub_scalar_kernel_device(void* out, const void* x, const void* scalar, size_t size, DType dtype);
int mul_scalar_kernel_device(void* out, const void* x, const void* scalar, size_t size, DType dtype);
int div_scalar_kernel_device(void* out, const void* x, const void* scalar, size_t size, DType dtype);
int pow_scalar_kernel_device(void* out, const void* x, const void* scalar, size_t size, DType dtype);
int rsub_scalar_kernel_device(void* out, const void* scalar, const void* x, size_t size, DType dtype);
int rdiv_scalar_kernel_device(void* out, const void* scalar, const void* x, size_t size, DType dtype);
int rpow_scalar_kernel_device(void* out, const void* scalar, const void* x, size_t size, DType dtype);
int logb_scalar_kernel_device(void* out, const void* x, const void* scalar, size_t size, DType dtype);
int rlogb_scalar_kernel_device(void* out, const void* scalar, const void* x, size_t size, DType dtype);

int exp_kernel_device(void* out, const void* x, size_t size, DType dtype);
int log_kernel_device(void* out, const void* x, size_t size, DType dtype);
int sqrt_kernel_device(void* out, const void* x, size_t size, DType dtype);
int abs_kernel_device(void* out, const void* x, size_t size, DType dtype);
int sin_kernel_device(void* out, const void* x, size_t size, DType dtype);
int cos_kernel_device(void* out, const void* x, size_t size, DType dtype);
int tan_kernel_device(void* out, const void* x, size_t size, DType dtype);
int asin_kernel_device(void* out, const void* x, size_t size, DType dtype);
int acos_kernel_device(void* out, const void* x, size_t size, DType dtype);
int atan_kernel_device(void* out, const void* x, size_t size, DType dtype);
int sinh_kernel_device(void* out, const void* x, size_t size, DType dtype);
int cosh_kernel_device(void* out, const void* x, size_t size, DType dtype);
int tanh_kernel_device(void* out, const void* x, size_t size, DType dtype);
int relu_kernel_device(void* out, const void* x, size_t size, DType dtype);
int sigmoid_kernel_device(void* out, const void* x, size_t size, DType dtype);
int leaky_relu_kernel_device(void* out, const void* x, const void* alpha, size_t size, DType dtype);
int square_kernel_device(void* out, const void* x, size_t size, DType dtype);
int ceil_kernel_device(void* out, const void* x, size_t size, DType dtype);
int floor_kernel_device(void* out, const void* x, size_t size, DType dtype);
int round_kernel_device(void* out, const void* x, size_t size, DType dtype);
int trunc_kernel_device(void* out, const void* x, size_t size, DType dtype);
int rsqrt_kernel_device(void* out, const void* x, size_t size, DType dtype);
int gelu_kernel_device(void* out, const void* x, size_t size, DType dtype);
int silu_kernel_device(void* out, const void* x, size_t size, DType dtype);

int sum_all_kernel_device(void* out, const void* x, size_t size, DType dtype);
int mean_all_kernel_device(void* out, const void* x, size_t size, DType dtype);

int matmul_kernel_device(void* C, const void* A, const void* B, int m, int k, int n, DType dtype);

Tensor* cat_kernel_device(Tensor** tensors, int num_tensors, int dim);

#ifdef __cplusplus
}
#endif

#endif 
