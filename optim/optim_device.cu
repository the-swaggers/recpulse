#include "optim.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>

__device__ inline float optim_h2f(__half v) { return __half2float(v); }
__device__ inline float optim_h2f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ inline void optim_f2h(__half* dst, float v) { *dst = __float2half(v); }
__device__ inline void optim_f2h(__nv_bfloat16* dst, float v) { *dst = __float2bfloat16(v); }

template<typename H>
__global__ void sgd_step_half_kernel(H* param, const H* grad, H* buf,
                                     size_t size, float lr, float momentum, float dampening,
                                     bool nesterov, float weight_decay,
                                     bool has_momentum_buf, bool use_momentum) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float g = optim_h2f(grad[idx]);
    float p = optim_h2f(param[idx]);
    if (weight_decay != 0.0f) g += weight_decay * p;
    if (use_momentum) {
        float b;
        if (!has_momentum_buf) {
            b = g;
        } else {
            b = momentum * optim_h2f(buf[idx]) + (1.0f - dampening) * g;
        }
        optim_f2h(&buf[idx], b);
        if (nesterov) {
            g = g + momentum * b;
        } else {
            g = b;
        }
    }
    optim_f2h(&param[idx], p - lr * g);
}

template<typename H>
__global__ void adam_step_half_kernel(H* param, const H* grad, H* m, H* v, H* v_max,
                                      size_t size, float lr, float beta1, float beta2, float eps,
                                      float weight_decay, bool amsgrad,
                                      float bias_correction1, float bias_correction2) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float g = optim_h2f(grad[idx]);
    float p = optim_h2f(param[idx]);
    if (weight_decay != 0.0f) g += weight_decay * p;
    float mi = beta1 * optim_h2f(m[idx]) + (1.0f - beta1) * g;
    float vi = beta2 * optim_h2f(v[idx]) + (1.0f - beta2) * g * g;
    optim_f2h(&m[idx], mi);
    optim_f2h(&v[idx], vi);
    float m_hat = mi / bias_correction1;
    float v_hat;
    if (amsgrad && v_max) {
        float vmi = optim_h2f(v_max[idx]);
        if (vi > vmi) vmi = vi;
        optim_f2h(&v_max[idx], vmi);
        v_hat = vmi / bias_correction2;
    } else {
        v_hat = vi / bias_correction2;
    }
    optim_f2h(&param[idx], p - lr * m_hat / (sqrtf(v_hat) + eps));
}

template<typename T>
__global__ void sgd_step_kernel(T* param, const T* grad, T* buf,
                                size_t size, T lr, T momentum, T dampening,
                                bool nesterov, T weight_decay,
                                bool has_momentum_buf, bool use_momentum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    T g = grad[idx];
    if (weight_decay != T(0)) g += weight_decay * param[idx];
    if (use_momentum) {
        if (!has_momentum_buf) {
            buf[idx] = g;
        } else {
            buf[idx] = momentum * buf[idx] + (T(1) - dampening) * g;
        }
        if (nesterov) {
            g = g + momentum * buf[idx];
        } else {
            g = buf[idx];
        }
    }
    param[idx] -= lr * g;
}

template<typename T>
__global__ void adam_step_kernel(T* param, const T* grad, T* m, T* v, T* v_max,
                                 size_t size, T lr, T beta1, T beta2, T eps,
                                 T weight_decay, bool amsgrad,
                                 T bias_correction1, T bias_correction2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    T g = grad[idx];
    if (weight_decay != T(0)) g += weight_decay * param[idx];
    m[idx] = beta1 * m[idx] + (T(1) - beta1) * g;
    v[idx] = beta2 * v[idx] + (T(1) - beta2) * g * g;
    T m_hat = m[idx] / bias_correction1;
    T v_hat;
    if (amsgrad && v_max) {
        if (v[idx] > v_max[idx]) v_max[idx] = v[idx];
        v_hat = v_max[idx] / bias_correction2;
    } else {
        v_hat = v[idx] / bias_correction2;
    }
    param[idx] -= lr * m_hat / (sqrt(v_hat) + eps);
}

extern "C" int sgd_step_device(void* param_data, const void* grad_data, void* buf_data,
                                size_t size, DType dtype, double lr, double momentum,
                                double dampening, bool nesterov, double weight_decay,
                                bool has_momentum_buf) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    bool use_momentum = (momentum != 0.0);

    if (dtype == DTYPE_FLOAT32) {
        sgd_step_kernel<float><<<blocks, threads>>>(
            (float*)param_data, (const float*)grad_data, (float*)buf_data,
            size, (float)lr, (float)momentum, (float)dampening,
            nesterov, (float)weight_decay, has_momentum_buf, use_momentum);
    } else if (dtype == DTYPE_FLOAT64) {
        sgd_step_kernel<double><<<blocks, threads>>>(
            (double*)param_data, (const double*)grad_data, (double*)buf_data,
            size, lr, momentum, dampening,
            nesterov, weight_decay, has_momentum_buf, use_momentum);
    } else if (dtype == DTYPE_FLOAT16) {
        sgd_step_half_kernel<__half><<<blocks, threads>>>(
            (__half*)param_data, (const __half*)grad_data, (__half*)buf_data,
            size, (float)lr, (float)momentum, (float)dampening,
            nesterov, (float)weight_decay, has_momentum_buf, use_momentum);
    } else if (dtype == DTYPE_BFLOAT16) {
        sgd_step_half_kernel<__nv_bfloat16><<<blocks, threads>>>(
            (__nv_bfloat16*)param_data, (const __nv_bfloat16*)grad_data, (__nv_bfloat16*)buf_data,
            size, (float)lr, (float)momentum, (float)dampening,
            nesterov, (float)weight_decay, has_momentum_buf, use_momentum);
    } else {
        return -1;
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int adam_step_device(void* param_data, const void* grad_data, void* m_data,
                                 void* v_data, void* v_max_data, size_t size, DType dtype,
                                 double lr, double beta1, double beta2, double eps,
                                 double weight_decay, bool amsgrad, int step_count) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    double bc1 = 1.0 - pow(beta1, step_count);
    double bc2 = 1.0 - pow(beta2, step_count);

    if (dtype == DTYPE_FLOAT32) {
        adam_step_kernel<float><<<blocks, threads>>>(
            (float*)param_data, (const float*)grad_data,
            (float*)m_data, (float*)v_data, (float*)v_max_data,
            size, (float)lr, (float)beta1, (float)beta2, (float)eps,
            (float)weight_decay, amsgrad, (float)bc1, (float)bc2);
    } else if (dtype == DTYPE_FLOAT64) {
        adam_step_kernel<double><<<blocks, threads>>>(
            (double*)param_data, (const double*)grad_data,
            (double*)m_data, (double*)v_data, (double*)v_max_data,
            size, lr, beta1, beta2, eps,
            weight_decay, amsgrad, bc1, bc2);
    } else if (dtype == DTYPE_FLOAT16) {
        adam_step_half_kernel<__half><<<blocks, threads>>>(
            (__half*)param_data, (const __half*)grad_data,
            (__half*)m_data, (__half*)v_data, (__half*)v_max_data,
            size, (float)lr, (float)beta1, (float)beta2, (float)eps,
            (float)weight_decay, amsgrad, (float)bc1, (float)bc2);
    } else if (dtype == DTYPE_BFLOAT16) {
        adam_step_half_kernel<__nv_bfloat16><<<blocks, threads>>>(
            (__nv_bfloat16*)param_data, (const __nv_bfloat16*)grad_data,
            (__nv_bfloat16*)m_data, (__nv_bfloat16*)v_data, (__nv_bfloat16*)v_max_data,
            size, (float)lr, (float)beta1, (float)beta2, (float)eps,
            (float)weight_decay, amsgrad, (float)bc1, (float)bc2);
    } else {
        return -1;
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
