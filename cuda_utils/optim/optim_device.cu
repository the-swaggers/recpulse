#include "optim.h"
#include <cuda_runtime.h>
#include <math.h>

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
    T v_hat = v[idx] / bias_correction2;
    if (amsgrad && v_max) {
        if (v_hat > v_max[idx]) v_max[idx] = v_hat;
        v_hat = v_max[idx];
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
    } else {
        sgd_step_kernel<double><<<blocks, threads>>>(
            (double*)param_data, (const double*)grad_data, (double*)buf_data,
            size, lr, momentum, dampening,
            nesterov, weight_decay, has_momentum_buf, use_momentum);
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
    } else {
        adam_step_kernel<double><<<blocks, threads>>>(
            (double*)param_data, (const double*)grad_data,
            (double*)m_data, (double*)v_data, (double*)v_max_data,
            size, lr, beta1, beta2, eps,
            weight_decay, amsgrad, bc1, bc2);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
