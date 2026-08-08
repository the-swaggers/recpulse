#include "optim.h"
#include "../core/half_precision.h"
#include <math.h>
#include <stdlib.h>

int sgd_step_host(void* param_data, const void* grad_data, void* buf_data,
                  size_t size, DType dtype, double lr, double momentum,
                  double dampening, bool nesterov, double weight_decay,
                  bool has_momentum_buf) {
    if (dtype == DTYPE_FLOAT32) {
        float* p = (float*)param_data;
        const float* g_in = (const float*)grad_data;
        float* buf = (float*)buf_data;
        float lr_f = (float)lr;
        float mom_f = (float)momentum;
        float damp_f = (float)dampening;
        float wd_f = (float)weight_decay;

        for (size_t i = 0; i < size; i++) {
            float g = g_in[i];
            if (wd_f != 0.0f) g += wd_f * p[i];
            if (mom_f != 0.0f) {
                if (!has_momentum_buf) {
                    buf[i] = g;
                } else {
                    buf[i] = mom_f * buf[i] + (1.0f - damp_f) * g;
                }
                if (nesterov) {
                    g = g + mom_f * buf[i];
                } else {
                    g = buf[i];
                }
            }
            p[i] -= lr_f * g;
        }
    } else if (dtype == DTYPE_FLOAT64) {
        double* p = (double*)param_data;
        const double* g_in = (const double*)grad_data;
        double* buf = (double*)buf_data;

        for (size_t i = 0; i < size; i++) {
            double g = g_in[i];
            if (weight_decay != 0.0) g += weight_decay * p[i];
            if (momentum != 0.0) {
                if (!has_momentum_buf) {
                    buf[i] = g;
                } else {
                    buf[i] = momentum * buf[i] + (1.0 - dampening) * g;
                }
                if (nesterov) {
                    g = g + momentum * buf[i];
                } else {
                    g = buf[i];
                }
            }
            p[i] -= lr * g;
        }
    } else if (dtype == DTYPE_FLOAT16 || dtype == DTYPE_BFLOAT16) {
        float* p = (float*)malloc(size * sizeof(float));
        float* g_f = (float*)malloc(size * sizeof(float));
        float* buf = buf_data ? (float*)malloc(size * sizeof(float)) : NULL;
        if (!p || !g_f || (buf_data && !buf)) {
            free(p); free(g_f); free(buf);
            return -1;
        }
        half_to_fp32_array(param_data, p, size, dtype);
        half_to_fp32_array(grad_data, g_f, size, dtype);
        if (buf_data) half_to_fp32_array(buf_data, buf, size, dtype);

        float lr_f = (float)lr;
        float mom_f = (float)momentum;
        float damp_f = (float)dampening;
        float wd_f = (float)weight_decay;

        for (size_t i = 0; i < size; i++) {
            float g = g_f[i];
            if (wd_f != 0.0f) g += wd_f * p[i];
            if (mom_f != 0.0f) {
                if (!has_momentum_buf) {
                    buf[i] = g;
                } else {
                    buf[i] = mom_f * buf[i] + (1.0f - damp_f) * g;
                }
                if (nesterov) {
                    g = g + mom_f * buf[i];
                } else {
                    g = buf[i];
                }
            }
            p[i] -= lr_f * g;
        }

        fp32_to_half_array(p, param_data, size, dtype);
        if (buf_data) fp32_to_half_array(buf, buf_data, size, dtype);
        free(p); free(g_f); free(buf);
    } else {
        return -1;
    }
    return 0;
}

int adam_step_host(void* param_data, const void* grad_data, void* m_data,
                   void* v_data, void* v_max_data, size_t size, DType dtype,
                   double lr, double beta1, double beta2, double eps,
                   double weight_decay, bool amsgrad, int step_count) {
    double bias_correction1 = 1.0 - pow(beta1, step_count);
    double bias_correction2 = 1.0 - pow(beta2, step_count);

    if (dtype == DTYPE_FLOAT32) {
        float* p = (float*)param_data;
        const float* g_in = (const float*)grad_data;
        float* m = (float*)m_data;
        float* v = (float*)v_data;
        float* vm = (float*)v_max_data;
        float b1 = (float)beta1;
        float b2 = (float)beta2;
        float eps_f = (float)eps;
        float wd_f = (float)weight_decay;
        float lr_f = (float)lr;
        float bc1 = (float)bias_correction1;
        float bc2 = (float)bias_correction2;

        for (size_t i = 0; i < size; i++) {
            float g = g_in[i];
            if (wd_f != 0.0f) g += wd_f * p[i];
            m[i] = b1 * m[i] + (1.0f - b1) * g;
            v[i] = b2 * v[i] + (1.0f - b2) * g * g;
            float m_hat = m[i] / bc1;
            float v_hat;
            if (amsgrad && vm) {
                if (v[i] > vm[i]) vm[i] = v[i];
                v_hat = vm[i] / bc2;
            } else {
                v_hat = v[i] / bc2;
            }
            p[i] -= lr_f * m_hat / (sqrtf(v_hat) + eps_f);
        }
    } else if (dtype == DTYPE_FLOAT64) {
        double* p = (double*)param_data;
        const double* g_in = (const double*)grad_data;
        double* m = (double*)m_data;
        double* v = (double*)v_data;
        double* vm = (double*)v_max_data;

        for (size_t i = 0; i < size; i++) {
            double g = g_in[i];
            if (weight_decay != 0.0) g += weight_decay * p[i];
            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
            double m_hat = m[i] / bias_correction1;
            double v_hat;
            if (amsgrad && vm) {
                if (v[i] > vm[i]) vm[i] = v[i];
                v_hat = vm[i] / bias_correction2;
            } else {
                v_hat = v[i] / bias_correction2;
            }
            p[i] -= lr * m_hat / (sqrt(v_hat) + eps);
        }
    } else if (dtype == DTYPE_FLOAT16 || dtype == DTYPE_BFLOAT16) {
        float* p = (float*)malloc(size * sizeof(float));
        float* g_f = (float*)malloc(size * sizeof(float));
        float* m = (float*)malloc(size * sizeof(float));
        float* v = (float*)malloc(size * sizeof(float));
        float* vm = v_max_data ? (float*)malloc(size * sizeof(float)) : NULL;
        if (!p || !g_f || !m || !v || (v_max_data && !vm)) {
            free(p); free(g_f); free(m); free(v); free(vm);
            return -1;
        }
        half_to_fp32_array(param_data, p, size, dtype);
        half_to_fp32_array(grad_data, g_f, size, dtype);
        half_to_fp32_array(m_data, m, size, dtype);
        half_to_fp32_array(v_data, v, size, dtype);
        if (v_max_data) half_to_fp32_array(v_max_data, vm, size, dtype);

        float b1 = (float)beta1;
        float b2 = (float)beta2;
        float eps_f = (float)eps;
        float wd_f = (float)weight_decay;
        float lr_f = (float)lr;
        float bc1 = (float)bias_correction1;
        float bc2 = (float)bias_correction2;

        for (size_t i = 0; i < size; i++) {
            float g = g_f[i];
            if (wd_f != 0.0f) g += wd_f * p[i];
            m[i] = b1 * m[i] + (1.0f - b1) * g;
            v[i] = b2 * v[i] + (1.0f - b2) * g * g;
            float m_hat = m[i] / bc1;
            float v_hat;
            if (amsgrad && vm) {
                if (v[i] > vm[i]) vm[i] = v[i];
                v_hat = vm[i] / bc2;
            } else {
                v_hat = v[i] / bc2;
            }
            p[i] -= lr_f * m_hat / (sqrtf(v_hat) + eps_f);
        }

        fp32_to_half_array(p, param_data, size, dtype);
        fp32_to_half_array(m, m_data, size, dtype);
        fp32_to_half_array(v, v_data, size, dtype);
        if (v_max_data) fp32_to_half_array(vm, v_max_data, size, dtype);
        free(p); free(g_f); free(m); free(v); free(vm);
    } else {
        return -1;
    }
    return 0;
}
