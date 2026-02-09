#include "optim.h"
#include <math.h>

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
    } else {
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
            float v_hat = v[i] / bc2;
            if (amsgrad && vm) {
                if (v_hat > vm[i]) vm[i] = v_hat;
                v_hat = vm[i];
            }
            p[i] -= lr_f * m_hat / (sqrtf(v_hat) + eps_f);
        }
    } else {
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
            double v_hat = v[i] / bias_correction2;
            if (amsgrad && vm) {
                if (v_hat > vm[i]) vm[i] = v_hat;
                v_hat = vm[i];
            }
            p[i] -= lr * m_hat / (sqrt(v_hat) + eps);
        }
    }
    return 0;
}
