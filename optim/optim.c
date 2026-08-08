#include "optim.h"
#include <stdio.h>

int sgd_step(Tensor* param, double lr, double momentum, double dampening,
             bool nesterov, double weight_decay, Tensor* momentum_buf,
             bool has_momentum_buf) {
    if (!param || !param->metadata || !param->metadata->grad) {
        fprintf(stderr, "Error: param must have gradient for sgd_step\n");
        return -1;
    }

    Tensor* grad = param->metadata->grad;
    void* buf_data = (momentum != 0.0 && momentum_buf) ? momentum_buf->data : NULL;

    if (param->device_id == -1) {
        return sgd_step_host(param->data, grad->data, buf_data,
                             param->size, param->dtype, lr, momentum,
                             dampening, nesterov, weight_decay, has_momentum_buf);
    } else {
        return sgd_step_device(param->data, grad->data, buf_data,
                               param->size, param->dtype, lr, momentum,
                               dampening, nesterov, weight_decay, has_momentum_buf);
    }
}

int adam_step(Tensor* param, double lr, double beta1, double beta2,
             double eps, double weight_decay, bool amsgrad,
             Tensor* m, Tensor* v, Tensor* v_max, int step_count) {
    if (!param || !param->metadata || !param->metadata->grad) {
        fprintf(stderr, "Error: param must have gradient for adam_step\n");
        return -1;
    }
    if (!m || !v) {
        fprintf(stderr, "Error: m and v state tensors required for adam_step\n");
        return -1;
    }

    Tensor* grad = param->metadata->grad;
    void* v_max_data = (amsgrad && v_max) ? v_max->data : NULL;

    if (param->device_id == -1) {
        return adam_step_host(param->data, grad->data, m->data, v->data,
                              v_max_data, param->size, param->dtype,
                              lr, beta1, beta2, eps, weight_decay,
                              amsgrad, step_count);
    } else {
        return adam_step_device(param->data, grad->data, m->data, v->data,
                                v_max_data, param->size, param->dtype,
                                lr, beta1, beta2, eps, weight_decay,
                                amsgrad, step_count);
    }
}
