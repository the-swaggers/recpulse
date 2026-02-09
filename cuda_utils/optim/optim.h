#ifndef OPTIM_H
#define OPTIM_H

#include "../core/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

int sgd_step(Tensor* param, double lr, double momentum, double dampening,
             bool nesterov, double weight_decay, Tensor* momentum_buf,
             bool has_momentum_buf);

int adam_step(Tensor* param, double lr, double beta1, double beta2,
             double eps, double weight_decay, bool amsgrad,
             Tensor* m, Tensor* v, Tensor* v_max, int step_count);

int sgd_step_host(void* param_data, const void* grad_data, void* buf_data,
                  size_t size, DType dtype, double lr, double momentum,
                  double dampening, bool nesterov, double weight_decay,
                  bool has_momentum_buf);

int sgd_step_device(void* param_data, const void* grad_data, void* buf_data,
                    size_t size, DType dtype, double lr, double momentum,
                    double dampening, bool nesterov, double weight_decay,
                    bool has_momentum_buf);

int adam_step_host(void* param_data, const void* grad_data, void* m_data,
                   void* v_data, void* v_max_data, size_t size, DType dtype,
                   double lr, double beta1, double beta2, double eps,
                   double weight_decay, bool amsgrad, int step_count);

int adam_step_device(void* param_data, const void* grad_data, void* m_data,
                     void* v_data, void* v_max_data, size_t size, DType dtype,
                     double lr, double beta1, double beta2, double eps,
                     double weight_decay, bool amsgrad, int step_count);

#ifdef __cplusplus
}
#endif

#endif
