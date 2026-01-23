#ifndef OPS_H
#define OPS_H

#include "../core/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GradFn GradFn;

typedef void (*BackwardFunction)(GradFn* self, Tensor* grad_output);

struct GradFn {
    BackwardFunction backward;
    Tensor** inputs;
    int num_inputs;
    void* saved_data;
};

Tensor* op_add(Tensor* x1, Tensor* x2);
Tensor* op_mul(Tensor* x1, Tensor* x2);
Tensor* op_sub(Tensor* x1, Tensor* x2);
Tensor* op_div(Tensor* x1, Tensor* x2);

Tensor* op_exp(Tensor* x);
Tensor* op_log(Tensor* x);
Tensor* op_sqrt(Tensor* x);
Tensor* op_tanh(Tensor* x);

Tensor* op_cat(Tensor** tensors, int num_tensors, int dim);
Tensor* op_slice(Tensor* src, int* start, int* stop, int* step);
Tensor* op_view(Tensor* src, int ndim, int* new_shape);
Tensor* op_reshape(Tensor* src, int ndim, int* new_shape);
Tensor* op_transpose(Tensor* src, int dim0, int dim1);
Tensor* op_squeeze(Tensor* src, int dim);
Tensor* op_unsqueeze(Tensor* src, int dim);
Tensor* op_flatten(Tensor* src, int start_dim, int end_dim);

int backwards_add_x1(const void* grad_c, void* grad_x1, size_t size, DType dtype, int device_id);
int backwards_add_x2(const void* grad_c, void* grad_x2, size_t size, DType dtype, int device_id);

int backwards_mul_x1(const void* grad_c, const void* x2, void* grad_x1, size_t size, DType dtype, int device_id);
int backwards_mul_x2(const void* grad_c, const void* x1, void* grad_x2, size_t size, DType dtype, int device_id);

int backwards_sub_x1(const void* grad_c, void* grad_x1, size_t size, DType dtype, int device_id);
int backwards_sub_x2(const void* grad_c, void* grad_x2, size_t size, DType dtype, int device_id);

int backwards_div_x1(const void* grad_c, const void* x2, void* grad_x1, size_t size, DType dtype, int device_id);
int backwards_div_x2(const void* grad_c, const void* x1, const void* x2, void* grad_x2, size_t size, DType dtype, int device_id);

int backwards_exp(const void* grad_c, const void* fn_output, void* grad_x, size_t size, DType dtype, int device_id);
int backwards_log(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype, int device_id);
int backwards_sqrt(const void* grad_c, const void* fn_output, void* grad_x, size_t size, DType dtype, int device_id);
int backwards_tanh(const void* grad_c, const void* fn_output, void* grad_x, size_t size, DType dtype, int device_id);

int backwards_add_x1_host(const void* grad_c, void* grad_x1, size_t size, DType dtype);
int backwards_add_x2_host(const void* grad_c, void* grad_x2, size_t size, DType dtype);

int backwards_mul_x1_host(const void* grad_c, const void* x2, void* grad_x1, size_t size, DType dtype);
int backwards_mul_x2_host(const void* grad_c, const void* x1, void* grad_x2, size_t size, DType dtype);

int backwards_sub_x1_host(const void* grad_c, void* grad_x1, size_t size, DType dtype);
int backwards_sub_x2_host(const void* grad_c, void* grad_x2, size_t size, DType dtype);

int backwards_div_x1_host(const void* grad_c, const void* x2, void* grad_x1, size_t size, DType dtype);
int backwards_div_x2_host(const void* grad_c, const void* x1, const void* x2, void* grad_x2, size_t size, DType dtype);

int backwards_exp_host(const void* grad_c, const void* fn_output, void* grad_x, size_t size, DType dtype);
int backwards_log_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_sqrt_host(const void* grad_c, const void* fn_output, void* grad_x, size_t size, DType dtype);
int backwards_tanh_host(const void* grad_c, const void* fn_output, void* grad_x, size_t size, DType dtype);

int backwards_add_x1_device(const void* grad_c, void* grad_x1, size_t size, DType dtype);
int backwards_add_x2_device(const void* grad_c, void* grad_x2, size_t size, DType dtype);

int backwards_mul_x1_device(const void* grad_c, const void* x2, void* grad_x1, size_t size, DType dtype);
int backwards_mul_x2_device(const void* grad_c, const void* x1, void* grad_x2, size_t size, DType dtype);

int backwards_sub_x1_device(const void* grad_c, void* grad_x1, size_t size, DType dtype);
int backwards_sub_x2_device(const void* grad_c, void* grad_x2, size_t size, DType dtype);

int backwards_div_x1_device(const void* grad_c, const void* x2, void* grad_x1, size_t size, DType dtype);
int backwards_div_x2_device(const void* grad_c, const void* x1, const void* x2, void* grad_x2, size_t size, DType dtype);

int backwards_exp_device(const void* grad_c, const void* fn_output, void* grad_x, size_t size, DType dtype);
int backwards_log_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_sqrt_device(const void* grad_c, const void* fn_output, void* grad_x, size_t size, DType dtype);
int backwards_tanh_device(const void* grad_c, const void* fn_output, void* grad_x, size_t size, DType dtype);

void free_grad_fn(GradFn* grad_fn);

#ifdef __cplusplus
}
#endif

#endif
