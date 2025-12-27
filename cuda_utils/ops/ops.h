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

int backwards_add_x1(const void* grad_c, void* grad_x1, size_t size, DType dtype, int device_id);
int backwards_add_x2(const void* grad_c, void* grad_x2, size_t size, DType dtype, int device_id);

int backwards_add_x1_host(const void* grad_c, void* grad_x1, size_t size, DType dtype);
int backwards_add_x2_host(const void* grad_c, void* grad_x2, size_t size, DType dtype);

int backwards_add_x1_device(const void* grad_c, void* grad_x1, size_t size, DType dtype);
int backwards_add_x2_device(const void* grad_c, void* grad_x2, size_t size, DType dtype);

void free_grad_fn(GradFn* grad_fn);

#ifdef __cplusplus
}
#endif

#endif
