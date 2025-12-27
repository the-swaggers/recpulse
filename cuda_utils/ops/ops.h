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

Tensor* op_add(Tensor* a, Tensor* b);

int backwards_add_a(const void* grad_c, void* grad_a, size_t size, DType dtype, int device_id);
int backwards_add_b(const void* grad_c, void* grad_b, size_t size, DType dtype, int device_id);

int backwards_add_a_host(const void* grad_c, void* grad_a, size_t size, DType dtype);
int backwards_add_b_host(const void* grad_c, void* grad_b, size_t size, DType dtype);

int backwards_add_a_device(const void* grad_c, void* grad_a, size_t size, DType dtype);
int backwards_add_b_device(const void* grad_c, void* grad_b, size_t size, DType dtype);

void free_grad_fn(GradFn* grad_fn);

#ifdef __cplusplus
}
#endif

#endif
