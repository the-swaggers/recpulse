#include "ops.h"
#include "../functional/functional.h"
#include <stdlib.h>
#include <string.h>

int backwards_add_x1(const void* grad_c, void* grad_x1,
                     size_t size, DType dtype, int device_id) {
    if (!grad_c || !grad_x1) return -1;

    if (device_id == -1) {
        return backwards_add_x1_host(grad_c, grad_x1, size, dtype);
    } else {
        return backwards_add_x1_device(grad_c, grad_x1, size, dtype);
    }
}

int backwards_add_x2(const void* grad_c, void* grad_x2,
                     size_t size, DType dtype, int device_id) {
    if (!grad_c || !grad_x2) return -1;

    if (device_id == -1) {
        return backwards_add_x2_host(grad_c, grad_x2, size, dtype);
    } else {
        return backwards_add_x2_device(grad_c, grad_x2, size, dtype);
    }
}

void backward_add_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x1 = self->inputs[0];
    Tensor* x2 = self->inputs[1];

    if (x1->metadata && x1->metadata->requires_grad) {
        if (!x1->metadata->grad) {
            x1->metadata->grad = zeros_tensor(x1->dtype, x1->device_id, x1->ndim, x1->shape, NULL);
            if (x1->metadata->grad) {
                backwards_add_x1(grad_output->data, x1->metadata->grad->data,
                                grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            rp_add(x1->metadata->grad->data, x1->metadata->grad->data, grad_output->data,
                   grad_output->size, grad_output->dtype, grad_output->device_id);
        }
    }

    if (x2->metadata && x2->metadata->requires_grad) {
        if (!x2->metadata->grad) {
            x2->metadata->grad = zeros_tensor(x2->dtype, x2->device_id, x2->ndim, x2->shape, NULL);
            if (x2->metadata->grad) {
                backwards_add_x2(grad_output->data, x2->metadata->grad->data,
                                grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            rp_add(x2->metadata->grad->data, x2->metadata->grad->data, grad_output->data,
                   grad_output->size, grad_output->dtype, grad_output->device_id);
        }
    }
}

Tensor* op_add(Tensor* x1, Tensor* x2) {
    if (!x1 || !x2) return NULL;

    if (x1->ndim != x2->ndim || x1->size != x2->size) return NULL;
    for (int i = 0; i < x1->ndim; i++) {
        if (x1->shape[i] != x2->shape[i]) return NULL;
    }

    if (x1->dtype != x2->dtype || x1->device_id != x2->device_id) return NULL;

    Tensor* out = zeros_tensor(x1->dtype, x1->device_id, x1->ndim, x1->shape, NULL);
    if (!out) return NULL;

    int result = rp_add(out->data, x1->data, x2->data, x1->size, x1->dtype, x1->device_id);
    if (result != 0) {
        free_tensor(out);
        return NULL;
    }

    bool requires_grad = false;
    if ((x1->metadata && x1->metadata->requires_grad) || (x2->metadata && x2->metadata->requires_grad)) {
        requires_grad = true;
    }

    if (requires_grad) {
        if (!out->metadata) {
            out->metadata = (Meta*)calloc(1, sizeof(Meta));
            if (!out->metadata) {
                free_tensor(out);
                return NULL;
            }
        }
        out->metadata->requires_grad = true;
        out->metadata->is_leaf = false;

        GradFn* grad_fn = (GradFn*)calloc(1, sizeof(GradFn));
        if (!grad_fn) {
            free_tensor(out);
            return NULL;
        }

        grad_fn->backward = backward_add_fn;
        grad_fn->num_inputs = 2;
        grad_fn->inputs = (Tensor**)malloc(2 * sizeof(Tensor*));
        if (!grad_fn->inputs) {
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        grad_fn->inputs[0] = x1;
        grad_fn->inputs[1] = x2;
        grad_fn->saved_data = NULL;

        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

void free_grad_fn(GradFn* grad_fn) {
    if (!grad_fn) return;
    if (grad_fn->inputs) free(grad_fn->inputs);
    if (grad_fn->saved_data) free(grad_fn->saved_data);
    free(grad_fn);
}
