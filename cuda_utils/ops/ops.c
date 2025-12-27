#include "ops.h"
#include "../functional/functional.h"
#include <stdlib.h>
#include <string.h>

int backwards_add_a(const void* grad_c, void* grad_a,
                    size_t size, DType dtype, int device_id) {
    if (!grad_c || !grad_a) return -1;

    if (device_id == -1) {
        return backwards_add_a_host(grad_c, grad_a, size, dtype);
    } else {
        return backwards_add_a_device(grad_c, grad_a, size, dtype);
    }
}

int backwards_add_b(const void* grad_c, void* grad_b,
                    size_t size, DType dtype, int device_id) {
    if (!grad_c || !grad_b) return -1;

    if (device_id == -1) {
        return backwards_add_b_host(grad_c, grad_b, size, dtype);
    } else {
        return backwards_add_b_device(grad_c, grad_b, size, dtype);
    }
}

void backward_add_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* a = self->inputs[0];
    Tensor* b = self->inputs[1];

    if (a->metadata && a->metadata->requires_grad) {
        if (!a->metadata->grad) {
            a->metadata->grad = zeros_tensor(a->dtype, a->device_id, a->ndim, a->shape, NULL);
            if (a->metadata->grad) {
                backwards_add_a(grad_output->data, a->metadata->grad->data,
                               grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            rp_add(a->metadata->grad->data, a->metadata->grad->data, grad_output->data,
                   grad_output->size, grad_output->dtype, grad_output->device_id);
        }
    }

    if (b->metadata && b->metadata->requires_grad) {
        if (!b->metadata->grad) {
            b->metadata->grad = zeros_tensor(b->dtype, b->device_id, b->ndim, b->shape, NULL);
            if (b->metadata->grad) {
                backwards_add_b(grad_output->data, b->metadata->grad->data,
                               grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            rp_add(b->metadata->grad->data, b->metadata->grad->data, grad_output->data,
                   grad_output->size, grad_output->dtype, grad_output->device_id);
        }
    }
}

Tensor* op_add(Tensor* a, Tensor* b) {
    if (!a || !b) return NULL;

    if (a->ndim != b->ndim || a->size != b->size) return NULL;
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return NULL;
    }

    if (a->dtype != b->dtype || a->device_id != b->device_id) return NULL;

    Tensor* out = zeros_tensor(a->dtype, a->device_id, a->ndim, a->shape, NULL);
    if (!out) return NULL;

    int result = rp_add(out->data, a->data, b->data, a->size, a->dtype, a->device_id);
    if (result != 0) {
        free_tensor(out);
        return NULL;
    }

    bool requires_grad = false;
    if ((a->metadata && a->metadata->requires_grad) || (b->metadata && b->metadata->requires_grad)) {
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
        grad_fn->inputs[0] = a;
        grad_fn->inputs[1] = b;
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
