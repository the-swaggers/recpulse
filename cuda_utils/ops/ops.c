#include "ops.h"
#include "../functional/functional.h"
#include <stdlib.h>
#include <string.h>
#include <limits.h>

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

int backwards_mul_x1(const void* grad_c, const void* x2, void* grad_x1,
                     size_t size, DType dtype, int device_id) {
    if (!grad_c || !x2 || !grad_x1) return -1;

    if (device_id == -1) {
        return backwards_mul_x1_host(grad_c, x2, grad_x1, size, dtype);
    } else {
        return backwards_mul_x1_device(grad_c, x2, grad_x1, size, dtype);
    }
}

int backwards_mul_x2(const void* grad_c, const void* x1, void* grad_x2,
                     size_t size, DType dtype, int device_id) {
    if (!grad_c || !x1 || !grad_x2) return -1;

    if (device_id == -1) {
        return backwards_mul_x2_host(grad_c, x1, grad_x2, size, dtype);
    } else {
        return backwards_mul_x2_device(grad_c, x1, grad_x2, size, dtype);
    }
}

int backwards_sub_x1(const void* grad_c, void* grad_x1,
                     size_t size, DType dtype, int device_id) {
    if (!grad_c || !grad_x1) return -1;

    if (device_id == -1) {
        return backwards_sub_x1_host(grad_c, grad_x1, size, dtype);
    } else {
        return backwards_sub_x1_device(grad_c, grad_x1, size, dtype);
    }
}

int backwards_sub_x2(const void* grad_c, void* grad_x2,
                     size_t size, DType dtype, int device_id) {
    if (!grad_c || !grad_x2) return -1;

    if (device_id == -1) {
        return backwards_sub_x2_host(grad_c, grad_x2, size, dtype);
    } else {
        return backwards_sub_x2_device(grad_c, grad_x2, size, dtype);
    }
}

int backwards_div_x1(const void* grad_c, const void* x2, void* grad_x1,
                     size_t size, DType dtype, int device_id) {
    if (!grad_c || !x2 || !grad_x1) return -1;

    if (device_id == -1) {
        return backwards_div_x1_host(grad_c, x2, grad_x1, size, dtype);
    } else {
        return backwards_div_x1_device(grad_c, x2, grad_x1, size, dtype);
    }
}

int backwards_div_x2(const void* grad_c, const void* x1, const void* x2, void* grad_x2,
                     size_t size, DType dtype, int device_id) {
    if (!grad_c || !x1 || !x2 || !grad_x2) return -1;

    if (device_id == -1) {
        return backwards_div_x2_host(grad_c, x1, x2, grad_x2, size, dtype);
    } else {
        return backwards_div_x2_device(grad_c, x1, x2, grad_x2, size, dtype);
    }
}

void backward_mul_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x1 = self->inputs[0];
    Tensor* x2 = self->inputs[1];

    if (x1->metadata && x1->metadata->requires_grad) {
        if (!x1->metadata->grad) {
            x1->metadata->grad = zeros_tensor(x1->dtype, x1->device_id, x1->ndim, x1->shape, NULL);
            if (x1->metadata->grad) {
                backwards_mul_x1(grad_output->data, x2->data, x1->metadata->grad->data,
                                grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x1->dtype, x1->device_id, x1->ndim, x1->shape, NULL);
            if (temp_grad) {
                backwards_mul_x1(grad_output->data, x2->data, temp_grad->data,
                                grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x1->metadata->grad->data, x1->metadata->grad->data, temp_grad->data,
                       grad_output->size, grad_output->dtype, grad_output->device_id);
                free_tensor(temp_grad);
            }
        }
    }

    if (x2->metadata && x2->metadata->requires_grad) {
        if (!x2->metadata->grad) {
            x2->metadata->grad = zeros_tensor(x2->dtype, x2->device_id, x2->ndim, x2->shape, NULL);
            if (x2->metadata->grad) {
                backwards_mul_x2(grad_output->data, x1->data, x2->metadata->grad->data,
                                grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x2->dtype, x2->device_id, x2->ndim, x2->shape, NULL);
            if (temp_grad) {
                backwards_mul_x2(grad_output->data, x1->data, temp_grad->data,
                                grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x2->metadata->grad->data, x2->metadata->grad->data, temp_grad->data,
                       grad_output->size, grad_output->dtype, grad_output->device_id);
                free_tensor(temp_grad);
            }
        }
    }
}

void backward_sub_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x1 = self->inputs[0];
    Tensor* x2 = self->inputs[1];

    if (x1->metadata && x1->metadata->requires_grad) {
        if (!x1->metadata->grad) {
            x1->metadata->grad = zeros_tensor(x1->dtype, x1->device_id, x1->ndim, x1->shape, NULL);
            if (x1->metadata->grad) {
                backwards_sub_x1(grad_output->data, x1->metadata->grad->data,
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
                backwards_sub_x2(grad_output->data, x2->metadata->grad->data,
                                grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x2->dtype, x2->device_id, x2->ndim, x2->shape, NULL);
            if (temp_grad) {
                backwards_sub_x2(grad_output->data, temp_grad->data,
                                grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x2->metadata->grad->data, x2->metadata->grad->data, temp_grad->data,
                       grad_output->size, grad_output->dtype, grad_output->device_id);
                free_tensor(temp_grad);
            }
        }
    }
}

void backward_div_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x1 = self->inputs[0];
    Tensor* x2 = self->inputs[1];

    if (x1->metadata && x1->metadata->requires_grad) {
        if (!x1->metadata->grad) {
            x1->metadata->grad = zeros_tensor(x1->dtype, x1->device_id, x1->ndim, x1->shape, NULL);
            if (x1->metadata->grad) {
                backwards_div_x1(grad_output->data, x2->data, x1->metadata->grad->data,
                                grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x1->dtype, x1->device_id, x1->ndim, x1->shape, NULL);
            if (temp_grad) {
                backwards_div_x1(grad_output->data, x2->data, temp_grad->data,
                                grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x1->metadata->grad->data, x1->metadata->grad->data, temp_grad->data,
                       grad_output->size, grad_output->dtype, grad_output->device_id);
                free_tensor(temp_grad);
            }
        }
    }

    if (x2->metadata && x2->metadata->requires_grad) {
        if (!x2->metadata->grad) {
            x2->metadata->grad = zeros_tensor(x2->dtype, x2->device_id, x2->ndim, x2->shape, NULL);
            if (x2->metadata->grad) {
                backwards_div_x2(grad_output->data, x1->data, x2->data, x2->metadata->grad->data,
                                grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x2->dtype, x2->device_id, x2->ndim, x2->shape, NULL);
            if (temp_grad) {
                backwards_div_x2(grad_output->data, x1->data, x2->data, temp_grad->data,
                                grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x2->metadata->grad->data, x2->metadata->grad->data, temp_grad->data,
                       grad_output->size, grad_output->dtype, grad_output->device_id);
                free_tensor(temp_grad);
            }
        }
    }
}

Tensor* op_mul(Tensor* x1, Tensor* x2) {
    if (!x1 || !x2) return NULL;

    if (x1->ndim != x2->ndim || x1->size != x2->size) return NULL;
    for (int i = 0; i < x1->ndim; i++) {
        if (x1->shape[i] != x2->shape[i]) return NULL;
    }

    if (x1->dtype != x2->dtype || x1->device_id != x2->device_id) return NULL;

    Tensor* out = zeros_tensor(x1->dtype, x1->device_id, x1->ndim, x1->shape, NULL);
    if (!out) return NULL;

    int result = rp_mul(out->data, x1->data, x2->data, x1->size, x1->dtype, x1->device_id);
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

        grad_fn->backward = backward_mul_fn;
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

Tensor* op_sub(Tensor* x1, Tensor* x2) {
    if (!x1 || !x2) return NULL;

    if (x1->ndim != x2->ndim || x1->size != x2->size) return NULL;
    for (int i = 0; i < x1->ndim; i++) {
        if (x1->shape[i] != x2->shape[i]) return NULL;
    }

    if (x1->dtype != x2->dtype || x1->device_id != x2->device_id) return NULL;

    Tensor* out = zeros_tensor(x1->dtype, x1->device_id, x1->ndim, x1->shape, NULL);
    if (!out) return NULL;

    int result = rp_sub(out->data, x1->data, x2->data, x1->size, x1->dtype, x1->device_id);
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

        grad_fn->backward = backward_sub_fn;
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

Tensor* op_div(Tensor* x1, Tensor* x2) {
    if (!x1 || !x2) return NULL;

    if (x1->ndim != x2->ndim || x1->size != x2->size) return NULL;
    for (int i = 0; i < x1->ndim; i++) {
        if (x1->shape[i] != x2->shape[i]) return NULL;
    }

    if (x1->dtype != x2->dtype || x1->device_id != x2->device_id) return NULL;

    Tensor* out = zeros_tensor(x1->dtype, x1->device_id, x1->ndim, x1->shape, NULL);
    if (!out) return NULL;

    int result = rp_divide(out->data, x1->data, x2->data, x1->size, x1->dtype, x1->device_id);
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

        grad_fn->backward = backward_div_fn;
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

int backwards_exp(const void* grad_c, const void* fn_output, void* grad_x,
                  size_t size, DType dtype, int device_id) {
    if (!grad_c || !fn_output || !grad_x) return -1;

    if (device_id == -1) {
        return backwards_exp_host(grad_c, fn_output, grad_x, size, dtype);
    } else {
        return backwards_exp_device(grad_c, fn_output, grad_x, size, dtype);
    }
}

int backwards_log(const void* grad_c, const void* x, void* grad_x,
                  size_t size, DType dtype, int device_id) {
    if (!grad_c || !x || !grad_x) return -1;

    if (device_id == -1) {
        return backwards_log_host(grad_c, x, grad_x, size, dtype);
    } else {
        return backwards_log_device(grad_c, x, grad_x, size, dtype);
    }
}

int backwards_sqrt(const void* grad_c, const void* fn_output, void* grad_x,
                   size_t size, DType dtype, int device_id) {
    if (!grad_c || !fn_output || !grad_x) return -1;

    if (device_id == -1) {
        return backwards_sqrt_host(grad_c, fn_output, grad_x, size, dtype);
    } else {
        return backwards_sqrt_device(grad_c, fn_output, grad_x, size, dtype);
    }
}

int backwards_tanh(const void* grad_c, const void* fn_output, void* grad_x,
                   size_t size, DType dtype, int device_id) {
    if (!grad_c || !fn_output || !grad_x) return -1;

    if (device_id == -1) {
        return backwards_tanh_host(grad_c, fn_output, grad_x, size, dtype);
    } else {
        return backwards_tanh_device(grad_c, fn_output, grad_x, size, dtype);
    }
}

void backward_exp_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];
    Tensor* output = self->inputs[1];

    if (x->metadata && x->metadata->requires_grad) {
        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (x->metadata->grad) {
                backwards_exp(grad_output->data, output->data, x->metadata->grad->data,
                             grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (temp_grad) {
                backwards_exp(grad_output->data, output->data, temp_grad->data,
                             grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x->metadata->grad->data, x->metadata->grad->data, temp_grad->data,
                       grad_output->size, grad_output->dtype, grad_output->device_id);
                free_tensor(temp_grad);
            }
        }
    }
}

void backward_log_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];

    if (x->metadata && x->metadata->requires_grad) {
        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (x->metadata->grad) {
                backwards_log(grad_output->data, x->data, x->metadata->grad->data,
                             grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (temp_grad) {
                backwards_log(grad_output->data, x->data, temp_grad->data,
                             grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x->metadata->grad->data, x->metadata->grad->data, temp_grad->data,
                       grad_output->size, grad_output->dtype, grad_output->device_id);
                free_tensor(temp_grad);
            }
        }
    }
}

void backward_sqrt_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];
    Tensor* output = self->inputs[1];

    if (x->metadata && x->metadata->requires_grad) {
        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (x->metadata->grad) {
                backwards_sqrt(grad_output->data, output->data, x->metadata->grad->data,
                              grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (temp_grad) {
                backwards_sqrt(grad_output->data, output->data, temp_grad->data,
                              grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x->metadata->grad->data, x->metadata->grad->data, temp_grad->data,
                       grad_output->size, grad_output->dtype, grad_output->device_id);
                free_tensor(temp_grad);
            }
        }
    }
}

void backward_tanh_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];
    Tensor* output = self->inputs[1];

    if (x->metadata && x->metadata->requires_grad) {
        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (x->metadata->grad) {
                backwards_tanh(grad_output->data, output->data, x->metadata->grad->data,
                              grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (temp_grad) {
                backwards_tanh(grad_output->data, output->data, temp_grad->data,
                              grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x->metadata->grad->data, x->metadata->grad->data, temp_grad->data,
                       grad_output->size, grad_output->dtype, grad_output->device_id);
                free_tensor(temp_grad);
            }
        }
    }
}

Tensor* op_exp(Tensor* x) {
    if (!x) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    int result = rp_exp(out->data, x->data, x->size, x->dtype, x->device_id);
    if (result != 0) {
        free_tensor(out);
        return NULL;
    }

    bool requires_grad = (x->metadata && x->metadata->requires_grad);

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

        grad_fn->backward = backward_exp_fn;
        grad_fn->num_inputs = 2;
        grad_fn->inputs = (Tensor**)malloc(2 * sizeof(Tensor*));
        if (!grad_fn->inputs) {
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        grad_fn->inputs[0] = x;
        grad_fn->inputs[1] = out;
        grad_fn->saved_data = NULL;

        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

Tensor* op_log(Tensor* x) {
    if (!x) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    int result = rp_log(out->data, x->data, x->size, x->dtype, x->device_id);
    if (result != 0) {
        free_tensor(out);
        return NULL;
    }

    bool requires_grad = (x->metadata && x->metadata->requires_grad);

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

        grad_fn->backward = backward_log_fn;
        grad_fn->num_inputs = 1;
        grad_fn->inputs = (Tensor**)malloc(1 * sizeof(Tensor*));
        if (!grad_fn->inputs) {
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        grad_fn->inputs[0] = x;
        grad_fn->saved_data = NULL;

        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

Tensor* op_sqrt(Tensor* x) {
    if (!x) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    int result = rp_sqrt(out->data, x->data, x->size, x->dtype, x->device_id);
    if (result != 0) {
        free_tensor(out);
        return NULL;
    }

    bool requires_grad = (x->metadata && x->metadata->requires_grad);

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

        grad_fn->backward = backward_sqrt_fn;
        grad_fn->num_inputs = 2;
        grad_fn->inputs = (Tensor**)malloc(2 * sizeof(Tensor*));
        if (!grad_fn->inputs) {
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        grad_fn->inputs[0] = x;
        grad_fn->inputs[1] = out;
        grad_fn->saved_data = NULL;

        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

Tensor* op_tanh(Tensor* x) {
    if (!x) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    int result = rp_tanh(out->data, x->data, x->size, x->dtype, x->device_id);
    if (result != 0) {
        free_tensor(out);
        return NULL;
    }

    bool requires_grad = (x->metadata && x->metadata->requires_grad);

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

        grad_fn->backward = backward_tanh_fn;
        grad_fn->num_inputs = 2;
        grad_fn->inputs = (Tensor**)malloc(2 * sizeof(Tensor*));
        if (!grad_fn->inputs) {
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        grad_fn->inputs[0] = x;
        grad_fn->inputs[1] = out;
        grad_fn->saved_data = NULL;

        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

typedef struct {
    int dim;
    int* input_sizes_at_dim;
} CatSavedData;

typedef struct {
    int* start;
    int* stop;
    int* step;
    int ndim;
    int* original_shape;
} SliceSavedData;

void backward_cat_fn(GradFn* self, Tensor* grad_output);
void backward_slice_fn(GradFn* self, Tensor* grad_output);

void free_grad_fn(GradFn* grad_fn) {
    if (!grad_fn) return;
    if (grad_fn->inputs) free(grad_fn->inputs);
    if (grad_fn->saved_data) {
        if (grad_fn->backward == backward_cat_fn) {
            CatSavedData* saved = (CatSavedData*)grad_fn->saved_data;
            if (saved->input_sizes_at_dim) free(saved->input_sizes_at_dim);
        } else if (grad_fn->backward == backward_slice_fn) {
            SliceSavedData* saved = (SliceSavedData*)grad_fn->saved_data;
            if (saved->original_shape) free(saved->original_shape);
            if (saved->start) free(saved->start);
            if (saved->stop) free(saved->stop);
            if (saved->step) free(saved->step);
        }
        free(grad_fn->saved_data);
    }
    free(grad_fn);
}

void backward_cat_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    CatSavedData* saved = (CatSavedData*)self->saved_data;
    if (!saved) return;

    int dim = saved->dim;
    int num_tensors = self->num_inputs;

    int offset = 0;
    for (int i = 0; i < num_tensors; i++) {
        Tensor* input = self->inputs[i];
        if (!input->metadata || !input->metadata->requires_grad) {
            offset += saved->input_sizes_at_dim[i];
            continue;
        }

        int slice_size = saved->input_sizes_at_dim[i];

        if (!input->metadata->grad) {
            input->metadata->grad = zeros_tensor(input->dtype, input->device_id, input->ndim, input->shape, NULL);
        }

        if (input->metadata->grad) {
            size_t elem_size = (input->dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);

            if (input->ndim == 1) {
                void* grad_src = (char*)grad_output->data + offset * elem_size;
                rp_add(input->metadata->grad->data, input->metadata->grad->data, grad_src,
                       slice_size, input->dtype, input->device_id);
            } else {
                int* indices = (int*)calloc(input->ndim, sizeof(int));
                if (!indices) {
                    offset += slice_size;
                    continue;
                }

                size_t total_elements = 1;
                for (int d = 0; d < input->ndim; d++) {
                    if (d == dim) {
                        total_elements *= slice_size;
                    } else {
                        total_elements *= input->shape[d];
                    }
                }

                for (size_t idx = 0; idx < total_elements; idx++) {
                    size_t temp = idx;
                    for (int d = input->ndim - 1; d >= 0; d--) {
                        if (d == dim) {
                            indices[d] = temp % slice_size;
                        } else {
                            indices[d] = temp % input->shape[d];
                        }
                        temp /= (d == dim) ? slice_size : input->shape[d];
                    }

                    size_t grad_idx = 0;
                    int* grad_shape = (int*)malloc(input->ndim * sizeof(int));
                    for (int d = 0; d < input->ndim; d++) {
                        grad_shape[d] = (d == dim) ? grad_output->shape[dim] : grad_output->shape[d];
                    }

                    indices[dim] += offset;
                    for (int d = 0; d < input->ndim; d++) {
                        grad_idx = grad_idx * grad_shape[d] + indices[d];
                    }
                    indices[dim] -= offset;

                    size_t input_idx = 0;
                    for (int d = 0; d < input->ndim; d++) {
                        input_idx = input_idx * input->shape[d] + indices[d];
                    }

                    if (input->dtype == DTYPE_FLOAT32) {
                        ((float*)input->metadata->grad->data)[input_idx] += ((float*)grad_output->data)[grad_idx];
                    } else {
                        ((double*)input->metadata->grad->data)[input_idx] += ((double*)grad_output->data)[grad_idx];
                    }

                    free(grad_shape);
                }

                free(indices);
            }
        }

        offset += slice_size;
    }
}

Tensor* op_cat(Tensor** tensors, int num_tensors, int dim) {
    if (!tensors || num_tensors <= 0) return NULL;

    Tensor* out = rp_cat(tensors, num_tensors, dim);
    if (!out) return NULL;

    bool requires_grad = false;
    for (int i = 0; i < num_tensors; i++) {
        if (tensors[i]->metadata && tensors[i]->metadata->requires_grad) {
            requires_grad = true;
            break;
        }
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

        grad_fn->backward = backward_cat_fn;
        grad_fn->num_inputs = num_tensors;
        grad_fn->inputs = (Tensor**)malloc(num_tensors * sizeof(Tensor*));
        if (!grad_fn->inputs) {
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }

        for (int i = 0; i < num_tensors; i++) {
            grad_fn->inputs[i] = tensors[i];
        }

        CatSavedData* saved = (CatSavedData*)malloc(sizeof(CatSavedData));
        if (!saved) {
            free(grad_fn->inputs);
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }

        saved->dim = dim;
        saved->input_sizes_at_dim = (int*)malloc(num_tensors * sizeof(int));
        if (!saved->input_sizes_at_dim) {
            free(saved);
            free(grad_fn->inputs);
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }

        for (int i = 0; i < num_tensors; i++) {
            saved->input_sizes_at_dim[i] = tensors[i]->shape[dim];
        }

        grad_fn->saved_data = saved;
        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

void backward_slice_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    SliceSavedData* saved = (SliceSavedData*)self->saved_data;
    if (!saved || self->num_inputs != 1) return;

    Tensor* input = self->inputs[0];
    if (!input->metadata || !input->metadata->requires_grad) return;

    if (!input->metadata->grad) {
        input->metadata->grad = zeros_tensor(input->dtype, input->device_id,
                                            saved->ndim, saved->original_shape, NULL);
    }

    if (!input->metadata->grad) return;

    size_t elem_size = (input->dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);

    for (size_t idx = 0; idx < grad_output->size; idx++) {
        int* grad_indices = (int*)malloc(grad_output->ndim * sizeof(int));
        if (!grad_indices) continue;

        size_t temp = idx;
        for (int d = grad_output->ndim - 1; d >= 0; d--) {
            grad_indices[d] = temp % grad_output->shape[d];
            temp /= grad_output->shape[d];
        }

        int* input_indices = (int*)malloc(saved->ndim * sizeof(int));
        if (!input_indices) {
            free(grad_indices);
            continue;
        }

        for (int d = 0; d < saved->ndim; d++) {
            int s = (saved->start && saved->start[d] != INT_MIN) ? saved->start[d] : 0;
            int st = (saved->step && saved->step[d] != 0) ? saved->step[d] : 1;

            if (s < 0) s += saved->original_shape[d];
            if (s < 0) s = 0;
            if (s > saved->original_shape[d]) s = saved->original_shape[d];

            input_indices[d] = s + grad_indices[d] * st;
        }

        size_t input_idx = 0;
        for (int d = 0; d < saved->ndim; d++) {
            input_idx = input_idx * saved->original_shape[d] + input_indices[d];
        }

        if (input->dtype == DTYPE_FLOAT32) {
            ((float*)input->metadata->grad->data)[input_idx] += ((float*)grad_output->data)[idx];
        } else {
            ((double*)input->metadata->grad->data)[input_idx] += ((double*)grad_output->data)[idx];
        }

        free(grad_indices);
        free(input_indices);
    }
}

Tensor* op_slice(Tensor* src, int* start, int* stop, int* step) {
    if (!src) return NULL;

    Tensor* out = rp_slice(src, start, stop, step);
    if (!out) return NULL;

    bool requires_grad = (src->metadata && src->metadata->requires_grad);

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

        grad_fn->backward = backward_slice_fn;
        grad_fn->num_inputs = 1;
        grad_fn->inputs = (Tensor**)malloc(sizeof(Tensor*));
        if (!grad_fn->inputs) {
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        grad_fn->inputs[0] = src;

        SliceSavedData* saved = (SliceSavedData*)malloc(sizeof(SliceSavedData));
        if (!saved) {
            free(grad_fn->inputs);
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }

        saved->ndim = src->ndim;
        saved->original_shape = (int*)malloc(src->ndim * sizeof(int));
        saved->start = (int*)malloc(src->ndim * sizeof(int));
        saved->stop = (int*)malloc(src->ndim * sizeof(int));
        saved->step = (int*)malloc(src->ndim * sizeof(int));

        if (!saved->original_shape || !saved->start || !saved->stop || !saved->step) {
            if (saved->original_shape) free(saved->original_shape);
            if (saved->start) free(saved->start);
            if (saved->stop) free(saved->stop);
            if (saved->step) free(saved->step);
            free(saved);
            free(grad_fn->inputs);
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }

        for (int i = 0; i < src->ndim; i++) {
            saved->original_shape[i] = src->shape[i];
            saved->start[i] = start ? start[i] : 0;
            saved->stop[i] = stop ? stop[i] : src->shape[i];
            saved->step[i] = step ? step[i] : 1;
        }

        grad_fn->saved_data = saved;
        out->metadata->grad_fn = grad_fn;
    }

    return out;
}
