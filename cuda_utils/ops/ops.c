#include "ops.h"
#include "../functional/functional.h"
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>

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

int backwards_power_x1(const void* grad_c, const void* x1, const void* x2, const void* out, void* grad_x1,
                       size_t size, DType dtype, int device_id) {
    if (!grad_c || !x1 || !x2 || !out || !grad_x1) return -1;

    if (device_id == -1) {
        return backwards_power_x1_host(grad_c, x1, x2, out, grad_x1, size, dtype);
    } else {
        return backwards_power_x1_device(grad_c, x1, x2, out, grad_x1, size, dtype);
    }
}

int backwards_power_x2(const void* grad_c, const void* x1, const void* out, void* grad_x2,
                       size_t size, DType dtype, int device_id) {
    if (!grad_c || !x1 || !out || !grad_x2) return -1;

    if (device_id == -1) {
        return backwards_power_x2_host(grad_c, x1, out, grad_x2, size, dtype);
    } else {
        return backwards_power_x2_device(grad_c, x1, out, grad_x2, size, dtype);
    }
}

int backwards_logb_x1(const void* grad_c, const void* x1, const void* x2, void* grad_x1,
                      size_t size, DType dtype, int device_id) {
    if (!grad_c || !x1 || !x2 || !grad_x1) return -1;

    if (device_id == -1) {
        return backwards_logb_x1_host(grad_c, x1, x2, grad_x1, size, dtype);
    } else {
        return backwards_logb_x1_device(grad_c, x1, x2, grad_x1, size, dtype);
    }
}

int backwards_logb_x2(const void* grad_c, const void* x1, const void* x2, void* grad_x2,
                      size_t size, DType dtype, int device_id) {
    if (!grad_c || !x1 || !x2 || !grad_x2) return -1;

    if (device_id == -1) {
        return backwards_logb_x2_host(grad_c, x1, x2, grad_x2, size, dtype);
    } else {
        return backwards_logb_x2_device(grad_c, x1, x2, grad_x2, size, dtype);
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

void backward_power_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x1 = self->inputs[0];
    Tensor* x2 = self->inputs[1];
    Tensor* output = self->inputs[2];

    if (x1->metadata && x1->metadata->requires_grad) {
        if (!x1->metadata->grad) {
            x1->metadata->grad = zeros_tensor(x1->dtype, x1->device_id, x1->ndim, x1->shape, NULL);
            if (x1->metadata->grad) {
                backwards_power_x1(grad_output->data, x1->data, x2->data, output->data, x1->metadata->grad->data,
                                  grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x1->dtype, x1->device_id, x1->ndim, x1->shape, NULL);
            if (temp_grad) {
                backwards_power_x1(grad_output->data, x1->data, x2->data, output->data, temp_grad->data,
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
                backwards_power_x2(grad_output->data, x1->data, output->data, x2->metadata->grad->data,
                                  grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x2->dtype, x2->device_id, x2->ndim, x2->shape, NULL);
            if (temp_grad) {
                backwards_power_x2(grad_output->data, x1->data, output->data, temp_grad->data,
                                  grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x2->metadata->grad->data, x2->metadata->grad->data, temp_grad->data,
                       grad_output->size, grad_output->dtype, grad_output->device_id);
                free_tensor(temp_grad);
            }
        }
    }
}

void backward_logb_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x1 = self->inputs[0];
    Tensor* x2 = self->inputs[1];

    if (x1->metadata && x1->metadata->requires_grad) {
        if (!x1->metadata->grad) {
            x1->metadata->grad = zeros_tensor(x1->dtype, x1->device_id, x1->ndim, x1->shape, NULL);
            if (x1->metadata->grad) {
                backwards_logb_x1(grad_output->data, x1->data, x2->data, x1->metadata->grad->data,
                                 grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x1->dtype, x1->device_id, x1->ndim, x1->shape, NULL);
            if (temp_grad) {
                backwards_logb_x1(grad_output->data, x1->data, x2->data, temp_grad->data,
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
                backwards_logb_x2(grad_output->data, x1->data, x2->data, x2->metadata->grad->data,
                                 grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x2->dtype, x2->device_id, x2->ndim, x2->shape, NULL);
            if (temp_grad) {
                backwards_logb_x2(grad_output->data, x1->data, x2->data, temp_grad->data,
                                 grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x2->metadata->grad->data, x2->metadata->grad->data, temp_grad->data,
                       grad_output->size, grad_output->dtype, grad_output->device_id);
                free_tensor(temp_grad);
            }
        }
    }
}

Tensor* op_power(Tensor* x1, Tensor* x2) {
    if (!x1 || !x2) return NULL;

    if (x1->ndim != x2->ndim || x1->size != x2->size) return NULL;
    for (int i = 0; i < x1->ndim; i++) {
        if (x1->shape[i] != x2->shape[i]) return NULL;
    }

    if (x1->dtype != x2->dtype || x1->device_id != x2->device_id) return NULL;

    Tensor* out = zeros_tensor(x1->dtype, x1->device_id, x1->ndim, x1->shape, NULL);
    if (!out) return NULL;

    int result = rp_power(out->data, x1->data, x2->data, x1->size, x1->dtype, x1->device_id);
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

        grad_fn->backward = backward_power_fn;
        grad_fn->num_inputs = 3;
        grad_fn->inputs = (Tensor**)malloc(3 * sizeof(Tensor*));
        if (!grad_fn->inputs) {
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        grad_fn->inputs[0] = x1;
        grad_fn->inputs[1] = x2;
        grad_fn->inputs[2] = out;
        grad_fn->saved_data = NULL;

        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

Tensor* op_logb(Tensor* x1, Tensor* x2) {
    if (!x1 || !x2) return NULL;

    if (x1->ndim != x2->ndim || x1->size != x2->size) return NULL;
    for (int i = 0; i < x1->ndim; i++) {
        if (x1->shape[i] != x2->shape[i]) return NULL;
    }

    if (x1->dtype != x2->dtype || x1->device_id != x2->device_id) return NULL;

    Tensor* out = zeros_tensor(x1->dtype, x1->device_id, x1->ndim, x1->shape, NULL);
    if (!out) return NULL;

    int result = rp_logb(out->data, x1->data, x2->data, x1->size, x1->dtype, x1->device_id);
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

        grad_fn->backward = backward_logb_fn;
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

int backwards_relu(const void* grad_c, const void* x, void* grad_x,
                   size_t size, DType dtype, int device_id) {
    if (!grad_c || !x || !grad_x) return -1;

    if (device_id == -1) {
        return backwards_relu_host(grad_c, x, grad_x, size, dtype);
    } else {
        return backwards_relu_device(grad_c, x, grad_x, size, dtype);
    }
}

int backwards_sigmoid(const void* grad_c, const void* fn_output, void* grad_x,
                      size_t size, DType dtype, int device_id) {
    if (!grad_c || !fn_output || !grad_x) return -1;

    if (device_id == -1) {
        return backwards_sigmoid_host(grad_c, fn_output, grad_x, size, dtype);
    } else {
        return backwards_sigmoid_device(grad_c, fn_output, grad_x, size, dtype);
    }
}

int backwards_abs(const void* grad_c, const void* x, void* grad_x,
                  size_t size, DType dtype, int device_id) {
    if (!grad_c || !x || !grad_x) return -1;

    if (device_id == -1) {
        return backwards_abs_host(grad_c, x, grad_x, size, dtype);
    } else {
        return backwards_abs_device(grad_c, x, grad_x, size, dtype);
    }
}

int backwards_square(const void* grad_c, const void* x, void* grad_x,
                     size_t size, DType dtype, int device_id) {
    if (!grad_c || !x || !grad_x) return -1;

    if (device_id == -1) {
        return backwards_square_host(grad_c, x, grad_x, size, dtype);
    } else {
        return backwards_square_device(grad_c, x, grad_x, size, dtype);
    }
}

int backwards_sin(const void* grad_c, const void* x, void* grad_x,
                  size_t size, DType dtype, int device_id) {
    if (!grad_c || !x || !grad_x) return -1;

    if (device_id == -1) {
        return backwards_sin_host(grad_c, x, grad_x, size, dtype);
    } else {
        return backwards_sin_device(grad_c, x, grad_x, size, dtype);
    }
}

int backwards_cos(const void* grad_c, const void* x, void* grad_x,
                  size_t size, DType dtype, int device_id) {
    if (!grad_c || !x || !grad_x) return -1;

    if (device_id == -1) {
        return backwards_cos_host(grad_c, x, grad_x, size, dtype);
    } else {
        return backwards_cos_device(grad_c, x, grad_x, size, dtype);
    }
}

int backwards_tan(const void* grad_c, const void* x, void* grad_x,
                  size_t size, DType dtype, int device_id) {
    if (!grad_c || !x || !grad_x) return -1;

    if (device_id == -1) {
        return backwards_tan_host(grad_c, x, grad_x, size, dtype);
    } else {
        return backwards_tan_device(grad_c, x, grad_x, size, dtype);
    }
}

int backwards_asin(const void* grad_c, const void* x, void* grad_x,
                   size_t size, DType dtype, int device_id) {
    if (!grad_c || !x || !grad_x) return -1;

    if (device_id == -1) {
        return backwards_asin_host(grad_c, x, grad_x, size, dtype);
    } else {
        return backwards_asin_device(grad_c, x, grad_x, size, dtype);
    }
}

int backwards_acos(const void* grad_c, const void* x, void* grad_x,
                   size_t size, DType dtype, int device_id) {
    if (!grad_c || !x || !grad_x) return -1;

    if (device_id == -1) {
        return backwards_acos_host(grad_c, x, grad_x, size, dtype);
    } else {
        return backwards_acos_device(grad_c, x, grad_x, size, dtype);
    }
}

int backwards_atan(const void* grad_c, const void* x, void* grad_x,
                   size_t size, DType dtype, int device_id) {
    if (!grad_c || !x || !grad_x) return -1;

    if (device_id == -1) {
        return backwards_atan_host(grad_c, x, grad_x, size, dtype);
    } else {
        return backwards_atan_device(grad_c, x, grad_x, size, dtype);
    }
}

int backwards_sinh(const void* grad_c, const void* x, void* grad_x,
                   size_t size, DType dtype, int device_id) {
    if (!grad_c || !x || !grad_x) return -1;

    if (device_id == -1) {
        return backwards_sinh_host(grad_c, x, grad_x, size, dtype);
    } else {
        return backwards_sinh_device(grad_c, x, grad_x, size, dtype);
    }
}

int backwards_cosh(const void* grad_c, const void* x, void* grad_x,
                   size_t size, DType dtype, int device_id) {
    if (!grad_c || !x || !grad_x) return -1;

    if (device_id == -1) {
        return backwards_cosh_host(grad_c, x, grad_x, size, dtype);
    } else {
        return backwards_cosh_device(grad_c, x, grad_x, size, dtype);
    }
}

int backwards_gelu(const void* grad_c, const void* x, void* grad_x,
                   size_t size, DType dtype, int device_id) {
    if (!grad_c || !x || !grad_x) return -1;

    if (device_id == -1) {
        return backwards_gelu_host(grad_c, x, grad_x, size, dtype);
    } else {
        return backwards_gelu_device(grad_c, x, grad_x, size, dtype);
    }
}

int backwards_silu(const void* grad_c, const void* x, void* grad_x,
                   size_t size, DType dtype, int device_id) {
    if (!grad_c || !x || !grad_x) return -1;

    if (device_id == -1) {
        return backwards_silu_host(grad_c, x, grad_x, size, dtype);
    } else {
        return backwards_silu_device(grad_c, x, grad_x, size, dtype);
    }
}

int backwards_leaky_relu(const void* grad_c, const void* x, const void* alpha, void* grad_x,
                         size_t size, DType dtype, int device_id) {
    if (!grad_c || !x || !alpha || !grad_x) return -1;

    if (device_id == -1) {
        return backwards_leaky_relu_host(grad_c, x, alpha, grad_x, size, dtype);
    } else {
        return backwards_leaky_relu_device(grad_c, x, alpha, grad_x, size, dtype);
    }
}

int backwards_rsqrt(const void* grad_c, const void* x, void* grad_x,
                    size_t size, DType dtype, int device_id) {
    if (!grad_c || !x || !grad_x) return -1;

    if (device_id == -1) {
        return backwards_rsqrt_host(grad_c, x, grad_x, size, dtype);
    } else {
        return backwards_rsqrt_device(grad_c, x, grad_x, size, dtype);
    }
}

int backwards_sum_all(const void* grad_c, void* grad_x,
                      size_t size, DType dtype, int device_id) {
    if (!grad_c || !grad_x) return -1;

    if (device_id == -1) {
        return backwards_sum_all_host(grad_c, grad_x, size, dtype);
    } else {
        return backwards_sum_all_device(grad_c, grad_x, size, dtype);
    }
}

int backwards_mean_all(const void* grad_c, void* grad_x,
                       size_t size, DType dtype, int device_id) {
    if (!grad_c || !grad_x) return -1;

    if (device_id == -1) {
        return backwards_mean_all_host(grad_c, grad_x, size, dtype);
    } else {
        return backwards_mean_all_device(grad_c, grad_x, size, dtype);
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

void backward_relu_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];

    if (x->metadata && x->metadata->requires_grad) {
        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (x->metadata->grad) {
                backwards_relu(grad_output->data, x->data, x->metadata->grad->data,
                              grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (temp_grad) {
                backwards_relu(grad_output->data, x->data, temp_grad->data,
                              grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x->metadata->grad->data, x->metadata->grad->data, temp_grad->data,
                       grad_output->size, grad_output->dtype, grad_output->device_id);
                free_tensor(temp_grad);
            }
        }
    }
}

void backward_sigmoid_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];
    Tensor* output = self->inputs[1];

    if (x->metadata && x->metadata->requires_grad) {
        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (x->metadata->grad) {
                backwards_sigmoid(grad_output->data, output->data, x->metadata->grad->data,
                                 grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (temp_grad) {
                backwards_sigmoid(grad_output->data, output->data, temp_grad->data,
                                 grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x->metadata->grad->data, x->metadata->grad->data, temp_grad->data,
                       grad_output->size, grad_output->dtype, grad_output->device_id);
                free_tensor(temp_grad);
            }
        }
    }
}

void backward_abs_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];

    if (x->metadata && x->metadata->requires_grad) {
        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (x->metadata->grad) {
                backwards_abs(grad_output->data, x->data, x->metadata->grad->data,
                             grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (temp_grad) {
                backwards_abs(grad_output->data, x->data, temp_grad->data,
                             grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x->metadata->grad->data, x->metadata->grad->data, temp_grad->data,
                       grad_output->size, grad_output->dtype, grad_output->device_id);
                free_tensor(temp_grad);
            }
        }
    }
}

void backward_square_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];

    if (x->metadata && x->metadata->requires_grad) {
        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (x->metadata->grad) {
                backwards_square(grad_output->data, x->data, x->metadata->grad->data,
                                grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (temp_grad) {
                backwards_square(grad_output->data, x->data, temp_grad->data,
                                grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x->metadata->grad->data, x->metadata->grad->data, temp_grad->data,
                       grad_output->size, grad_output->dtype, grad_output->device_id);
                free_tensor(temp_grad);
            }
        }
    }
}

void backward_sin_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];

    if (x->metadata && x->metadata->requires_grad) {
        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (x->metadata->grad) {
                backwards_sin(grad_output->data, x->data, x->metadata->grad->data,
                             grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (temp_grad) {
                backwards_sin(grad_output->data, x->data, temp_grad->data,
                             grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x->metadata->grad->data, x->metadata->grad->data, temp_grad->data,
                       grad_output->size, grad_output->dtype, grad_output->device_id);
                free_tensor(temp_grad);
            }
        }
    }
}

void backward_cos_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];

    if (x->metadata && x->metadata->requires_grad) {
        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (x->metadata->grad) {
                backwards_cos(grad_output->data, x->data, x->metadata->grad->data,
                             grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (temp_grad) {
                backwards_cos(grad_output->data, x->data, temp_grad->data,
                             grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x->metadata->grad->data, x->metadata->grad->data, temp_grad->data,
                       grad_output->size, grad_output->dtype, grad_output->device_id);
                free_tensor(temp_grad);
            }
        }
    }
}

void backward_tan_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];

    if (x->metadata && x->metadata->requires_grad) {
        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (x->metadata->grad) {
                backwards_tan(grad_output->data, x->data, x->metadata->grad->data,
                             grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (temp_grad) {
                backwards_tan(grad_output->data, x->data, temp_grad->data,
                             grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x->metadata->grad->data, x->metadata->grad->data, temp_grad->data,
                       grad_output->size, grad_output->dtype, grad_output->device_id);
                free_tensor(temp_grad);
            }
        }
    }
}

void backward_asin_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];

    if (x->metadata && x->metadata->requires_grad) {
        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (x->metadata->grad) {
                backwards_asin(grad_output->data, x->data, x->metadata->grad->data,
                              grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (temp_grad) {
                backwards_asin(grad_output->data, x->data, temp_grad->data,
                              grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x->metadata->grad->data, x->metadata->grad->data, temp_grad->data,
                       grad_output->size, grad_output->dtype, grad_output->device_id);
                free_tensor(temp_grad);
            }
        }
    }
}

void backward_acos_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];

    if (x->metadata && x->metadata->requires_grad) {
        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (x->metadata->grad) {
                backwards_acos(grad_output->data, x->data, x->metadata->grad->data,
                              grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (temp_grad) {
                backwards_acos(grad_output->data, x->data, temp_grad->data,
                              grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x->metadata->grad->data, x->metadata->grad->data, temp_grad->data,
                       grad_output->size, grad_output->dtype, grad_output->device_id);
                free_tensor(temp_grad);
            }
        }
    }
}

void backward_atan_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];

    if (x->metadata && x->metadata->requires_grad) {
        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (x->metadata->grad) {
                backwards_atan(grad_output->data, x->data, x->metadata->grad->data,
                              grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (temp_grad) {
                backwards_atan(grad_output->data, x->data, temp_grad->data,
                              grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x->metadata->grad->data, x->metadata->grad->data, temp_grad->data,
                       grad_output->size, grad_output->dtype, grad_output->device_id);
                free_tensor(temp_grad);
            }
        }
    }
}

void backward_sinh_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];

    if (x->metadata && x->metadata->requires_grad) {
        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (x->metadata->grad) {
                backwards_sinh(grad_output->data, x->data, x->metadata->grad->data,
                              grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (temp_grad) {
                backwards_sinh(grad_output->data, x->data, temp_grad->data,
                              grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x->metadata->grad->data, x->metadata->grad->data, temp_grad->data,
                       grad_output->size, grad_output->dtype, grad_output->device_id);
                free_tensor(temp_grad);
            }
        }
    }
}

void backward_cosh_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];

    if (x->metadata && x->metadata->requires_grad) {
        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (x->metadata->grad) {
                backwards_cosh(grad_output->data, x->data, x->metadata->grad->data,
                              grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (temp_grad) {
                backwards_cosh(grad_output->data, x->data, temp_grad->data,
                              grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x->metadata->grad->data, x->metadata->grad->data, temp_grad->data,
                       grad_output->size, grad_output->dtype, grad_output->device_id);
                free_tensor(temp_grad);
            }
        }
    }
}

void backward_gelu_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];

    if (x->metadata && x->metadata->requires_grad) {
        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (x->metadata->grad) {
                backwards_gelu(grad_output->data, x->data, x->metadata->grad->data,
                              grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (temp_grad) {
                backwards_gelu(grad_output->data, x->data, temp_grad->data,
                              grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x->metadata->grad->data, x->metadata->grad->data, temp_grad->data,
                       grad_output->size, grad_output->dtype, grad_output->device_id);
                free_tensor(temp_grad);
            }
        }
    }
}

void backward_silu_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];

    if (x->metadata && x->metadata->requires_grad) {
        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (x->metadata->grad) {
                backwards_silu(grad_output->data, x->data, x->metadata->grad->data,
                              grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (temp_grad) {
                backwards_silu(grad_output->data, x->data, temp_grad->data,
                              grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x->metadata->grad->data, x->metadata->grad->data, temp_grad->data,
                       grad_output->size, grad_output->dtype, grad_output->device_id);
                free_tensor(temp_grad);
            }
        }
    }
}

typedef struct {
    float alpha_f32;
    double alpha_f64;
} LeakyReluSavedData;

void backward_leaky_relu_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];
    LeakyReluSavedData* saved = (LeakyReluSavedData*)self->saved_data;

    if (x->metadata && x->metadata->requires_grad) {
        void* alpha_ptr = (x->dtype == DTYPE_FLOAT32) ? (void*)&saved->alpha_f32 : (void*)&saved->alpha_f64;

        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (x->metadata->grad) {
                backwards_leaky_relu(grad_output->data, x->data, alpha_ptr, x->metadata->grad->data,
                                    grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (temp_grad) {
                backwards_leaky_relu(grad_output->data, x->data, alpha_ptr, temp_grad->data,
                                    grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x->metadata->grad->data, x->metadata->grad->data, temp_grad->data,
                       grad_output->size, grad_output->dtype, grad_output->device_id);
                free_tensor(temp_grad);
            }
        }
    }
}

void backward_rsqrt_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];

    if (x->metadata && x->metadata->requires_grad) {
        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (x->metadata->grad) {
                backwards_rsqrt(grad_output->data, x->data, x->metadata->grad->data,
                               grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (temp_grad) {
                backwards_rsqrt(grad_output->data, x->data, temp_grad->data,
                               grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x->metadata->grad->data, x->metadata->grad->data, temp_grad->data,
                       grad_output->size, grad_output->dtype, grad_output->device_id);
                free_tensor(temp_grad);
            }
        }
    }
}

void backward_sum_all_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];

    if (x->metadata && x->metadata->requires_grad) {
        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (x->metadata->grad) {
                backwards_sum_all(grad_output->data, x->metadata->grad->data,
                                  x->size, x->dtype, x->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (temp_grad) {
                backwards_sum_all(grad_output->data, temp_grad->data,
                                  x->size, x->dtype, x->device_id);
                rp_add(x->metadata->grad->data, x->metadata->grad->data, temp_grad->data,
                       x->size, x->dtype, x->device_id);
                free_tensor(temp_grad);
            }
        }
    }
}

void backward_mean_all_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];

    if (x->metadata && x->metadata->requires_grad) {
        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (x->metadata->grad) {
                backwards_mean_all(grad_output->data, x->metadata->grad->data,
                                   x->size, x->dtype, x->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (temp_grad) {
                backwards_mean_all(grad_output->data, temp_grad->data,
                                   x->size, x->dtype, x->device_id);
                rp_add(x->metadata->grad->data, x->metadata->grad->data, temp_grad->data,
                       x->size, x->dtype, x->device_id);
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

Tensor* op_relu(Tensor* x) {
    if (!x) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    int result = rp_relu(out->data, x->data, x->size, x->dtype, x->device_id);
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

        grad_fn->backward = backward_relu_fn;
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

Tensor* op_sigmoid(Tensor* x) {
    if (!x) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    int result = rp_sigmoid(out->data, x->data, x->size, x->dtype, x->device_id);
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

        grad_fn->backward = backward_sigmoid_fn;
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

Tensor* op_abs(Tensor* x) {
    if (!x) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    int result = rp_abs(out->data, x->data, x->size, x->dtype, x->device_id);
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

        grad_fn->backward = backward_abs_fn;
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

Tensor* op_square(Tensor* x) {
    if (!x) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    int result = rp_square(out->data, x->data, x->size, x->dtype, x->device_id);
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

        grad_fn->backward = backward_square_fn;
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

Tensor* op_sin(Tensor* x) {
    if (!x) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    int result = rp_sin(out->data, x->data, x->size, x->dtype, x->device_id);
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

        grad_fn->backward = backward_sin_fn;
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

Tensor* op_cos(Tensor* x) {
    if (!x) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    int result = rp_cos(out->data, x->data, x->size, x->dtype, x->device_id);
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

        grad_fn->backward = backward_cos_fn;
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

Tensor* op_tan(Tensor* x) {
    if (!x) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    int result = rp_tan(out->data, x->data, x->size, x->dtype, x->device_id);
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

        grad_fn->backward = backward_tan_fn;
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

Tensor* op_asin(Tensor* x) {
    if (!x) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    int result = rp_asin(out->data, x->data, x->size, x->dtype, x->device_id);
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

        grad_fn->backward = backward_asin_fn;
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

Tensor* op_acos(Tensor* x) {
    if (!x) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    int result = rp_acos(out->data, x->data, x->size, x->dtype, x->device_id);
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

        grad_fn->backward = backward_acos_fn;
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

Tensor* op_atan(Tensor* x) {
    if (!x) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    int result = rp_atan(out->data, x->data, x->size, x->dtype, x->device_id);
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

        grad_fn->backward = backward_atan_fn;
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

Tensor* op_sinh(Tensor* x) {
    if (!x) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    int result = rp_sinh(out->data, x->data, x->size, x->dtype, x->device_id);
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

        grad_fn->backward = backward_sinh_fn;
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

Tensor* op_cosh(Tensor* x) {
    if (!x) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    int result = rp_cosh(out->data, x->data, x->size, x->dtype, x->device_id);
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

        grad_fn->backward = backward_cosh_fn;
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

Tensor* op_gelu(Tensor* x) {
    if (!x) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    int result = rp_gelu(out->data, x->data, x->size, x->dtype, x->device_id);
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

        grad_fn->backward = backward_gelu_fn;
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

Tensor* op_silu(Tensor* x) {
    if (!x) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    int result = rp_silu(out->data, x->data, x->size, x->dtype, x->device_id);
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

        grad_fn->backward = backward_silu_fn;
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

Tensor* op_leaky_relu(Tensor* x, float alpha) {
    if (!x) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    void* alpha_ptr;
    float alpha_f32 = alpha;
    double alpha_f64 = (double)alpha;
    if (x->dtype == DTYPE_FLOAT32) {
        alpha_ptr = &alpha_f32;
    } else {
        alpha_ptr = &alpha_f64;
    }

    int result = rp_leaky_relu(out->data, x->data, alpha_ptr, x->size, x->dtype, x->device_id);
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

        grad_fn->backward = backward_leaky_relu_fn;
        grad_fn->num_inputs = 1;
        grad_fn->inputs = (Tensor**)malloc(1 * sizeof(Tensor*));
        if (!grad_fn->inputs) {
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        grad_fn->inputs[0] = x;

        LeakyReluSavedData* saved = (LeakyReluSavedData*)malloc(sizeof(LeakyReluSavedData));
        if (!saved) {
            free(grad_fn->inputs);
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        saved->alpha_f32 = alpha_f32;
        saved->alpha_f64 = alpha_f64;
        grad_fn->saved_data = saved;

        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

Tensor* op_rsqrt(Tensor* x) {
    if (!x) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    int result = rp_rsqrt(out->data, x->data, x->size, x->dtype, x->device_id);
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

        grad_fn->backward = backward_rsqrt_fn;
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

Tensor* op_sum_all(Tensor* x) {
    if (!x) return NULL;

    int scalar_shape[1] = {1};
    Tensor* out = zeros_tensor(x->dtype, x->device_id, 1, scalar_shape, NULL);
    if (!out) return NULL;

    int result = rp_sum_all(out->data, x->data, x->size, x->dtype, x->device_id);
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

        grad_fn->backward = backward_sum_all_fn;
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

Tensor* op_mean_all(Tensor* x) {
    if (!x) return NULL;

    int scalar_shape[1] = {1};
    Tensor* out = zeros_tensor(x->dtype, x->device_id, 1, scalar_shape, NULL);
    if (!out) return NULL;

    int result = rp_mean_all(out->data, x->data, x->size, x->dtype, x->device_id);
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

        grad_fn->backward = backward_mean_all_fn;
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

void backward_add_scalar_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];

    if (x->metadata && x->metadata->requires_grad) {
        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (x->metadata->grad) {
                backwards_add_x1(grad_output->data, x->metadata->grad->data,
                                grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            rp_add(x->metadata->grad->data, x->metadata->grad->data, grad_output->data,
                   grad_output->size, grad_output->dtype, grad_output->device_id);
        }
    }
}

void backward_sub_scalar_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];

    if (x->metadata && x->metadata->requires_grad) {
        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (x->metadata->grad) {
                backwards_add_x1(grad_output->data, x->metadata->grad->data,
                                grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            rp_add(x->metadata->grad->data, x->metadata->grad->data, grad_output->data,
                   grad_output->size, grad_output->dtype, grad_output->device_id);
        }
    }
}

Tensor* op_add_scalar(Tensor* x, const void* scalar) {
    if (!x || !scalar) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    int result = rp_add_scalar(out->data, x->data, scalar, x->size, x->dtype, x->device_id);
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

        grad_fn->backward = backward_add_scalar_fn;
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

Tensor* op_sub_scalar(Tensor* x, const void* scalar) {
    if (!x || !scalar) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    int result = rp_sub_scalar(out->data, x->data, scalar, x->size, x->dtype, x->device_id);
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

        grad_fn->backward = backward_sub_scalar_fn;
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

typedef struct {
    float scalar_f32;
    double scalar_f64;
} ScalarSavedData;

void backward_mul_scalar_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];
    ScalarSavedData* saved = (ScalarSavedData*)self->saved_data;

    if (x->metadata && x->metadata->requires_grad) {
        void* scalar_ptr = (x->dtype == DTYPE_FLOAT32) ? (void*)&saved->scalar_f32 : (void*)&saved->scalar_f64;

        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (x->metadata->grad) {
                rp_mul_scalar(x->metadata->grad->data, grad_output->data, scalar_ptr,
                             grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (temp) {
                rp_mul_scalar(temp->data, grad_output->data, scalar_ptr,
                             grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x->metadata->grad->data, x->metadata->grad->data, temp->data,
                       x->size, x->dtype, x->device_id);
                free_tensor(temp);
            }
        }
    }
}

void backward_div_scalar_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];
    ScalarSavedData* saved = (ScalarSavedData*)self->saved_data;

    if (x->metadata && x->metadata->requires_grad) {
        void* scalar_ptr = (x->dtype == DTYPE_FLOAT32) ? (void*)&saved->scalar_f32 : (void*)&saved->scalar_f64;

        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (x->metadata->grad) {
                rp_div_scalar(x->metadata->grad->data, grad_output->data, scalar_ptr,
                             grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (temp) {
                rp_div_scalar(temp->data, grad_output->data, scalar_ptr,
                             grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x->metadata->grad->data, x->metadata->grad->data, temp->data,
                       x->size, x->dtype, x->device_id);
                free_tensor(temp);
            }
        }
    }
}

Tensor* op_mul_scalar(Tensor* x, const void* scalar) {
    if (!x || !scalar) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    int result = rp_mul_scalar(out->data, x->data, scalar, x->size, x->dtype, x->device_id);
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

        grad_fn->backward = backward_mul_scalar_fn;
        grad_fn->num_inputs = 1;
        grad_fn->inputs = (Tensor**)malloc(1 * sizeof(Tensor*));
        if (!grad_fn->inputs) {
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        grad_fn->inputs[0] = x;

        ScalarSavedData* saved = (ScalarSavedData*)malloc(sizeof(ScalarSavedData));
        if (!saved) {
            free(grad_fn->inputs);
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        if (x->dtype == DTYPE_FLOAT32) {
            saved->scalar_f32 = *(const float*)scalar;
            saved->scalar_f64 = (double)saved->scalar_f32;
        } else {
            saved->scalar_f64 = *(const double*)scalar;
            saved->scalar_f32 = (float)saved->scalar_f64;
        }
        grad_fn->saved_data = saved;

        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

Tensor* op_div_scalar(Tensor* x, const void* scalar) {
    if (!x || !scalar) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    int result = rp_div_scalar(out->data, x->data, scalar, x->size, x->dtype, x->device_id);
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

        grad_fn->backward = backward_div_scalar_fn;
        grad_fn->num_inputs = 1;
        grad_fn->inputs = (Tensor**)malloc(1 * sizeof(Tensor*));
        if (!grad_fn->inputs) {
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        grad_fn->inputs[0] = x;

        ScalarSavedData* saved = (ScalarSavedData*)malloc(sizeof(ScalarSavedData));
        if (!saved) {
            free(grad_fn->inputs);
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        if (x->dtype == DTYPE_FLOAT32) {
            saved->scalar_f32 = *(const float*)scalar;
            saved->scalar_f64 = (double)saved->scalar_f32;
        } else {
            saved->scalar_f64 = *(const double*)scalar;
            saved->scalar_f32 = (float)saved->scalar_f64;
        }
        grad_fn->saved_data = saved;

        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

void backward_pow_scalar_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];
    ScalarSavedData* saved = (ScalarSavedData*)self->saved_data;

    if (x->metadata && x->metadata->requires_grad) {
        float exp_m1_f32 = saved->scalar_f32 - 1.0f;
        double exp_m1_f64 = saved->scalar_f64 - 1.0;
        void* exp_m1_ptr = (x->dtype == DTYPE_FLOAT32) ? (void*)&exp_m1_f32 : (void*)&exp_m1_f64;
        void* scalar_ptr = (x->dtype == DTYPE_FLOAT32) ? (void*)&saved->scalar_f32 : (void*)&saved->scalar_f64;

        Tensor* x_pow = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
        if (!x_pow) return;
        rp_pow_scalar(x_pow->data, x->data, exp_m1_ptr, x->size, x->dtype, x->device_id);

        Tensor* scaled = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
        if (!scaled) { free_tensor(x_pow); return; }
        rp_mul_scalar(scaled->data, x_pow->data, scalar_ptr, x->size, x->dtype, x->device_id);
        free_tensor(x_pow);

        Tensor* local_grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
        if (!local_grad) { free_tensor(scaled); return; }
        rp_mul(local_grad->data, grad_output->data, scaled->data, x->size, x->dtype, x->device_id);
        free_tensor(scaled);

        if (!x->metadata->grad) {
            x->metadata->grad = local_grad;
        } else {
            rp_add(x->metadata->grad->data, x->metadata->grad->data, local_grad->data,
                   x->size, x->dtype, x->device_id);
            free_tensor(local_grad);
        }
    }
}

Tensor* op_pow_scalar(Tensor* x, const void* scalar) {
    if (!x || !scalar) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    int result = rp_pow_scalar(out->data, x->data, scalar, x->size, x->dtype, x->device_id);
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

        grad_fn->backward = backward_pow_scalar_fn;
        grad_fn->num_inputs = 1;
        grad_fn->inputs = (Tensor**)malloc(1 * sizeof(Tensor*));
        if (!grad_fn->inputs) {
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        grad_fn->inputs[0] = x;

        ScalarSavedData* saved = (ScalarSavedData*)malloc(sizeof(ScalarSavedData));
        if (!saved) {
            free(grad_fn->inputs);
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        if (x->dtype == DTYPE_FLOAT32) {
            saved->scalar_f32 = *(const float*)scalar;
            saved->scalar_f64 = (double)saved->scalar_f32;
        } else {
            saved->scalar_f64 = *(const double*)scalar;
            saved->scalar_f32 = (float)saved->scalar_f64;
        }
        grad_fn->saved_data = saved;

        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

void backward_rsub_scalar_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];

    if (x->metadata && x->metadata->requires_grad) {
        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (x->metadata->grad) {
                backwards_sub_x2(grad_output->data, x->metadata->grad->data,
                                grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            Tensor* temp = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (temp) {
                backwards_sub_x2(grad_output->data, temp->data,
                                grad_output->size, grad_output->dtype, grad_output->device_id);
                rp_add(x->metadata->grad->data, x->metadata->grad->data, temp->data,
                       x->size, x->dtype, x->device_id);
                free_tensor(temp);
            }
        }
    }
}

Tensor* op_rsub_scalar(Tensor* x, const void* scalar) {
    if (!x || !scalar) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    int result = rp_rsub_scalar(out->data, scalar, x->data, x->size, x->dtype, x->device_id);
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

        grad_fn->backward = backward_rsub_scalar_fn;
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

void backward_rdiv_scalar_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];
    ScalarSavedData* saved = (ScalarSavedData*)self->saved_data;

    if (x->metadata && x->metadata->requires_grad) {
        void* scalar_ptr = (x->dtype == DTYPE_FLOAT32) ? (void*)&saved->scalar_f32 : (void*)&saved->scalar_f64;

        Tensor* x_sq = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
        if (!x_sq) return;
        rp_mul(x_sq->data, x->data, x->data, x->size, x->dtype, x->device_id);

        Tensor* neg_c_over_x2 = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
        if (!neg_c_over_x2) { free_tensor(x_sq); return; }
        rp_rdiv_scalar(neg_c_over_x2->data, scalar_ptr, x_sq->data, x->size, x->dtype, x->device_id);
        free_tensor(x_sq);

        float neg1_f32 = -1.0f;
        double neg1_f64 = -1.0;
        void* neg1_ptr = (x->dtype == DTYPE_FLOAT32) ? (void*)&neg1_f32 : (void*)&neg1_f64;
        rp_mul_scalar(neg_c_over_x2->data, neg_c_over_x2->data, neg1_ptr,
                     x->size, x->dtype, x->device_id);

        Tensor* local_grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
        if (!local_grad) { free_tensor(neg_c_over_x2); return; }
        rp_mul(local_grad->data, grad_output->data, neg_c_over_x2->data,
               x->size, x->dtype, x->device_id);
        free_tensor(neg_c_over_x2);

        if (!x->metadata->grad) {
            x->metadata->grad = local_grad;
        } else {
            rp_add(x->metadata->grad->data, x->metadata->grad->data, local_grad->data,
                   x->size, x->dtype, x->device_id);
            free_tensor(local_grad);
        }
    }
}

Tensor* op_rdiv_scalar(Tensor* x, const void* scalar) {
    if (!x || !scalar) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    int result = rp_rdiv_scalar(out->data, scalar, x->data, x->size, x->dtype, x->device_id);
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

        grad_fn->backward = backward_rdiv_scalar_fn;
        grad_fn->num_inputs = 1;
        grad_fn->inputs = (Tensor**)malloc(1 * sizeof(Tensor*));
        if (!grad_fn->inputs) {
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        grad_fn->inputs[0] = x;

        ScalarSavedData* saved = (ScalarSavedData*)malloc(sizeof(ScalarSavedData));
        if (!saved) {
            free(grad_fn->inputs);
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        if (x->dtype == DTYPE_FLOAT32) {
            saved->scalar_f32 = *(const float*)scalar;
            saved->scalar_f64 = (double)saved->scalar_f32;
        } else {
            saved->scalar_f64 = *(const double*)scalar;
            saved->scalar_f32 = (float)saved->scalar_f64;
        }
        grad_fn->saved_data = saved;

        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

void backward_rpow_scalar_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];
    Tensor* output = self->inputs[1];
    ScalarSavedData* saved = (ScalarSavedData*)self->saved_data;

    if (x->metadata && x->metadata->requires_grad) {
        float ln_c_f32 = logf(saved->scalar_f32);
        double ln_c_f64 = log(saved->scalar_f64);
        void* ln_c_ptr = (x->dtype == DTYPE_FLOAT32) ? (void*)&ln_c_f32 : (void*)&ln_c_f64;

        Tensor* out_ln_c = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
        if (!out_ln_c) return;
        rp_mul_scalar(out_ln_c->data, output->data, ln_c_ptr, x->size, x->dtype, x->device_id);

        Tensor* local_grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
        if (!local_grad) { free_tensor(out_ln_c); return; }
        rp_mul(local_grad->data, grad_output->data, out_ln_c->data, x->size, x->dtype, x->device_id);
        free_tensor(out_ln_c);

        if (!x->metadata->grad) {
            x->metadata->grad = local_grad;
        } else {
            rp_add(x->metadata->grad->data, x->metadata->grad->data, local_grad->data,
                   x->size, x->dtype, x->device_id);
            free_tensor(local_grad);
        }
    }
}

Tensor* op_rpow_scalar(Tensor* x, const void* scalar) {
    if (!x || !scalar) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    int result = rp_rpow_scalar(out->data, scalar, x->data, x->size, x->dtype, x->device_id);
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

        grad_fn->backward = backward_rpow_scalar_fn;
        grad_fn->num_inputs = 2;
        grad_fn->inputs = (Tensor**)malloc(2 * sizeof(Tensor*));
        if (!grad_fn->inputs) {
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        grad_fn->inputs[0] = x;
        grad_fn->inputs[1] = out;

        ScalarSavedData* saved = (ScalarSavedData*)malloc(sizeof(ScalarSavedData));
        if (!saved) {
            free(grad_fn->inputs);
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        if (x->dtype == DTYPE_FLOAT32) {
            saved->scalar_f32 = *(const float*)scalar;
            saved->scalar_f64 = (double)saved->scalar_f32;
        } else {
            saved->scalar_f64 = *(const double*)scalar;
            saved->scalar_f32 = (float)saved->scalar_f64;
        }
        grad_fn->saved_data = saved;

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

typedef struct {
    int ndim;
    int* original_shape;
} ReshapeSavedData;

typedef struct {
    int dim0;
    int dim1;
} TransposeSavedData;

typedef struct {
    int dim;
} SqueezeSavedData;

typedef struct {
    int dim;
} UnsqueezeSavedData;

typedef struct {
    int ndim;
    int* original_shape;
} FlattenSavedData;

typedef struct {
    int ndim;
    int* dims;
} PermuteSavedData;

typedef struct {
    int input_ndim;
    int* input_shape;
} ExpandSavedData;

typedef struct {
    int dim;
    int offset;
    int input_ndim;
    int* input_shape;
} ChunkSavedData;

typedef struct {
    int ndim;
    int* repeats;
    int* input_shape;
} RepeatSavedData;

void backward_cat_fn(GradFn* self, Tensor* grad_output);
void backward_permute_fn(GradFn* self, Tensor* grad_output);
void backward_expand_fn(GradFn* self, Tensor* grad_output);
void backward_chunk_fn(GradFn* self, Tensor* grad_output);
void backward_repeat_fn(GradFn* self, Tensor* grad_output);
void backward_slice_fn(GradFn* self, Tensor* grad_output);
void backward_reshape_fn(GradFn* self, Tensor* grad_output);
void backward_transpose_fn(GradFn* self, Tensor* grad_output);
void backward_squeeze_fn(GradFn* self, Tensor* grad_output);
void backward_unsqueeze_fn(GradFn* self, Tensor* grad_output);
void backward_flatten_fn(GradFn* self, Tensor* grad_output);

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
        } else if (grad_fn->backward == backward_reshape_fn) {
            ReshapeSavedData* saved = (ReshapeSavedData*)grad_fn->saved_data;
            if (saved->original_shape) free(saved->original_shape);
        } else if (grad_fn->backward == backward_flatten_fn) {
            FlattenSavedData* saved = (FlattenSavedData*)grad_fn->saved_data;
            if (saved->original_shape) free(saved->original_shape);
        } else if (grad_fn->backward == backward_permute_fn) {
            PermuteSavedData* saved = (PermuteSavedData*)grad_fn->saved_data;
            if (saved->dims) free(saved->dims);
        } else if (grad_fn->backward == backward_expand_fn) {
            ExpandSavedData* saved = (ExpandSavedData*)grad_fn->saved_data;
            if (saved->input_shape) free(saved->input_shape);
        } else if (grad_fn->backward == backward_chunk_fn) {
            ChunkSavedData* saved = (ChunkSavedData*)grad_fn->saved_data;
            if (saved->input_shape) free(saved->input_shape);
        } else if (grad_fn->backward == backward_repeat_fn) {
            RepeatSavedData* saved = (RepeatSavedData*)grad_fn->saved_data;
            if (saved->repeats) free(saved->repeats);
            if (saved->input_shape) free(saved->input_shape);
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

void backward_view_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    ReshapeSavedData* saved = (ReshapeSavedData*)self->saved_data;
    if (!saved || self->num_inputs != 1) return;

    Tensor* input = self->inputs[0];
    if (!input || !input->metadata || !input->metadata->requires_grad) return;

    Tensor* grad_reshaped = rp_view(grad_output, saved->ndim, saved->original_shape);
    if (!grad_reshaped) return;

    if (!input->metadata->grad) {
        input->metadata->grad = grad_reshaped;
    } else {
        if (input->dtype == DTYPE_FLOAT32) {
            rp_add(input->metadata->grad->data, input->metadata->grad->data,
                   grad_reshaped->data, input->size, input->dtype, input->device_id);
        } else {
            rp_add(input->metadata->grad->data, input->metadata->grad->data,
                   grad_reshaped->data, input->size, input->dtype, input->device_id);
        }
        free_tensor(grad_reshaped);
    }
}

void backward_reshape_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    ReshapeSavedData* saved = (ReshapeSavedData*)self->saved_data;
    if (!saved || self->num_inputs != 1) return;

    Tensor* input = self->inputs[0];
    if (!input || !input->metadata || !input->metadata->requires_grad) return;

    Tensor* grad_reshaped = rp_reshape(grad_output, saved->ndim, saved->original_shape);
    if (!grad_reshaped) return;

    if (!input->metadata->grad) {
        input->metadata->grad = grad_reshaped;
    } else {
        if (input->dtype == DTYPE_FLOAT32) {
            rp_add(input->metadata->grad->data, input->metadata->grad->data,
                   grad_reshaped->data, input->size, input->dtype, input->device_id);
        } else {
            rp_add(input->metadata->grad->data, input->metadata->grad->data,
                   grad_reshaped->data, input->size, input->dtype, input->device_id);
        }
        free_tensor(grad_reshaped);
    }
}

Tensor* op_view(Tensor* src, int ndim, int* new_shape) {
    if (!src || !new_shape) return NULL;

    Tensor* out = rp_view(src, ndim, new_shape);
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

        grad_fn->backward = backward_view_fn;
        grad_fn->num_inputs = 1;
        grad_fn->inputs = (Tensor**)malloc(sizeof(Tensor*));
        if (!grad_fn->inputs) {
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        grad_fn->inputs[0] = src;

        ReshapeSavedData* saved = (ReshapeSavedData*)malloc(sizeof(ReshapeSavedData));
        if (!saved) {
            free(grad_fn->inputs);
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }

        saved->ndim = src->ndim;
        saved->original_shape = (int*)malloc(src->ndim * sizeof(int));
        if (!saved->original_shape) {
            free(saved);
            free(grad_fn->inputs);
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }

        for (int i = 0; i < src->ndim; i++) {
            saved->original_shape[i] = src->shape[i];
        }

        grad_fn->saved_data = saved;
        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

Tensor* op_reshape(Tensor* src, int ndim, int* new_shape) {
    if (!src || !new_shape) return NULL;

    Tensor* out = rp_reshape(src, ndim, new_shape);
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

        grad_fn->backward = backward_reshape_fn;
        grad_fn->num_inputs = 1;
        grad_fn->inputs = (Tensor**)malloc(sizeof(Tensor*));
        if (!grad_fn->inputs) {
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        grad_fn->inputs[0] = src;

        ReshapeSavedData* saved = (ReshapeSavedData*)malloc(sizeof(ReshapeSavedData));
        if (!saved) {
            free(grad_fn->inputs);
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }

        saved->ndim = src->ndim;
        saved->original_shape = (int*)malloc(src->ndim * sizeof(int));
        if (!saved->original_shape) {
            free(saved);
            free(grad_fn->inputs);
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }

        for (int i = 0; i < src->ndim; i++) {
            saved->original_shape[i] = src->shape[i];
        }

        grad_fn->saved_data = saved;
        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

void backward_transpose_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    TransposeSavedData* saved = (TransposeSavedData*)self->saved_data;
    if (!saved || self->num_inputs != 1) return;

    Tensor* input = self->inputs[0];
    if (!input || !input->metadata || !input->metadata->requires_grad) return;

    Tensor* grad_transposed = rp_transpose(grad_output, saved->dim0, saved->dim1);
    if (!grad_transposed) return;

    if (!input->metadata->grad) {
        input->metadata->grad = grad_transposed;
    } else {
        rp_add(input->metadata->grad->data, input->metadata->grad->data,
               grad_transposed->data, input->size, input->dtype, input->device_id);
        free_tensor(grad_transposed);
    }
}

void backward_squeeze_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    SqueezeSavedData* saved = (SqueezeSavedData*)self->saved_data;
    if (!saved || self->num_inputs != 1) return;

    Tensor* input = self->inputs[0];
    if (!input || !input->metadata || !input->metadata->requires_grad) return;

    Tensor* grad_unsqueezed = rp_unsqueeze(grad_output, saved->dim);
    if (!grad_unsqueezed) return;

    if (!input->metadata->grad) {
        input->metadata->grad = grad_unsqueezed;
    } else {
        rp_add(input->metadata->grad->data, input->metadata->grad->data,
               grad_unsqueezed->data, input->size, input->dtype, input->device_id);
        free_tensor(grad_unsqueezed);
    }
}

void backward_unsqueeze_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    UnsqueezeSavedData* saved = (UnsqueezeSavedData*)self->saved_data;
    if (!saved || self->num_inputs != 1) return;

    Tensor* input = self->inputs[0];
    if (!input || !input->metadata || !input->metadata->requires_grad) return;

    Tensor* grad_squeezed = rp_squeeze(grad_output, saved->dim);
    if (!grad_squeezed) return;

    if (!input->metadata->grad) {
        input->metadata->grad = grad_squeezed;
    } else {
        rp_add(input->metadata->grad->data, input->metadata->grad->data,
               grad_squeezed->data, input->size, input->dtype, input->device_id);
        free_tensor(grad_squeezed);
    }
}

void backward_flatten_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    FlattenSavedData* saved = (FlattenSavedData*)self->saved_data;
    if (!saved || self->num_inputs != 1) return;

    Tensor* input = self->inputs[0];
    if (!input || !input->metadata || !input->metadata->requires_grad) return;

    Tensor* grad_reshaped = rp_reshape(grad_output, saved->ndim, saved->original_shape);
    if (!grad_reshaped) return;

    if (!input->metadata->grad) {
        input->metadata->grad = grad_reshaped;
    } else {
        rp_add(input->metadata->grad->data, input->metadata->grad->data,
               grad_reshaped->data, input->size, input->dtype, input->device_id);
        free_tensor(grad_reshaped);
    }
}

void backward_permute_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    PermuteSavedData* saved = (PermuteSavedData*)self->saved_data;
    if (!saved || self->num_inputs != 1) return;

    Tensor* input = self->inputs[0];
    if (!input || !input->metadata || !input->metadata->requires_grad) return;

    int* inverse_dims = (int*)malloc(saved->ndim * sizeof(int));
    if (!inverse_dims) return;

    for (int i = 0; i < saved->ndim; i++) {
        inverse_dims[saved->dims[i]] = i;
    }

    Tensor* grad_permuted = rp_permute(grad_output, inverse_dims);
    free(inverse_dims);

    if (!grad_permuted) return;

    if (!input->metadata->grad) {
        input->metadata->grad = grad_permuted;
    } else {
        rp_add(input->metadata->grad->data, input->metadata->grad->data,
               grad_permuted->data, input->size, input->dtype, input->device_id);
        free_tensor(grad_permuted);
    }
}

void backward_expand_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    ExpandSavedData* saved = (ExpandSavedData*)self->saved_data;
    if (!saved || self->num_inputs != 1) return;

    Tensor* input = self->inputs[0];
    if (!input || !input->metadata || !input->metadata->requires_grad) return;

    int target_device = input->device_id;
    Tensor* grad_out_host = grad_output;
    bool need_free_grad_out = false;

    if (grad_output->device_id != -1) {
        grad_out_host = tensor_to(grad_output, -1, grad_output->dtype, false);
        if (!grad_out_host) return;
        need_free_grad_out = true;
    }

    Tensor* grad_input = zeros_tensor(input->dtype, -1,
                                       saved->input_ndim, saved->input_shape, NULL);
    if (!grad_input) {
        if (need_free_grad_out) free_tensor(grad_out_host);
        return;
    }

    int* input_strides = (int*)malloc(saved->input_ndim * sizeof(int));
    if (!input_strides) {
        free_tensor(grad_input);
        if (need_free_grad_out) free_tensor(grad_out_host);
        return;
    }
    input_strides[saved->input_ndim - 1] = 1;
    for (int d = saved->input_ndim - 2; d >= 0; d--) {
        input_strides[d] = input_strides[d + 1] * saved->input_shape[d + 1];
    }

    int* output_strides = (int*)malloc(grad_out_host->ndim * sizeof(int));
    if (!output_strides) {
        free(input_strides);
        free_tensor(grad_input);
        if (need_free_grad_out) free_tensor(grad_out_host);
        return;
    }
    output_strides[grad_out_host->ndim - 1] = 1;
    for (int d = grad_out_host->ndim - 2; d >= 0; d--) {
        output_strides[d] = output_strides[d + 1] * grad_out_host->shape[d + 1];
    }

    size_t output_size = grad_out_host->size;
    int ndim = grad_out_host->ndim;

    if (input->dtype == DTYPE_FLOAT32) {
        float* grad_in = (float*)grad_input->data;
        const float* grad_out = (const float*)grad_out_host->data;

        for (size_t i = 0; i < output_size; i++) {
            size_t out_idx = i;
            size_t in_idx = 0;

            for (int d = 0; d < ndim; d++) {
                int coord = out_idx / output_strides[d];
                out_idx %= output_strides[d];

                int in_coord = (saved->input_shape[d] == 1) ? 0 : coord;
                in_idx += in_coord * input_strides[d];
            }

            grad_in[in_idx] += grad_out[i];
        }
    } else {
        double* grad_in = (double*)grad_input->data;
        const double* grad_out = (const double*)grad_out_host->data;

        for (size_t i = 0; i < output_size; i++) {
            size_t out_idx = i;
            size_t in_idx = 0;

            for (int d = 0; d < ndim; d++) {
                int coord = out_idx / output_strides[d];
                out_idx %= output_strides[d];

                int in_coord = (saved->input_shape[d] == 1) ? 0 : coord;
                in_idx += in_coord * input_strides[d];
            }

            grad_in[in_idx] += grad_out[i];
        }
    }

    free(input_strides);
    free(output_strides);
    if (need_free_grad_out) free_tensor(grad_out_host);

    if (target_device != -1) {
        Tensor* grad_input_device = tensor_to(grad_input, target_device, grad_input->dtype, false);
        free_tensor(grad_input);
        if (!grad_input_device) return;
        grad_input = grad_input_device;
    }

    if (!input->metadata->grad) {
        input->metadata->grad = grad_input;
    } else {
        rp_add(input->metadata->grad->data, input->metadata->grad->data,
               grad_input->data, input->size, input->dtype, input->device_id);
        free_tensor(grad_input);
    }
}

void backward_chunk_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    ChunkSavedData* saved = (ChunkSavedData*)self->saved_data;
    if (!saved || self->num_inputs != 1) return;

    Tensor* input = self->inputs[0];
    if (!input || !input->metadata || !input->metadata->requires_grad) return;

    int dim = saved->dim;
    int offset = saved->offset;
    int chunk_size = grad_output->shape[dim];

    if (!input->metadata->grad) {
        input->metadata->grad = zeros_tensor(input->dtype, input->device_id,
                                              saved->input_ndim, saved->input_shape, NULL);
        if (!input->metadata->grad) return;
    }

    int ndim = saved->input_ndim;
    size_t before_size = 1;
    size_t after_size = 1;
    for (int d = 0; d < dim; d++) {
        before_size *= saved->input_shape[d];
    }
    for (int d = dim + 1; d < ndim; d++) {
        after_size *= saved->input_shape[d];
    }

    size_t input_dim_size = saved->input_shape[dim];
    size_t grad_size = grad_output->size;

    if (input->dtype == DTYPE_FLOAT32) {
        float* grad_in = (float*)input->metadata->grad->data;
        const float* grad_out = (const float*)grad_output->data;

        for (size_t i = 0; i < grad_size; i++) {
            size_t after_idx = i % after_size;
            size_t chunk_idx = (i / after_size) % chunk_size;
            size_t before_idx = i / (after_size * chunk_size);

            size_t input_idx = before_idx * (input_dim_size * after_size) +
                              (offset + chunk_idx) * after_size + after_idx;

            grad_in[input_idx] += grad_out[i];
        }
    } else {
        double* grad_in = (double*)input->metadata->grad->data;
        const double* grad_out = (const double*)grad_output->data;

        for (size_t i = 0; i < grad_size; i++) {
            size_t after_idx = i % after_size;
            size_t chunk_idx = (i / after_size) % chunk_size;
            size_t before_idx = i / (after_size * chunk_size);

            size_t input_idx = before_idx * (input_dim_size * after_size) +
                              (offset + chunk_idx) * after_size + after_idx;

            grad_in[input_idx] += grad_out[i];
        }
    }
}

void backward_repeat_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    RepeatSavedData* saved = (RepeatSavedData*)self->saved_data;
    if (!saved || self->num_inputs != 1) return;

    Tensor* input = self->inputs[0];
    if (!input || !input->metadata || !input->metadata->requires_grad) return;

    int ndim = saved->ndim;
    int target_device = input->device_id;

    Tensor* grad_out_host = grad_output;
    bool need_free_grad_out = false;
    if (grad_output->device_id != -1) {
        grad_out_host = tensor_to(grad_output, -1, grad_output->dtype, false);
        if (!grad_out_host) return;
        need_free_grad_out = true;
    }

    Tensor* grad_input = zeros_tensor(input->dtype, -1, ndim, saved->input_shape, NULL);
    if (!grad_input) {
        if (need_free_grad_out) free_tensor(grad_out_host);
        return;
    }

    int* input_strides = (int*)malloc(ndim * sizeof(int));
    int* output_strides = (int*)malloc(ndim * sizeof(int));
    int* output_shape = (int*)malloc(ndim * sizeof(int));
    if (!input_strides || !output_strides || !output_shape) {
        if (input_strides) free(input_strides);
        if (output_strides) free(output_strides);
        if (output_shape) free(output_shape);
        free_tensor(grad_input);
        if (need_free_grad_out) free_tensor(grad_out_host);
        return;
    }

    for (int d = 0; d < ndim; d++) {
        output_shape[d] = saved->input_shape[d] * saved->repeats[d];
    }

    input_strides[ndim - 1] = 1;
    output_strides[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; d--) {
        input_strides[d] = input_strides[d + 1] * saved->input_shape[d + 1];
        output_strides[d] = output_strides[d + 1] * output_shape[d + 1];
    }

    size_t output_size = grad_out_host->size;

    if (input->dtype == DTYPE_FLOAT32) {
        float* grad_in = (float*)grad_input->data;
        const float* grad_out = (const float*)grad_out_host->data;

        for (size_t i = 0; i < output_size; i++) {
            size_t out_idx = i;
            size_t in_idx = 0;

            for (int d = 0; d < ndim; d++) {
                int coord = out_idx / output_strides[d];
                out_idx %= output_strides[d];
                int in_coord = coord % saved->input_shape[d];
                in_idx += in_coord * input_strides[d];
            }

            grad_in[in_idx] += grad_out[i];
        }
    } else {
        double* grad_in = (double*)grad_input->data;
        const double* grad_out = (const double*)grad_out_host->data;

        for (size_t i = 0; i < output_size; i++) {
            size_t out_idx = i;
            size_t in_idx = 0;

            for (int d = 0; d < ndim; d++) {
                int coord = out_idx / output_strides[d];
                out_idx %= output_strides[d];
                int in_coord = coord % saved->input_shape[d];
                in_idx += in_coord * input_strides[d];
            }

            grad_in[in_idx] += grad_out[i];
        }
    }

    free(input_strides);
    free(output_strides);
    free(output_shape);
    if (need_free_grad_out) free_tensor(grad_out_host);

    if (target_device != -1) {
        Tensor* grad_input_device = tensor_to(grad_input, target_device, grad_input->dtype, false);
        free_tensor(grad_input);
        if (!grad_input_device) return;
        grad_input = grad_input_device;
    }

    if (!input->metadata->grad) {
        input->metadata->grad = grad_input;
    } else {
        rp_add(input->metadata->grad->data, input->metadata->grad->data,
               grad_input->data, input->size, input->dtype, input->device_id);
        free_tensor(grad_input);
    }
}

Tensor* op_transpose(Tensor* src, int dim0, int dim1) {
    if (!src) return NULL;

    Tensor* out = rp_transpose(src, dim0, dim1);
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

        grad_fn->backward = backward_transpose_fn;
        grad_fn->num_inputs = 1;
        grad_fn->inputs = (Tensor**)malloc(sizeof(Tensor*));
        if (!grad_fn->inputs) {
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        grad_fn->inputs[0] = src;

        TransposeSavedData* saved = (TransposeSavedData*)malloc(sizeof(TransposeSavedData));
        if (!saved) {
            free(grad_fn->inputs);
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }

        if (dim0 < 0) dim0 += src->ndim;
        if (dim1 < 0) dim1 += src->ndim;

        saved->dim0 = dim0;
        saved->dim1 = dim1;

        grad_fn->saved_data = saved;
        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

Tensor* op_squeeze(Tensor* src, int dim) {
    if (!src) return NULL;

    int actual_dim = dim;
    if (dim >= 0 && dim < src->ndim) {
        actual_dim = dim;
    } else if (dim < 0 && dim >= -src->ndim) {
        actual_dim = dim + src->ndim;
    }

    Tensor* out = rp_squeeze(src, dim);
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

        grad_fn->backward = backward_squeeze_fn;
        grad_fn->num_inputs = 1;
        grad_fn->inputs = (Tensor**)malloc(sizeof(Tensor*));
        if (!grad_fn->inputs) {
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        grad_fn->inputs[0] = src;

        SqueezeSavedData* saved = (SqueezeSavedData*)malloc(sizeof(SqueezeSavedData));
        if (!saved) {
            free(grad_fn->inputs);
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }

        saved->dim = actual_dim;
        grad_fn->saved_data = saved;
        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

Tensor* op_unsqueeze(Tensor* src, int dim) {
    if (!src) return NULL;

    int actual_dim = dim;
    if (dim >= 0 && dim <= src->ndim) {
        actual_dim = dim;
    } else if (dim < 0 && dim >= -(src->ndim + 1)) {
        actual_dim = dim + src->ndim + 1;
    }

    Tensor* out = rp_unsqueeze(src, dim);
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

        grad_fn->backward = backward_unsqueeze_fn;
        grad_fn->num_inputs = 1;
        grad_fn->inputs = (Tensor**)malloc(sizeof(Tensor*));
        if (!grad_fn->inputs) {
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        grad_fn->inputs[0] = src;

        UnsqueezeSavedData* saved = (UnsqueezeSavedData*)malloc(sizeof(UnsqueezeSavedData));
        if (!saved) {
            free(grad_fn->inputs);
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }

        saved->dim = actual_dim;
        grad_fn->saved_data = saved;
        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

Tensor* op_flatten(Tensor* src, int start_dim, int end_dim) {
    if (!src) return NULL;

    Tensor* out = rp_flatten(src, start_dim, end_dim);
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

        grad_fn->backward = backward_flatten_fn;
        grad_fn->num_inputs = 1;
        grad_fn->inputs = (Tensor**)malloc(sizeof(Tensor*));
        if (!grad_fn->inputs) {
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        grad_fn->inputs[0] = src;

        FlattenSavedData* saved = (FlattenSavedData*)malloc(sizeof(FlattenSavedData));
        if (!saved) {
            free(grad_fn->inputs);
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }

        saved->ndim = src->ndim;
        saved->original_shape = (int*)malloc(src->ndim * sizeof(int));
        if (!saved->original_shape) {
            free(saved);
            free(grad_fn->inputs);
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }

        for (int i = 0; i < src->ndim; i++) {
            saved->original_shape[i] = src->shape[i];
        }

        grad_fn->saved_data = saved;
        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

Tensor* op_permute(Tensor* src, int* dims) {
    if (!src) return NULL;

    Tensor* out = rp_permute(src, dims);
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

        grad_fn->backward = backward_permute_fn;
        grad_fn->num_inputs = 1;
        grad_fn->inputs = (Tensor**)malloc(sizeof(Tensor*));
        if (!grad_fn->inputs) {
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        grad_fn->inputs[0] = src;

        PermuteSavedData* saved = (PermuteSavedData*)malloc(sizeof(PermuteSavedData));
        if (!saved) {
            free(grad_fn->inputs);
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        saved->ndim = src->ndim;
        saved->dims = (int*)malloc(src->ndim * sizeof(int));
        if (!saved->dims) {
            free(saved);
            free(grad_fn->inputs);
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        memcpy(saved->dims, dims, src->ndim * sizeof(int));
        grad_fn->saved_data = saved;

        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

Tensor** op_chunk(Tensor* src, int chunks, int dim) {
    if (!src) return NULL;

    Tensor** result = rp_chunk(src, chunks, dim);
    if (!result) return NULL;

    bool requires_grad = (src->metadata && src->metadata->requires_grad);

    if (requires_grad) {
        int actual_chunks = 0;
        while (result[actual_chunks] != NULL) {
            actual_chunks++;
        }

        int offset = 0;
        for (int i = 0; i < actual_chunks; i++) {
            GradFn* grad_fn = (GradFn*)calloc(1, sizeof(GradFn));
            if (!grad_fn) {
                for (int j = 0; j < actual_chunks; j++) {
                    if (result[j]->metadata && result[j]->metadata->grad_fn) {
                        free_grad_fn(result[j]->metadata->grad_fn);
                    }
                    free_tensor(result[j]);
                }
                free(result);
                return NULL;
            }

            grad_fn->backward = backward_chunk_fn;
            grad_fn->num_inputs = 1;
            grad_fn->inputs = (Tensor**)malloc(sizeof(Tensor*));
            if (!grad_fn->inputs) {
                free(grad_fn);
                for (int j = 0; j < actual_chunks; j++) {
                    if (result[j]->metadata && result[j]->metadata->grad_fn) {
                        free_grad_fn(result[j]->metadata->grad_fn);
                    }
                    free_tensor(result[j]);
                }
                free(result);
                return NULL;
            }
            grad_fn->inputs[0] = src;

            ChunkSavedData* saved = (ChunkSavedData*)malloc(sizeof(ChunkSavedData));
            if (!saved) {
                free(grad_fn->inputs);
                free(grad_fn);
                for (int j = 0; j < actual_chunks; j++) {
                    if (result[j]->metadata && result[j]->metadata->grad_fn) {
                        free_grad_fn(result[j]->metadata->grad_fn);
                    }
                    free_tensor(result[j]);
                }
                free(result);
                return NULL;
            }

            saved->dim = dim;
            saved->offset = offset;
            saved->input_ndim = src->ndim;
            saved->input_shape = (int*)malloc(src->ndim * sizeof(int));
            if (!saved->input_shape) {
                free(saved);
                free(grad_fn->inputs);
                free(grad_fn);
                for (int j = 0; j < actual_chunks; j++) {
                    if (result[j]->metadata && result[j]->metadata->grad_fn) {
                        free_grad_fn(result[j]->metadata->grad_fn);
                    }
                    free_tensor(result[j]);
                }
                free(result);
                return NULL;
            }
            memcpy(saved->input_shape, src->shape, src->ndim * sizeof(int));
            grad_fn->saved_data = saved;

            if (!result[i]->metadata) {
                result[i]->metadata = (Meta*)calloc(1, sizeof(Meta));
                if (!result[i]->metadata) {
                    free(saved->input_shape);
                    free(saved);
                    free(grad_fn->inputs);
                    free(grad_fn);
                    for (int j = 0; j < actual_chunks; j++) {
                        if (result[j]->metadata && result[j]->metadata->grad_fn) {
                            free_grad_fn(result[j]->metadata->grad_fn);
                        }
                        free_tensor(result[j]);
                    }
                    free(result);
                    return NULL;
                }
            }
            result[i]->metadata->requires_grad = true;
            result[i]->metadata->is_leaf = false;
            result[i]->metadata->grad_fn = grad_fn;

            offset += result[i]->shape[dim];
        }
    }

    return result;
}

Tensor* op_expand(Tensor* src, int ndim, int* shape) {
    if (!src) return NULL;

    Tensor* out = rp_expand(src, ndim, shape);
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

        grad_fn->backward = backward_expand_fn;
        grad_fn->num_inputs = 1;
        grad_fn->inputs = (Tensor**)malloc(sizeof(Tensor*));
        if (!grad_fn->inputs) {
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        grad_fn->inputs[0] = src;

        ExpandSavedData* saved = (ExpandSavedData*)malloc(sizeof(ExpandSavedData));
        if (!saved) {
            free(grad_fn->inputs);
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        saved->input_ndim = src->ndim;
        saved->input_shape = (int*)malloc(src->ndim * sizeof(int));
        if (!saved->input_shape) {
            free(saved);
            free(grad_fn->inputs);
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        memcpy(saved->input_shape, src->shape, src->ndim * sizeof(int));
        grad_fn->saved_data = saved;

        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

Tensor* op_repeat(Tensor* src, int* repeats) {
    if (!src) return NULL;

    Tensor* out = rp_repeat(src, repeats);
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

        grad_fn->backward = backward_repeat_fn;
        grad_fn->num_inputs = 1;
        grad_fn->inputs = (Tensor**)malloc(sizeof(Tensor*));
        if (!grad_fn->inputs) {
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        grad_fn->inputs[0] = src;

        RepeatSavedData* saved = (RepeatSavedData*)malloc(sizeof(RepeatSavedData));
        if (!saved) {
            free(grad_fn->inputs);
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        saved->ndim = src->ndim;
        saved->repeats = (int*)malloc(src->ndim * sizeof(int));
        saved->input_shape = (int*)malloc(src->ndim * sizeof(int));
        if (!saved->repeats || !saved->input_shape) {
            if (saved->repeats) free(saved->repeats);
            if (saved->input_shape) free(saved->input_shape);
            free(saved);
            free(grad_fn->inputs);
            free(grad_fn);
            free_tensor(out);
            return NULL;
        }
        memcpy(saved->repeats, repeats, src->ndim * sizeof(int));
        memcpy(saved->input_shape, src->shape, src->ndim * sizeof(int));
        grad_fn->saved_data = saved;

        out->metadata->grad_fn = grad_fn;
    }

    return out;
}
