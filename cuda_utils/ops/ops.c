#include "ops.h"
#include "../functional/functional.h"
#include "../core/half_precision.h"
#include <cuda_runtime.h>
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
    BroadcastSavedData* bcast = (BroadcastSavedData*)self->saved_data;

    if (x1->metadata && x1->metadata->requires_grad) {
        Tensor* grad_x1 = bcast ? reduce_grad_to_shape(grad_output, bcast->x1_ndim, bcast->x1_shape) : NULL;
        Tensor* g = bcast ? grad_x1 : grad_output;
        if (!g) goto skip_x1;
        if (!x1->metadata->grad) {
            if (bcast) {
                x1->metadata->grad = grad_x1;
                grad_x1 = NULL;
            } else {
                x1->metadata->grad = zeros_tensor(x1->dtype, x1->device_id, x1->ndim, x1->shape, NULL);
                if (x1->metadata->grad)
                    backwards_add_x1(grad_output->data, x1->metadata->grad->data,
                                    grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            rp_add(x1->metadata->grad->data, x1->metadata->grad->data, g->data,
                   x1->size, x1->dtype, x1->device_id);
            if (grad_x1) free_tensor(grad_x1);
        }
        skip_x1:;
    }

    if (x2->metadata && x2->metadata->requires_grad) {
        Tensor* grad_x2 = bcast ? reduce_grad_to_shape(grad_output, bcast->x2_ndim, bcast->x2_shape) : NULL;
        Tensor* g = bcast ? grad_x2 : grad_output;
        if (!g) goto skip_x2;
        if (!x2->metadata->grad) {
            if (bcast) {
                x2->metadata->grad = grad_x2;
                grad_x2 = NULL;
            } else {
                x2->metadata->grad = zeros_tensor(x2->dtype, x2->device_id, x2->ndim, x2->shape, NULL);
                if (x2->metadata->grad)
                    backwards_add_x2(grad_output->data, x2->metadata->grad->data,
                                    grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            rp_add(x2->metadata->grad->data, x2->metadata->grad->data, g->data,
                   x2->size, x2->dtype, x2->device_id);
            if (grad_x2) free_tensor(grad_x2);
        }
        skip_x2:;
    }
}

Tensor* op_add(Tensor* x1, Tensor* x2) {
    if (!x1 || !x2) return NULL;
    if (x1->dtype != x2->dtype || x1->device_id != x2->device_id) return NULL;

    int out_ndim;
    int* out_shape;
    if (broadcast_shapes(x1->ndim, x1->shape, x2->ndim, x2->shape, &out_ndim, &out_shape) != 0) return NULL;

    size_t out_size = 1;
    for (int i = 0; i < out_ndim; i++) out_size *= out_shape[i];

    Tensor* out = zeros_tensor(x1->dtype, x1->device_id, out_ndim, out_shape, NULL);
    if (!out) { free(out_shape); return NULL; }

    bool same_shape = (x1->ndim == x2->ndim && x1->size == x2->size);
    if (same_shape) {
        for (int i = 0; i < x1->ndim; i++) {
            if (x1->shape[i] != x2->shape[i]) { same_shape = false; break; }
        }
    }

    int result;
    if (same_shape) {
        result = rp_add(out->data, x1->data, x2->data, x1->size, x1->dtype, x1->device_id);
    } else {
        Tensor* e1 = rp_expand(x1, out_ndim, out_shape);
        Tensor* e2 = rp_expand(x2, out_ndim, out_shape);
        if (!e1 || !e2) { free_tensor(e1); free_tensor(e2); free(out_shape); free_tensor(out); return NULL; }
        result = rp_add_strided(out->data, e1->data, e2->data, out_ndim, out_shape,
                                e1->strides, e2->strides, out_size, x1->dtype, x1->device_id);
        free_tensor(e1);
        free_tensor(e2);
    }
    free(out_shape);

    if (result != 0) { free_tensor(out); return NULL; }

    bool requires_grad = (x1->metadata && x1->metadata->requires_grad) ||
                         (x2->metadata && x2->metadata->requires_grad);

    if (requires_grad) {
        if (!out->metadata) {
            out->metadata = (Meta*)calloc(1, sizeof(Meta));
            if (!out->metadata) { free_tensor(out); return NULL; }
        }
        out->metadata->requires_grad = true;
        out->metadata->is_leaf = false;

        GradFn* grad_fn = (GradFn*)calloc(1, sizeof(GradFn));
        if (!grad_fn) { free_tensor(out); return NULL; }

        grad_fn->backward = backward_add_fn;
        grad_fn->num_inputs = 2;
        grad_fn->inputs = (Tensor**)malloc(2 * sizeof(Tensor*));
        if (!grad_fn->inputs) { free(grad_fn); free_tensor(out); return NULL; }
        grad_fn->inputs[0] = x1;
        grad_fn->inputs[1] = x2;

        if (!same_shape) {
            BroadcastSavedData* saved = (BroadcastSavedData*)malloc(sizeof(BroadcastSavedData));
            if (!saved) { free(grad_fn->inputs); free(grad_fn); free_tensor(out); return NULL; }
            saved->x1_ndim = x1->ndim;
            saved->x1_shape = (int*)malloc(x1->ndim * sizeof(int));
            memcpy(saved->x1_shape, x1->shape, x1->ndim * sizeof(int));
            saved->x2_ndim = x2->ndim;
            saved->x2_shape = (int*)malloc(x2->ndim * sizeof(int));
            memcpy(saved->x2_shape, x2->shape, x2->ndim * sizeof(int));
            grad_fn->saved_data = saved;
        } else {
            grad_fn->saved_data = NULL;
        }

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
    BroadcastSavedData* bcast = (BroadcastSavedData*)self->saved_data;

    if (bcast) {
        Tensor* x2_exp = rp_expand(x2, grad_output->ndim, grad_output->shape);
        Tensor* x2_c = x2_exp ? rp_contiguous(x2_exp) : NULL;
        if (x2_exp) free_tensor(x2_exp);

        Tensor* x1_exp = rp_expand(x1, grad_output->ndim, grad_output->shape);
        Tensor* x1_c = x1_exp ? rp_contiguous(x1_exp) : NULL;
        if (x1_exp) free_tensor(x1_exp);

        if (x1->metadata && x1->metadata->requires_grad && x2_c) {
            Tensor* full_grad = zeros_tensor(grad_output->dtype, grad_output->device_id,
                                             grad_output->ndim, grad_output->shape, NULL);
            if (full_grad) {
                backwards_mul_x1(grad_output->data, x2_c->data, full_grad->data,
                                grad_output->size, grad_output->dtype, grad_output->device_id);
                Tensor* grad_x1 = reduce_grad_to_shape(full_grad, bcast->x1_ndim, bcast->x1_shape);
                free_tensor(full_grad);
                if (grad_x1) {
                    if (!x1->metadata->grad) {
                        x1->metadata->grad = grad_x1;
                    } else {
                        rp_add(x1->metadata->grad->data, x1->metadata->grad->data, grad_x1->data,
                               x1->size, x1->dtype, x1->device_id);
                        free_tensor(grad_x1);
                    }
                }
            }
        }

        if (x2->metadata && x2->metadata->requires_grad && x1_c) {
            Tensor* full_grad = zeros_tensor(grad_output->dtype, grad_output->device_id,
                                             grad_output->ndim, grad_output->shape, NULL);
            if (full_grad) {
                backwards_mul_x2(grad_output->data, x1_c->data, full_grad->data,
                                grad_output->size, grad_output->dtype, grad_output->device_id);
                Tensor* grad_x2 = reduce_grad_to_shape(full_grad, bcast->x2_ndim, bcast->x2_shape);
                free_tensor(full_grad);
                if (grad_x2) {
                    if (!x2->metadata->grad) {
                        x2->metadata->grad = grad_x2;
                    } else {
                        rp_add(x2->metadata->grad->data, x2->metadata->grad->data, grad_x2->data,
                               x2->size, x2->dtype, x2->device_id);
                        free_tensor(grad_x2);
                    }
                }
            }
        }

        if (x2_c) free_tensor(x2_c);
        if (x1_c) free_tensor(x1_c);
    } else {
        if (x1->metadata && x1->metadata->requires_grad) {
            if (!x1->metadata->grad) {
                x1->metadata->grad = zeros_tensor(x1->dtype, x1->device_id, x1->ndim, x1->shape, NULL);
                if (x1->metadata->grad)
                    backwards_mul_x1(grad_output->data, x2->data, x1->metadata->grad->data,
                                    grad_output->size, grad_output->dtype, grad_output->device_id);
            } else {
                Tensor* temp_grad = zeros_tensor(x1->dtype, x1->device_id, x1->ndim, x1->shape, NULL);
                if (temp_grad) {
                    backwards_mul_x1(grad_output->data, x2->data, temp_grad->data,
                                    grad_output->size, grad_output->dtype, grad_output->device_id);
                    rp_add(x1->metadata->grad->data, x1->metadata->grad->data, temp_grad->data,
                           x1->size, x1->dtype, x1->device_id);
                    free_tensor(temp_grad);
                }
            }
        }

        if (x2->metadata && x2->metadata->requires_grad) {
            if (!x2->metadata->grad) {
                x2->metadata->grad = zeros_tensor(x2->dtype, x2->device_id, x2->ndim, x2->shape, NULL);
                if (x2->metadata->grad)
                    backwards_mul_x2(grad_output->data, x1->data, x2->metadata->grad->data,
                                    grad_output->size, grad_output->dtype, grad_output->device_id);
            } else {
                Tensor* temp_grad = zeros_tensor(x2->dtype, x2->device_id, x2->ndim, x2->shape, NULL);
                if (temp_grad) {
                    backwards_mul_x2(grad_output->data, x1->data, temp_grad->data,
                                    grad_output->size, grad_output->dtype, grad_output->device_id);
                    rp_add(x2->metadata->grad->data, x2->metadata->grad->data, temp_grad->data,
                           x2->size, x2->dtype, x2->device_id);
                    free_tensor(temp_grad);
                }
            }
        }
    }
}

void backward_sub_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x1 = self->inputs[0];
    Tensor* x2 = self->inputs[1];
    BroadcastSavedData* bcast = (BroadcastSavedData*)self->saved_data;

    if (x1->metadata && x1->metadata->requires_grad) {
        Tensor* grad_x1 = bcast ? reduce_grad_to_shape(grad_output, bcast->x1_ndim, bcast->x1_shape) : NULL;
        Tensor* g = bcast ? grad_x1 : grad_output;
        if (!g) goto skip_sub_x1;
        if (!x1->metadata->grad) {
            if (bcast) {
                x1->metadata->grad = grad_x1;
                grad_x1 = NULL;
            } else {
                x1->metadata->grad = zeros_tensor(x1->dtype, x1->device_id, x1->ndim, x1->shape, NULL);
                if (x1->metadata->grad)
                    backwards_sub_x1(grad_output->data, x1->metadata->grad->data,
                                    grad_output->size, grad_output->dtype, grad_output->device_id);
            }
        } else {
            rp_add(x1->metadata->grad->data, x1->metadata->grad->data, g->data,
                   x1->size, x1->dtype, x1->device_id);
            if (grad_x1) free_tensor(grad_x1);
        }
        skip_sub_x1:;
    }

    if (x2->metadata && x2->metadata->requires_grad) {
        if (bcast) {
            Tensor* neg_grad = zeros_tensor(grad_output->dtype, grad_output->device_id,
                                            grad_output->ndim, grad_output->shape, NULL);
            if (!neg_grad) return;
            backwards_sub_x2(grad_output->data, neg_grad->data,
                            grad_output->size, grad_output->dtype, grad_output->device_id);
            Tensor* grad_x2 = reduce_grad_to_shape(neg_grad, bcast->x2_ndim, bcast->x2_shape);
            free_tensor(neg_grad);
            if (!grad_x2) return;
            if (!x2->metadata->grad) {
                x2->metadata->grad = grad_x2;
            } else {
                rp_add(x2->metadata->grad->data, x2->metadata->grad->data, grad_x2->data,
                       x2->size, x2->dtype, x2->device_id);
                free_tensor(grad_x2);
            }
        } else {
            if (!x2->metadata->grad) {
                x2->metadata->grad = zeros_tensor(x2->dtype, x2->device_id, x2->ndim, x2->shape, NULL);
                if (x2->metadata->grad)
                    backwards_sub_x2(grad_output->data, x2->metadata->grad->data,
                                    grad_output->size, grad_output->dtype, grad_output->device_id);
            } else {
                Tensor* temp_grad = zeros_tensor(x2->dtype, x2->device_id, x2->ndim, x2->shape, NULL);
                if (temp_grad) {
                    backwards_sub_x2(grad_output->data, temp_grad->data,
                                    grad_output->size, grad_output->dtype, grad_output->device_id);
                    rp_add(x2->metadata->grad->data, x2->metadata->grad->data, temp_grad->data,
                           x2->size, x2->dtype, x2->device_id);
                    free_tensor(temp_grad);
                }
            }
        }
    }
}

void backward_div_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x1 = self->inputs[0];
    Tensor* x2 = self->inputs[1];
    BroadcastSavedData* bcast = (BroadcastSavedData*)self->saved_data;

    if (bcast) {
        Tensor* x2_exp = rp_expand(x2, grad_output->ndim, grad_output->shape);
        Tensor* x2_c = x2_exp ? rp_contiguous(x2_exp) : NULL;
        if (x2_exp) free_tensor(x2_exp);

        Tensor* x1_exp = rp_expand(x1, grad_output->ndim, grad_output->shape);
        Tensor* x1_c = x1_exp ? rp_contiguous(x1_exp) : NULL;
        if (x1_exp) free_tensor(x1_exp);

        if (x1->metadata && x1->metadata->requires_grad && x2_c) {
            Tensor* full_grad = zeros_tensor(grad_output->dtype, grad_output->device_id,
                                             grad_output->ndim, grad_output->shape, NULL);
            if (full_grad) {
                backwards_div_x1(grad_output->data, x2_c->data, full_grad->data,
                                grad_output->size, grad_output->dtype, grad_output->device_id);
                Tensor* grad_x1 = reduce_grad_to_shape(full_grad, bcast->x1_ndim, bcast->x1_shape);
                free_tensor(full_grad);
                if (grad_x1) {
                    if (!x1->metadata->grad) {
                        x1->metadata->grad = grad_x1;
                    } else {
                        rp_add(x1->metadata->grad->data, x1->metadata->grad->data, grad_x1->data,
                               x1->size, x1->dtype, x1->device_id);
                        free_tensor(grad_x1);
                    }
                }
            }
        }

        if (x2->metadata && x2->metadata->requires_grad && x1_c && x2_c) {
            Tensor* full_grad = zeros_tensor(grad_output->dtype, grad_output->device_id,
                                             grad_output->ndim, grad_output->shape, NULL);
            if (full_grad) {
                backwards_div_x2(grad_output->data, x1_c->data, x2_c->data, full_grad->data,
                                grad_output->size, grad_output->dtype, grad_output->device_id);
                Tensor* grad_x2 = reduce_grad_to_shape(full_grad, bcast->x2_ndim, bcast->x2_shape);
                free_tensor(full_grad);
                if (grad_x2) {
                    if (!x2->metadata->grad) {
                        x2->metadata->grad = grad_x2;
                    } else {
                        rp_add(x2->metadata->grad->data, x2->metadata->grad->data, grad_x2->data,
                               x2->size, x2->dtype, x2->device_id);
                        free_tensor(grad_x2);
                    }
                }
            }
        }

        if (x2_c) free_tensor(x2_c);
        if (x1_c) free_tensor(x1_c);
    } else {
        if (x1->metadata && x1->metadata->requires_grad) {
            if (!x1->metadata->grad) {
                x1->metadata->grad = zeros_tensor(x1->dtype, x1->device_id, x1->ndim, x1->shape, NULL);
                if (x1->metadata->grad)
                    backwards_div_x1(grad_output->data, x2->data, x1->metadata->grad->data,
                                    grad_output->size, grad_output->dtype, grad_output->device_id);
            } else {
                Tensor* temp_grad = zeros_tensor(x1->dtype, x1->device_id, x1->ndim, x1->shape, NULL);
                if (temp_grad) {
                    backwards_div_x1(grad_output->data, x2->data, temp_grad->data,
                                    grad_output->size, grad_output->dtype, grad_output->device_id);
                    rp_add(x1->metadata->grad->data, x1->metadata->grad->data, temp_grad->data,
                           x1->size, x1->dtype, x1->device_id);
                    free_tensor(temp_grad);
                }
            }
        }

        if (x2->metadata && x2->metadata->requires_grad) {
            if (!x2->metadata->grad) {
                x2->metadata->grad = zeros_tensor(x2->dtype, x2->device_id, x2->ndim, x2->shape, NULL);
                if (x2->metadata->grad)
                    backwards_div_x2(grad_output->data, x1->data, x2->data, x2->metadata->grad->data,
                                    grad_output->size, grad_output->dtype, grad_output->device_id);
            } else {
                Tensor* temp_grad = zeros_tensor(x2->dtype, x2->device_id, x2->ndim, x2->shape, NULL);
                if (temp_grad) {
                    backwards_div_x2(grad_output->data, x1->data, x2->data, temp_grad->data,
                                    grad_output->size, grad_output->dtype, grad_output->device_id);
                    rp_add(x2->metadata->grad->data, x2->metadata->grad->data, temp_grad->data,
                           x2->size, x2->dtype, x2->device_id);
                    free_tensor(temp_grad);
                }
            }
        }
    }
}

Tensor* op_mul(Tensor* x1, Tensor* x2) {
    if (!x1 || !x2) return NULL;
    if (x1->dtype != x2->dtype || x1->device_id != x2->device_id) return NULL;

    int out_ndim;
    int* out_shape;
    if (broadcast_shapes(x1->ndim, x1->shape, x2->ndim, x2->shape, &out_ndim, &out_shape) != 0) return NULL;

    size_t out_size = 1;
    for (int i = 0; i < out_ndim; i++) out_size *= out_shape[i];

    Tensor* out = zeros_tensor(x1->dtype, x1->device_id, out_ndim, out_shape, NULL);
    if (!out) { free(out_shape); return NULL; }

    bool same_shape = (x1->ndim == x2->ndim && x1->size == x2->size);
    if (same_shape) {
        for (int i = 0; i < x1->ndim; i++) {
            if (x1->shape[i] != x2->shape[i]) { same_shape = false; break; }
        }
    }

    int result;
    if (same_shape) {
        result = rp_mul(out->data, x1->data, x2->data, x1->size, x1->dtype, x1->device_id);
    } else {
        Tensor* e1 = rp_expand(x1, out_ndim, out_shape);
        Tensor* e2 = rp_expand(x2, out_ndim, out_shape);
        if (!e1 || !e2) { free_tensor(e1); free_tensor(e2); free(out_shape); free_tensor(out); return NULL; }
        result = rp_mul_strided(out->data, e1->data, e2->data, out_ndim, out_shape,
                                e1->strides, e2->strides, out_size, x1->dtype, x1->device_id);
        free_tensor(e1);
        free_tensor(e2);
    }
    free(out_shape);

    if (result != 0) { free_tensor(out); return NULL; }

    bool requires_grad = (x1->metadata && x1->metadata->requires_grad) ||
                         (x2->metadata && x2->metadata->requires_grad);

    if (requires_grad) {
        if (!out->metadata) {
            out->metadata = (Meta*)calloc(1, sizeof(Meta));
            if (!out->metadata) { free_tensor(out); return NULL; }
        }
        out->metadata->requires_grad = true;
        out->metadata->is_leaf = false;

        GradFn* grad_fn = (GradFn*)calloc(1, sizeof(GradFn));
        if (!grad_fn) { free_tensor(out); return NULL; }

        grad_fn->backward = backward_mul_fn;
        grad_fn->num_inputs = 2;
        grad_fn->inputs = (Tensor**)malloc(2 * sizeof(Tensor*));
        if (!grad_fn->inputs) { free(grad_fn); free_tensor(out); return NULL; }
        grad_fn->inputs[0] = x1;
        grad_fn->inputs[1] = x2;

        if (!same_shape) {
            BroadcastSavedData* saved = (BroadcastSavedData*)malloc(sizeof(BroadcastSavedData));
            if (!saved) { free(grad_fn->inputs); free(grad_fn); free_tensor(out); return NULL; }
            saved->x1_ndim = x1->ndim;
            saved->x1_shape = (int*)malloc(x1->ndim * sizeof(int));
            memcpy(saved->x1_shape, x1->shape, x1->ndim * sizeof(int));
            saved->x2_ndim = x2->ndim;
            saved->x2_shape = (int*)malloc(x2->ndim * sizeof(int));
            memcpy(saved->x2_shape, x2->shape, x2->ndim * sizeof(int));
            grad_fn->saved_data = saved;
        } else {
            grad_fn->saved_data = NULL;
        }

        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

Tensor* op_sub(Tensor* x1, Tensor* x2) {
    if (!x1 || !x2) return NULL;
    if (x1->dtype != x2->dtype || x1->device_id != x2->device_id) return NULL;

    int out_ndim;
    int* out_shape;
    if (broadcast_shapes(x1->ndim, x1->shape, x2->ndim, x2->shape, &out_ndim, &out_shape) != 0) return NULL;

    size_t out_size = 1;
    for (int i = 0; i < out_ndim; i++) out_size *= out_shape[i];

    Tensor* out = zeros_tensor(x1->dtype, x1->device_id, out_ndim, out_shape, NULL);
    if (!out) { free(out_shape); return NULL; }

    bool same_shape = (x1->ndim == x2->ndim && x1->size == x2->size);
    if (same_shape) {
        for (int i = 0; i < x1->ndim; i++) {
            if (x1->shape[i] != x2->shape[i]) { same_shape = false; break; }
        }
    }

    int result;
    if (same_shape) {
        result = rp_sub(out->data, x1->data, x2->data, x1->size, x1->dtype, x1->device_id);
    } else {
        Tensor* e1 = rp_expand(x1, out_ndim, out_shape);
        Tensor* e2 = rp_expand(x2, out_ndim, out_shape);
        if (!e1 || !e2) { free_tensor(e1); free_tensor(e2); free(out_shape); free_tensor(out); return NULL; }
        result = rp_sub_strided(out->data, e1->data, e2->data, out_ndim, out_shape,
                                e1->strides, e2->strides, out_size, x1->dtype, x1->device_id);
        free_tensor(e1);
        free_tensor(e2);
    }
    free(out_shape);

    if (result != 0) { free_tensor(out); return NULL; }

    bool requires_grad = (x1->metadata && x1->metadata->requires_grad) ||
                         (x2->metadata && x2->metadata->requires_grad);

    if (requires_grad) {
        if (!out->metadata) {
            out->metadata = (Meta*)calloc(1, sizeof(Meta));
            if (!out->metadata) { free_tensor(out); return NULL; }
        }
        out->metadata->requires_grad = true;
        out->metadata->is_leaf = false;

        GradFn* grad_fn = (GradFn*)calloc(1, sizeof(GradFn));
        if (!grad_fn) { free_tensor(out); return NULL; }

        grad_fn->backward = backward_sub_fn;
        grad_fn->num_inputs = 2;
        grad_fn->inputs = (Tensor**)malloc(2 * sizeof(Tensor*));
        if (!grad_fn->inputs) { free(grad_fn); free_tensor(out); return NULL; }
        grad_fn->inputs[0] = x1;
        grad_fn->inputs[1] = x2;

        if (!same_shape) {
            BroadcastSavedData* saved = (BroadcastSavedData*)malloc(sizeof(BroadcastSavedData));
            if (!saved) { free(grad_fn->inputs); free(grad_fn); free_tensor(out); return NULL; }
            saved->x1_ndim = x1->ndim;
            saved->x1_shape = (int*)malloc(x1->ndim * sizeof(int));
            memcpy(saved->x1_shape, x1->shape, x1->ndim * sizeof(int));
            saved->x2_ndim = x2->ndim;
            saved->x2_shape = (int*)malloc(x2->ndim * sizeof(int));
            memcpy(saved->x2_shape, x2->shape, x2->ndim * sizeof(int));
            grad_fn->saved_data = saved;
        } else {
            grad_fn->saved_data = NULL;
        }

        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

Tensor* op_div(Tensor* x1, Tensor* x2) {
    if (!x1 || !x2) return NULL;
    if (x1->dtype != x2->dtype || x1->device_id != x2->device_id) return NULL;

    int out_ndim;
    int* out_shape;
    if (broadcast_shapes(x1->ndim, x1->shape, x2->ndim, x2->shape, &out_ndim, &out_shape) != 0) return NULL;

    size_t out_size = 1;
    for (int i = 0; i < out_ndim; i++) out_size *= out_shape[i];

    Tensor* out = zeros_tensor(x1->dtype, x1->device_id, out_ndim, out_shape, NULL);
    if (!out) { free(out_shape); return NULL; }

    bool same_shape = (x1->ndim == x2->ndim && x1->size == x2->size);
    if (same_shape) {
        for (int i = 0; i < x1->ndim; i++) {
            if (x1->shape[i] != x2->shape[i]) { same_shape = false; break; }
        }
    }

    int result;
    if (same_shape) {
        result = rp_divide(out->data, x1->data, x2->data, x1->size, x1->dtype, x1->device_id);
    } else {
        Tensor* e1 = rp_expand(x1, out_ndim, out_shape);
        Tensor* e2 = rp_expand(x2, out_ndim, out_shape);
        if (!e1 || !e2) { free_tensor(e1); free_tensor(e2); free(out_shape); free_tensor(out); return NULL; }
        result = rp_divide_strided(out->data, e1->data, e2->data, out_ndim, out_shape,
                                   e1->strides, e2->strides, out_size, x1->dtype, x1->device_id);
        free_tensor(e1);
        free_tensor(e2);
    }
    free(out_shape);

    if (result != 0) { free_tensor(out); return NULL; }

    bool requires_grad = (x1->metadata && x1->metadata->requires_grad) ||
                         (x2->metadata && x2->metadata->requires_grad);

    if (requires_grad) {
        if (!out->metadata) {
            out->metadata = (Meta*)calloc(1, sizeof(Meta));
            if (!out->metadata) { free_tensor(out); return NULL; }
        }
        out->metadata->requires_grad = true;
        out->metadata->is_leaf = false;

        GradFn* grad_fn = (GradFn*)calloc(1, sizeof(GradFn));
        if (!grad_fn) { free_tensor(out); return NULL; }

        grad_fn->backward = backward_div_fn;
        grad_fn->num_inputs = 2;
        grad_fn->inputs = (Tensor**)malloc(2 * sizeof(Tensor*));
        if (!grad_fn->inputs) { free(grad_fn); free_tensor(out); return NULL; }
        grad_fn->inputs[0] = x1;
        grad_fn->inputs[1] = x2;

        if (!same_shape) {
            BroadcastSavedData* saved = (BroadcastSavedData*)malloc(sizeof(BroadcastSavedData));
            if (!saved) { free(grad_fn->inputs); free(grad_fn); free_tensor(out); return NULL; }
            saved->x1_ndim = x1->ndim;
            saved->x1_shape = (int*)malloc(x1->ndim * sizeof(int));
            memcpy(saved->x1_shape, x1->shape, x1->ndim * sizeof(int));
            saved->x2_ndim = x2->ndim;
            saved->x2_shape = (int*)malloc(x2->ndim * sizeof(int));
            memcpy(saved->x2_shape, x2->shape, x2->ndim * sizeof(int));
            grad_fn->saved_data = saved;
        } else {
            grad_fn->saved_data = NULL;
        }

        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

void backward_power_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x1 = self->inputs[0];
    Tensor* x2 = self->inputs[1];
    Tensor* output = self->inputs[2];
    BroadcastSavedData* bcast = (BroadcastSavedData*)self->saved_data;

    if (bcast) {
        Tensor* x1_exp = rp_expand(x1, grad_output->ndim, grad_output->shape);
        Tensor* x1_c = x1_exp ? rp_contiguous(x1_exp) : NULL;
        if (x1_exp) free_tensor(x1_exp);

        Tensor* x2_exp = rp_expand(x2, grad_output->ndim, grad_output->shape);
        Tensor* x2_c = x2_exp ? rp_contiguous(x2_exp) : NULL;
        if (x2_exp) free_tensor(x2_exp);

        if (x1->metadata && x1->metadata->requires_grad && x1_c && x2_c) {
            Tensor* full_grad = zeros_tensor(grad_output->dtype, grad_output->device_id,
                                             grad_output->ndim, grad_output->shape, NULL);
            if (full_grad) {
                backwards_power_x1(grad_output->data, x1_c->data, x2_c->data, output->data, full_grad->data,
                                  grad_output->size, grad_output->dtype, grad_output->device_id);
                Tensor* grad_x1 = reduce_grad_to_shape(full_grad, bcast->x1_ndim, bcast->x1_shape);
                free_tensor(full_grad);
                if (grad_x1) {
                    if (!x1->metadata->grad) {
                        x1->metadata->grad = grad_x1;
                    } else {
                        rp_add(x1->metadata->grad->data, x1->metadata->grad->data, grad_x1->data,
                               x1->size, x1->dtype, x1->device_id);
                        free_tensor(grad_x1);
                    }
                }
            }
        }

        if (x2->metadata && x2->metadata->requires_grad && x1_c) {
            Tensor* full_grad = zeros_tensor(grad_output->dtype, grad_output->device_id,
                                             grad_output->ndim, grad_output->shape, NULL);
            if (full_grad) {
                backwards_power_x2(grad_output->data, x1_c->data, output->data, full_grad->data,
                                  grad_output->size, grad_output->dtype, grad_output->device_id);
                Tensor* grad_x2 = reduce_grad_to_shape(full_grad, bcast->x2_ndim, bcast->x2_shape);
                free_tensor(full_grad);
                if (grad_x2) {
                    if (!x2->metadata->grad) {
                        x2->metadata->grad = grad_x2;
                    } else {
                        rp_add(x2->metadata->grad->data, x2->metadata->grad->data, grad_x2->data,
                               x2->size, x2->dtype, x2->device_id);
                        free_tensor(grad_x2);
                    }
                }
            }
        }

        if (x1_c) free_tensor(x1_c);
        if (x2_c) free_tensor(x2_c);
    } else {
        if (x1->metadata && x1->metadata->requires_grad) {
            if (!x1->metadata->grad) {
                x1->metadata->grad = zeros_tensor(x1->dtype, x1->device_id, x1->ndim, x1->shape, NULL);
                if (x1->metadata->grad)
                    backwards_power_x1(grad_output->data, x1->data, x2->data, output->data, x1->metadata->grad->data,
                                      grad_output->size, grad_output->dtype, grad_output->device_id);
            } else {
                Tensor* temp_grad = zeros_tensor(x1->dtype, x1->device_id, x1->ndim, x1->shape, NULL);
                if (temp_grad) {
                    backwards_power_x1(grad_output->data, x1->data, x2->data, output->data, temp_grad->data,
                                      grad_output->size, grad_output->dtype, grad_output->device_id);
                    rp_add(x1->metadata->grad->data, x1->metadata->grad->data, temp_grad->data,
                           x1->size, x1->dtype, x1->device_id);
                    free_tensor(temp_grad);
                }
            }
        }

        if (x2->metadata && x2->metadata->requires_grad) {
            if (!x2->metadata->grad) {
                x2->metadata->grad = zeros_tensor(x2->dtype, x2->device_id, x2->ndim, x2->shape, NULL);
                if (x2->metadata->grad)
                    backwards_power_x2(grad_output->data, x1->data, output->data, x2->metadata->grad->data,
                                      grad_output->size, grad_output->dtype, grad_output->device_id);
            } else {
                Tensor* temp_grad = zeros_tensor(x2->dtype, x2->device_id, x2->ndim, x2->shape, NULL);
                if (temp_grad) {
                    backwards_power_x2(grad_output->data, x1->data, output->data, temp_grad->data,
                                      grad_output->size, grad_output->dtype, grad_output->device_id);
                    rp_add(x2->metadata->grad->data, x2->metadata->grad->data, temp_grad->data,
                           x2->size, x2->dtype, x2->device_id);
                    free_tensor(temp_grad);
                }
            }
        }
    }
}

void backward_logb_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x1 = self->inputs[0];
    Tensor* x2 = self->inputs[1];
    BroadcastSavedData* bcast = (BroadcastSavedData*)self->saved_data;

    if (bcast) {
        Tensor* x1_exp = rp_expand(x1, grad_output->ndim, grad_output->shape);
        Tensor* x1_c = x1_exp ? rp_contiguous(x1_exp) : NULL;
        if (x1_exp) free_tensor(x1_exp);

        Tensor* x2_exp = rp_expand(x2, grad_output->ndim, grad_output->shape);
        Tensor* x2_c = x2_exp ? rp_contiguous(x2_exp) : NULL;
        if (x2_exp) free_tensor(x2_exp);

        if (x1->metadata && x1->metadata->requires_grad && x1_c && x2_c) {
            Tensor* full_grad = zeros_tensor(grad_output->dtype, grad_output->device_id,
                                             grad_output->ndim, grad_output->shape, NULL);
            if (full_grad) {
                backwards_logb_x1(grad_output->data, x1_c->data, x2_c->data, full_grad->data,
                                 grad_output->size, grad_output->dtype, grad_output->device_id);
                Tensor* grad_x1 = reduce_grad_to_shape(full_grad, bcast->x1_ndim, bcast->x1_shape);
                free_tensor(full_grad);
                if (grad_x1) {
                    if (!x1->metadata->grad) {
                        x1->metadata->grad = grad_x1;
                    } else {
                        rp_add(x1->metadata->grad->data, x1->metadata->grad->data, grad_x1->data,
                               x1->size, x1->dtype, x1->device_id);
                        free_tensor(grad_x1);
                    }
                }
            }
        }

        if (x2->metadata && x2->metadata->requires_grad && x1_c && x2_c) {
            Tensor* full_grad = zeros_tensor(grad_output->dtype, grad_output->device_id,
                                             grad_output->ndim, grad_output->shape, NULL);
            if (full_grad) {
                backwards_logb_x2(grad_output->data, x1_c->data, x2_c->data, full_grad->data,
                                 grad_output->size, grad_output->dtype, grad_output->device_id);
                Tensor* grad_x2 = reduce_grad_to_shape(full_grad, bcast->x2_ndim, bcast->x2_shape);
                free_tensor(full_grad);
                if (grad_x2) {
                    if (!x2->metadata->grad) {
                        x2->metadata->grad = grad_x2;
                    } else {
                        rp_add(x2->metadata->grad->data, x2->metadata->grad->data, grad_x2->data,
                               x2->size, x2->dtype, x2->device_id);
                        free_tensor(grad_x2);
                    }
                }
            }
        }

        if (x1_c) free_tensor(x1_c);
        if (x2_c) free_tensor(x2_c);
    } else {
        if (x1->metadata && x1->metadata->requires_grad) {
            if (!x1->metadata->grad) {
                x1->metadata->grad = zeros_tensor(x1->dtype, x1->device_id, x1->ndim, x1->shape, NULL);
                if (x1->metadata->grad)
                    backwards_logb_x1(grad_output->data, x1->data, x2->data, x1->metadata->grad->data,
                                     grad_output->size, grad_output->dtype, grad_output->device_id);
            } else {
                Tensor* temp_grad = zeros_tensor(x1->dtype, x1->device_id, x1->ndim, x1->shape, NULL);
                if (temp_grad) {
                    backwards_logb_x1(grad_output->data, x1->data, x2->data, temp_grad->data,
                                     grad_output->size, grad_output->dtype, grad_output->device_id);
                    rp_add(x1->metadata->grad->data, x1->metadata->grad->data, temp_grad->data,
                           x1->size, x1->dtype, x1->device_id);
                    free_tensor(temp_grad);
                }
            }
        }

        if (x2->metadata && x2->metadata->requires_grad) {
            if (!x2->metadata->grad) {
                x2->metadata->grad = zeros_tensor(x2->dtype, x2->device_id, x2->ndim, x2->shape, NULL);
                if (x2->metadata->grad)
                    backwards_logb_x2(grad_output->data, x1->data, x2->data, x2->metadata->grad->data,
                                     grad_output->size, grad_output->dtype, grad_output->device_id);
            } else {
                Tensor* temp_grad = zeros_tensor(x2->dtype, x2->device_id, x2->ndim, x2->shape, NULL);
                if (temp_grad) {
                    backwards_logb_x2(grad_output->data, x1->data, x2->data, temp_grad->data,
                                     grad_output->size, grad_output->dtype, grad_output->device_id);
                    rp_add(x2->metadata->grad->data, x2->metadata->grad->data, temp_grad->data,
                           x2->size, x2->dtype, x2->device_id);
                    free_tensor(temp_grad);
                }
            }
        }
    }
}

Tensor* op_power(Tensor* x1, Tensor* x2) {
    if (!x1 || !x2) return NULL;
    if (x1->dtype != x2->dtype || x1->device_id != x2->device_id) return NULL;

    int out_ndim;
    int* out_shape;
    if (broadcast_shapes(x1->ndim, x1->shape, x2->ndim, x2->shape, &out_ndim, &out_shape) != 0) return NULL;

    size_t out_size = 1;
    for (int i = 0; i < out_ndim; i++) out_size *= out_shape[i];

    Tensor* out = zeros_tensor(x1->dtype, x1->device_id, out_ndim, out_shape, NULL);
    if (!out) { free(out_shape); return NULL; }

    bool same_shape = (x1->ndim == x2->ndim && x1->size == x2->size);
    if (same_shape) {
        for (int i = 0; i < x1->ndim; i++) {
            if (x1->shape[i] != x2->shape[i]) { same_shape = false; break; }
        }
    }

    int result;
    if (same_shape) {
        result = rp_power(out->data, x1->data, x2->data, x1->size, x1->dtype, x1->device_id);
    } else {
        Tensor* e1 = rp_expand(x1, out_ndim, out_shape);
        Tensor* e2 = rp_expand(x2, out_ndim, out_shape);
        if (!e1 || !e2) { free_tensor(e1); free_tensor(e2); free(out_shape); free_tensor(out); return NULL; }
        result = rp_power_strided(out->data, e1->data, e2->data, out_ndim, out_shape,
                                  e1->strides, e2->strides, out_size, x1->dtype, x1->device_id);
        free_tensor(e1);
        free_tensor(e2);
    }
    free(out_shape);

    if (result != 0) { free_tensor(out); return NULL; }

    bool requires_grad = (x1->metadata && x1->metadata->requires_grad) ||
                         (x2->metadata && x2->metadata->requires_grad);

    if (requires_grad) {
        if (!out->metadata) {
            out->metadata = (Meta*)calloc(1, sizeof(Meta));
            if (!out->metadata) { free_tensor(out); return NULL; }
        }
        out->metadata->requires_grad = true;
        out->metadata->is_leaf = false;

        GradFn* grad_fn = (GradFn*)calloc(1, sizeof(GradFn));
        if (!grad_fn) { free_tensor(out); return NULL; }

        grad_fn->backward = backward_power_fn;
        grad_fn->num_inputs = 3;
        grad_fn->inputs = (Tensor**)malloc(3 * sizeof(Tensor*));
        if (!grad_fn->inputs) { free(grad_fn); free_tensor(out); return NULL; }
        grad_fn->inputs[0] = x1;
        grad_fn->inputs[1] = x2;
        grad_fn->inputs[2] = out;

        if (!same_shape) {
            BroadcastSavedData* saved = (BroadcastSavedData*)malloc(sizeof(BroadcastSavedData));
            if (!saved) { free(grad_fn->inputs); free(grad_fn); free_tensor(out); return NULL; }
            saved->x1_ndim = x1->ndim;
            saved->x1_shape = (int*)malloc(x1->ndim * sizeof(int));
            memcpy(saved->x1_shape, x1->shape, x1->ndim * sizeof(int));
            saved->x2_ndim = x2->ndim;
            saved->x2_shape = (int*)malloc(x2->ndim * sizeof(int));
            memcpy(saved->x2_shape, x2->shape, x2->ndim * sizeof(int));
            grad_fn->saved_data = saved;
        } else {
            grad_fn->saved_data = NULL;
        }

        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

Tensor* op_logb(Tensor* x1, Tensor* x2) {
    if (!x1 || !x2) return NULL;
    if (x1->dtype != x2->dtype || x1->device_id != x2->device_id) return NULL;

    int out_ndim;
    int* out_shape;
    if (broadcast_shapes(x1->ndim, x1->shape, x2->ndim, x2->shape, &out_ndim, &out_shape) != 0) return NULL;

    size_t out_size = 1;
    for (int i = 0; i < out_ndim; i++) out_size *= out_shape[i];

    Tensor* out = zeros_tensor(x1->dtype, x1->device_id, out_ndim, out_shape, NULL);
    if (!out) { free(out_shape); return NULL; }

    bool same_shape = (x1->ndim == x2->ndim && x1->size == x2->size);
    if (same_shape) {
        for (int i = 0; i < x1->ndim; i++) {
            if (x1->shape[i] != x2->shape[i]) { same_shape = false; break; }
        }
    }

    int result;
    if (same_shape) {
        result = rp_logb(out->data, x1->data, x2->data, x1->size, x1->dtype, x1->device_id);
    } else {
        Tensor* e1 = rp_expand(x1, out_ndim, out_shape);
        Tensor* e2 = rp_expand(x2, out_ndim, out_shape);
        if (!e1 || !e2) { free_tensor(e1); free_tensor(e2); free(out_shape); free_tensor(out); return NULL; }
        result = rp_logb_strided(out->data, e1->data, e2->data, out_ndim, out_shape,
                                 e1->strides, e2->strides, out_size, x1->dtype, x1->device_id);
        free_tensor(e1);
        free_tensor(e2);
    }
    free(out_shape);

    if (result != 0) { free_tensor(out); return NULL; }

    bool requires_grad = (x1->metadata && x1->metadata->requires_grad) ||
                         (x2->metadata && x2->metadata->requires_grad);

    if (requires_grad) {
        if (!out->metadata) {
            out->metadata = (Meta*)calloc(1, sizeof(Meta));
            if (!out->metadata) { free_tensor(out); return NULL; }
        }
        out->metadata->requires_grad = true;
        out->metadata->is_leaf = false;

        GradFn* grad_fn = (GradFn*)calloc(1, sizeof(GradFn));
        if (!grad_fn) { free_tensor(out); return NULL; }

        grad_fn->backward = backward_logb_fn;
        grad_fn->num_inputs = 2;
        grad_fn->inputs = (Tensor**)malloc(2 * sizeof(Tensor*));
        if (!grad_fn->inputs) { free(grad_fn); free_tensor(out); return NULL; }
        grad_fn->inputs[0] = x1;
        grad_fn->inputs[1] = x2;

        if (!same_shape) {
            BroadcastSavedData* saved = (BroadcastSavedData*)malloc(sizeof(BroadcastSavedData));
            if (!saved) { free(grad_fn->inputs); free(grad_fn); free_tensor(out); return NULL; }
            saved->x1_ndim = x1->ndim;
            saved->x1_shape = (int*)malloc(x1->ndim * sizeof(int));
            memcpy(saved->x1_shape, x1->shape, x1->ndim * sizeof(int));
            saved->x2_ndim = x2->ndim;
            saved->x2_shape = (int*)malloc(x2->ndim * sizeof(int));
            memcpy(saved->x2_shape, x2->shape, x2->ndim * sizeof(int));
            grad_fn->saved_data = saved;
        } else {
            grad_fn->saved_data = NULL;
        }

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

int backwards_sum_dim(const void* grad_c, void* grad_x,
                      size_t outer_size, size_t dim_size, size_t inner_size,
                      DType dtype, int device_id) {
    if (!grad_c || !grad_x) return -1;

    if (device_id == -1) {
        return backwards_sum_dim_host(grad_c, grad_x, outer_size, dim_size, inner_size, dtype);
    } else {
        return backwards_sum_dim_device(grad_c, grad_x, outer_size, dim_size, inner_size, dtype);
    }
}

int backwards_mean_dim(const void* grad_c, void* grad_x,
                       size_t outer_size, size_t dim_size, size_t inner_size,
                       DType dtype, int device_id) {
    if (!grad_c || !grad_x) return -1;

    if (device_id == -1) {
        return backwards_mean_dim_host(grad_c, grad_x, outer_size, dim_size, inner_size, dtype);
    } else {
        return backwards_mean_dim_device(grad_c, grad_x, outer_size, dim_size, inner_size, dtype);
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
        void* alpha_ptr = (x->dtype == DTYPE_FLOAT64) ? (void*)&saved->alpha_f64 : (void*)&saved->alpha_f32;

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

typedef struct {
    size_t outer_size;
    size_t dim_size;
    size_t inner_size;
    int input_ndim;
    int* input_shape;
} SumDimSavedData;

void backward_sum_dim_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];
    SumDimSavedData* saved = (SumDimSavedData*)self->saved_data;
    if (!saved) return;

    if (x->metadata && x->metadata->requires_grad) {
        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (x->metadata->grad) {
                backwards_sum_dim(grad_output->data, x->metadata->grad->data,
                                  saved->outer_size, saved->dim_size, saved->inner_size,
                                  x->dtype, x->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (temp_grad) {
                backwards_sum_dim(grad_output->data, temp_grad->data,
                                  saved->outer_size, saved->dim_size, saved->inner_size,
                                  x->dtype, x->device_id);
                rp_add(x->metadata->grad->data, x->metadata->grad->data, temp_grad->data,
                       x->size, x->dtype, x->device_id);
                free_tensor(temp_grad);
            }
        }
    }
}

void backward_mean_dim_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];
    SumDimSavedData* saved = (SumDimSavedData*)self->saved_data;
    if (!saved) return;

    if (x->metadata && x->metadata->requires_grad) {
        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (x->metadata->grad) {
                backwards_mean_dim(grad_output->data, x->metadata->grad->data,
                                   saved->outer_size, saved->dim_size, saved->inner_size,
                                   x->dtype, x->device_id);
            }
        } else {
            Tensor* temp_grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (temp_grad) {
                backwards_mean_dim(grad_output->data, temp_grad->data,
                                   saved->outer_size, saved->dim_size, saved->inner_size,
                                   x->dtype, x->device_id);
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
    if (x->dtype == DTYPE_FLOAT64) {
        alpha_ptr = &alpha_f64;
    } else {
        alpha_ptr = &alpha_f32;
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

Tensor* op_sum_dim(Tensor* x, int dim, bool keepdim) {
    if (!x) return NULL;

    if (dim < 0) dim += x->ndim;
    if (dim < 0 || dim >= x->ndim) return NULL;

    Tensor* input = x;
    bool made_contiguous = false;
    if (!rp_is_contiguous(x)) {
        input = rp_contiguous(x);
        if (!input) return NULL;
        made_contiguous = true;
    }

    size_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= input->shape[i];
    size_t dim_size = input->shape[dim];
    size_t inner_size = 1;
    for (int i = dim + 1; i < input->ndim; i++) inner_size *= input->shape[i];

    int out_ndim;
    int* out_shape;
    if (keepdim) {
        out_ndim = input->ndim;
        out_shape = (int*)malloc(out_ndim * sizeof(int));
        if (!out_shape) { if (made_contiguous) free_tensor(input); return NULL; }
        memcpy(out_shape, input->shape, out_ndim * sizeof(int));
        out_shape[dim] = 1;
    } else {
        out_ndim = input->ndim - 1;
        if (out_ndim == 0) out_ndim = 1;
        out_shape = (int*)malloc(out_ndim * sizeof(int));
        if (!out_shape) { if (made_contiguous) free_tensor(input); return NULL; }
        if (input->ndim == 1) {
            out_shape[0] = 1;
        } else {
            int j = 0;
            for (int i = 0; i < input->ndim; i++) {
                if (i != dim) out_shape[j++] = input->shape[i];
            }
        }
    }

    Tensor* out = zeros_tensor(input->dtype, input->device_id, out_ndim, out_shape, NULL);
    free(out_shape);
    if (!out) { if (made_contiguous) free_tensor(input); return NULL; }

    int result = rp_sum_dim(out->data, input->data, outer_size, dim_size, inner_size,
                            input->dtype, input->device_id);
    if (result != 0) {
        free_tensor(out);
        if (made_contiguous) free_tensor(input);
        return NULL;
    }

    bool requires_grad = (x->metadata && x->metadata->requires_grad);

    if (requires_grad) {
        if (!out->metadata) {
            out->metadata = (Meta*)calloc(1, sizeof(Meta));
            if (!out->metadata) {
                free_tensor(out);
                if (made_contiguous) free_tensor(input);
                return NULL;
            }
        }
        out->metadata->requires_grad = true;
        out->metadata->is_leaf = false;

        GradFn* grad_fn = (GradFn*)calloc(1, sizeof(GradFn));
        if (!grad_fn) {
            free_tensor(out);
            if (made_contiguous) free_tensor(input);
            return NULL;
        }

        grad_fn->backward = backward_sum_dim_fn;
        grad_fn->num_inputs = 1;
        grad_fn->inputs = (Tensor**)malloc(1 * sizeof(Tensor*));
        if (!grad_fn->inputs) {
            free(grad_fn);
            free_tensor(out);
            if (made_contiguous) free_tensor(input);
            return NULL;
        }
        grad_fn->inputs[0] = x;

        SumDimSavedData* saved = (SumDimSavedData*)malloc(sizeof(SumDimSavedData));
        if (!saved) {
            free(grad_fn->inputs);
            free(grad_fn);
            free_tensor(out);
            if (made_contiguous) free_tensor(input);
            return NULL;
        }
        saved->outer_size = outer_size;
        saved->dim_size = dim_size;
        saved->inner_size = inner_size;
        saved->input_ndim = x->ndim;
        saved->input_shape = (int*)malloc(x->ndim * sizeof(int));
        if (!saved->input_shape) {
            free(saved);
            free(grad_fn->inputs);
            free(grad_fn);
            free_tensor(out);
            if (made_contiguous) free_tensor(input);
            return NULL;
        }
        memcpy(saved->input_shape, x->shape, x->ndim * sizeof(int));
        grad_fn->saved_data = saved;

        out->metadata->grad_fn = grad_fn;
    }

    if (made_contiguous) free_tensor(input);
    return out;
}

Tensor* op_mean_dim(Tensor* x, int dim, bool keepdim) {
    if (!x) return NULL;

    if (dim < 0) dim += x->ndim;
    if (dim < 0 || dim >= x->ndim) return NULL;

    Tensor* input = x;
    bool made_contiguous = false;
    if (!rp_is_contiguous(x)) {
        input = rp_contiguous(x);
        if (!input) return NULL;
        made_contiguous = true;
    }

    size_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= input->shape[i];
    size_t dim_size = input->shape[dim];
    size_t inner_size = 1;
    for (int i = dim + 1; i < input->ndim; i++) inner_size *= input->shape[i];

    int out_ndim;
    int* out_shape;
    if (keepdim) {
        out_ndim = input->ndim;
        out_shape = (int*)malloc(out_ndim * sizeof(int));
        if (!out_shape) { if (made_contiguous) free_tensor(input); return NULL; }
        memcpy(out_shape, input->shape, out_ndim * sizeof(int));
        out_shape[dim] = 1;
    } else {
        out_ndim = input->ndim - 1;
        if (out_ndim == 0) out_ndim = 1;
        out_shape = (int*)malloc(out_ndim * sizeof(int));
        if (!out_shape) { if (made_contiguous) free_tensor(input); return NULL; }
        if (input->ndim == 1) {
            out_shape[0] = 1;
        } else {
            int j = 0;
            for (int i = 0; i < input->ndim; i++) {
                if (i != dim) out_shape[j++] = input->shape[i];
            }
        }
    }

    Tensor* out = zeros_tensor(input->dtype, input->device_id, out_ndim, out_shape, NULL);
    free(out_shape);
    if (!out) { if (made_contiguous) free_tensor(input); return NULL; }

    int result = rp_mean_dim(out->data, input->data, outer_size, dim_size, inner_size,
                             input->dtype, input->device_id);
    if (result != 0) {
        free_tensor(out);
        if (made_contiguous) free_tensor(input);
        return NULL;
    }

    bool requires_grad = (x->metadata && x->metadata->requires_grad);

    if (requires_grad) {
        if (!out->metadata) {
            out->metadata = (Meta*)calloc(1, sizeof(Meta));
            if (!out->metadata) {
                free_tensor(out);
                if (made_contiguous) free_tensor(input);
                return NULL;
            }
        }
        out->metadata->requires_grad = true;
        out->metadata->is_leaf = false;

        GradFn* grad_fn = (GradFn*)calloc(1, sizeof(GradFn));
        if (!grad_fn) {
            free_tensor(out);
            if (made_contiguous) free_tensor(input);
            return NULL;
        }

        grad_fn->backward = backward_mean_dim_fn;
        grad_fn->num_inputs = 1;
        grad_fn->inputs = (Tensor**)malloc(1 * sizeof(Tensor*));
        if (!grad_fn->inputs) {
            free(grad_fn);
            free_tensor(out);
            if (made_contiguous) free_tensor(input);
            return NULL;
        }
        grad_fn->inputs[0] = x;

        SumDimSavedData* saved = (SumDimSavedData*)malloc(sizeof(SumDimSavedData));
        if (!saved) {
            free(grad_fn->inputs);
            free(grad_fn);
            free_tensor(out);
            if (made_contiguous) free_tensor(input);
            return NULL;
        }
        saved->outer_size = outer_size;
        saved->dim_size = dim_size;
        saved->inner_size = inner_size;
        saved->input_ndim = x->ndim;
        saved->input_shape = (int*)malloc(x->ndim * sizeof(int));
        if (!saved->input_shape) {
            free(saved);
            free(grad_fn->inputs);
            free(grad_fn);
            free_tensor(out);
            if (made_contiguous) free_tensor(input);
            return NULL;
        }
        memcpy(saved->input_shape, x->shape, x->ndim * sizeof(int));
        grad_fn->saved_data = saved;

        out->metadata->grad_fn = grad_fn;
    }

    if (made_contiguous) free_tensor(input);
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

typedef struct {
    int dim;
    int* indices;
    int* d_indices;
    int index_ndim;
    int* index_shape;
    int input_ndim;
    int* input_shape;
    size_t index_size;
} GatherSavedData;

void backward_gather_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* input = self->inputs[0];
    GatherSavedData* saved = (GatherSavedData*)self->saved_data;
    if (!saved) return;

    if (input->metadata && input->metadata->requires_grad) {
        if (!input->metadata->grad) {
            input->metadata->grad = zeros_tensor(input->dtype, input->device_id,
                                                  saved->input_ndim, saved->input_shape, NULL);
            if (!input->metadata->grad) return;
        }

        const int* idx_ptr = saved->indices;
        if (input->device_id >= 0 && saved->d_indices) {
            idx_ptr = saved->d_indices;
        }

        Tensor* temp = zeros_tensor(input->dtype, input->device_id,
                                     saved->input_ndim, saved->input_shape, NULL);
        if (!temp) return;

        rp_scatter_add(temp->data, grad_output->data, idx_ptr,
                       saved->index_ndim, saved->input_shape, saved->index_shape,
                       saved->dim, saved->index_size, input->dtype, input->device_id);

        rp_add(input->metadata->grad->data, input->metadata->grad->data, temp->data,
               input->size, input->dtype, input->device_id);
        free_tensor(temp);
    }
}

Tensor* op_gather(Tensor* input, int dim, const int* indices, int index_ndim, const int* index_shape, size_t index_size) {
    if (!input || !indices || !index_shape) return NULL;

    if (dim < 0) dim += input->ndim;
    if (dim < 0 || dim >= input->ndim) return NULL;
    if (index_ndim != input->ndim) return NULL;

    Tensor* out = zeros_tensor(input->dtype, input->device_id, index_ndim, (int*)index_shape, NULL);
    if (!out) return NULL;

    const int* idx_ptr = indices;
    int* d_indices = NULL;

    if (input->device_id >= 0) {
        cudaMalloc((void**)&d_indices, index_size * sizeof(int));
        if (!d_indices) { free_tensor(out); return NULL; }
        cudaMemcpy(d_indices, indices, index_size * sizeof(int), cudaMemcpyHostToDevice);
        idx_ptr = d_indices;
    }

    int result = rp_gather(out->data, input->data, idx_ptr,
                           input->ndim, input->shape, index_shape,
                           dim, index_size, input->dtype, input->device_id);
    if (result != 0) {
        if (d_indices) cudaFree(d_indices);
        free_tensor(out);
        return NULL;
    }

    bool requires_grad = (input->metadata && input->metadata->requires_grad);
    if (requires_grad) {
        if (!out->metadata) {
            out->metadata = (Meta*)calloc(1, sizeof(Meta));
            if (!out->metadata) { if (d_indices) cudaFree(d_indices); free_tensor(out); return NULL; }
        }
        out->metadata->requires_grad = true;
        out->metadata->is_leaf = false;

        GradFn* grad_fn = (GradFn*)calloc(1, sizeof(GradFn));
        if (!grad_fn) { if (d_indices) cudaFree(d_indices); free_tensor(out); return NULL; }

        grad_fn->backward = backward_gather_fn;
        grad_fn->num_inputs = 1;
        grad_fn->inputs = (Tensor**)malloc(1 * sizeof(Tensor*));
        if (!grad_fn->inputs) { free(grad_fn); if (d_indices) cudaFree(d_indices); free_tensor(out); return NULL; }
        grad_fn->inputs[0] = input;

        GatherSavedData* saved = (GatherSavedData*)malloc(sizeof(GatherSavedData));
        if (!saved) { free(grad_fn->inputs); free(grad_fn); if (d_indices) cudaFree(d_indices); free_tensor(out); return NULL; }
        saved->dim = dim;
        saved->index_ndim = index_ndim;
        saved->index_size = index_size;
        saved->input_ndim = input->ndim;

        saved->indices = (int*)malloc(index_size * sizeof(int));
        memcpy(saved->indices, indices, index_size * sizeof(int));
        saved->d_indices = d_indices;

        saved->index_shape = (int*)malloc(index_ndim * sizeof(int));
        memcpy(saved->index_shape, index_shape, index_ndim * sizeof(int));

        saved->input_shape = (int*)malloc(input->ndim * sizeof(int));
        memcpy(saved->input_shape, input->shape, input->ndim * sizeof(int));

        grad_fn->saved_data = saved;
        out->metadata->grad_fn = grad_fn;
    } else {
        if (d_indices) cudaFree(d_indices);
    }

    return out;
}

void backward_softmax_fn(GradFn* self, Tensor* grad_output);
void backward_log_softmax_fn(GradFn* self, Tensor* grad_output);
void backward_sigmoid_fn(GradFn* self, Tensor* grad_output);

typedef struct {
    int stride_h, stride_w;
    int pad_h, pad_w;
    int dilation_h, dilation_w;
    int N, C_in, H, W;
    int C_out, kH, kW;
    int out_H, out_W;
    int has_bias;
} Conv2dSavedData;

void backward_conv2d_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* input = self->inputs[0];
    Tensor* weight = self->inputs[1];
    Conv2dSavedData* s = (Conv2dSavedData*)self->saved_data;
    if (!s) return;

    DType dtype = input->dtype;
    int dev = input->device_id;
    size_t esz = dtype_size(dtype);
    int col_rows = s->C_in * s->kH * s->kW;
    int col_cols = s->out_H * s->out_W;

    if (input->metadata && input->metadata->requires_grad) {
        if (!input->metadata->grad) {
            input->metadata->grad = zeros_tensor(dtype, dev, input->ndim, input->shape, NULL);
            if (!input->metadata->grad) return;
        }

        void* col_buf = calloc(1, (size_t)col_rows * col_cols * esz);
        void* wt_buf = calloc(1, (size_t)s->C_out * col_rows * esz);
        void* grad_im = calloc(1, (size_t)s->C_in * s->H * s->W * esz);
        if (!col_buf || !wt_buf || !grad_im) { free(col_buf); free(wt_buf); free(grad_im); return; }
        if (dev >= 0) {
            void* d1; void* d2; void* d3;
            cudaMalloc(&d1, (size_t)col_rows * col_cols * esz);
            cudaMalloc(&d2, (size_t)s->C_out * col_rows * esz);
            cudaMalloc(&d3, (size_t)s->C_in * s->H * s->W * esz);
            free(col_buf); free(wt_buf); free(grad_im);
            col_buf = d1; wt_buf = d2; grad_im = d3;
        }

        if (dev == -1) {
            if (dtype == DTYPE_FLOAT32) {
                float* ws = (float*)weight->data;
                float* wd = (float*)wt_buf;
                for (int r = 0; r < s->C_out; r++)
                    for (int c = 0; c < col_rows; c++)
                        wd[c * s->C_out + r] = ws[r * col_rows + c];
            } else if (dtype == DTYPE_FLOAT64) {
                double* ws = (double*)weight->data;
                double* wd = (double*)wt_buf;
                for (int r = 0; r < s->C_out; r++)
                    for (int c = 0; c < col_rows; c++)
                        wd[c * s->C_out + r] = ws[r * col_rows + c];
            }
        } else {
            int wt_shape[] = {s->C_out, col_rows};
            int wt_strides[] = {col_rows, 1};
            Tensor tmp_w = {.dtype = dtype, .data = weight->data, .ndim = 2,
                            .size = (size_t)s->C_out * col_rows,
                            .shape = wt_shape, .strides = wt_strides, .device_id = dev,
                            .owns_data = false, .base_tensor = NULL, .data_offset = 0, .metadata = NULL};
            Tensor* wt_t = rp_transpose(&tmp_w, 0, 1);
            if (wt_t) {
                Tensor* wt_c = rp_contiguous(wt_t);
                if (wt_c) {
                    cudaMemcpy(wt_buf, wt_c->data, (size_t)s->C_out * col_rows * esz, cudaMemcpyDeviceToDevice);
                    free(wt_c->shape); free(wt_c->strides); free(wt_c);
                }
                free(wt_t->shape); free(wt_t->strides); free(wt_t);
            }
        }

        for (int n = 0; n < s->N; n++) {
            void* grad_out_n = (char*)grad_output->data + (size_t)n * s->C_out * col_cols * esz;

            rp_matmul(col_buf, wt_buf, grad_out_n, col_rows, s->C_out, col_cols, dtype, dev);

            if (dev >= 0) cudaMemset(grad_im, 0, (size_t)s->C_in * s->H * s->W * esz);
            else memset(grad_im, 0, (size_t)s->C_in * s->H * s->W * esz);

            rp_col2im_2d(grad_im, col_buf, s->C_in, s->H, s->W, s->kH, s->kW,
                         s->stride_h, s->stride_w, s->pad_h, s->pad_w,
                         s->dilation_h, s->dilation_w, s->out_H, s->out_W, dtype, dev);

            void* grad_ptr = (char*)input->metadata->grad->data + (size_t)n * s->C_in * s->H * s->W * esz;
            rp_add(grad_ptr, grad_ptr, grad_im, (size_t)s->C_in * s->H * s->W, dtype, dev);
        }

        if (dev >= 0) { cudaFree(col_buf); cudaFree(wt_buf); cudaFree(grad_im); }
        else { free(col_buf); free(wt_buf); free(grad_im); }
    }

    if (weight->metadata && weight->metadata->requires_grad) {
        if (!weight->metadata->grad) {
            weight->metadata->grad = zeros_tensor(dtype, dev, weight->ndim, weight->shape, NULL);
            if (!weight->metadata->grad) return;
        }

        void* col_buf = calloc(1, (size_t)col_rows * col_cols * esz);
        void* col_t_buf = calloc(1, (size_t)col_rows * col_cols * esz);
        void* grad_w_tmp = calloc(1, (size_t)s->C_out * col_rows * esz);
        if (!col_buf || !col_t_buf || !grad_w_tmp) { free(col_buf); free(col_t_buf); free(grad_w_tmp); return; }
        if (dev >= 0) {
            void* d1; void* d2; void* d3;
            cudaMalloc(&d1, (size_t)col_rows * col_cols * esz);
            cudaMalloc(&d2, (size_t)col_rows * col_cols * esz);
            cudaMalloc(&d3, (size_t)s->C_out * col_rows * esz);
            free(col_buf); free(col_t_buf); free(grad_w_tmp);
            col_buf = d1; col_t_buf = d2; grad_w_tmp = d3;
        }

        int col_t_shape[] = {col_rows, col_cols};
        int col_t_strides[] = {col_cols, 1};

        for (int n = 0; n < s->N; n++) {
            void* input_n = (char*)input->data + (size_t)n * s->C_in * s->H * s->W * esz;
            void* grad_out_n = (char*)grad_output->data + (size_t)n * s->C_out * col_cols * esz;

            rp_im2col_2d(col_buf, input_n, s->C_in, s->H, s->W, s->kH, s->kW,
                         s->stride_h, s->stride_w, s->pad_h, s->pad_w,
                         s->dilation_h, s->dilation_w, s->out_H, s->out_W, dtype, dev);

            if (dev == -1) {
                if (dtype == DTYPE_FLOAT32) {
                    float* src = (float*)col_buf;
                    float* dst = (float*)col_t_buf;
                    for (int r = 0; r < col_rows; r++)
                        for (int c = 0; c < col_cols; c++)
                            dst[c * col_rows + r] = src[r * col_cols + c];
                } else if (dtype == DTYPE_FLOAT64) {
                    double* src = (double*)col_buf;
                    double* dst = (double*)col_t_buf;
                    for (int r = 0; r < col_rows; r++)
                        for (int c = 0; c < col_cols; c++)
                            dst[c * col_rows + r] = src[r * col_cols + c];
                }
            } else {
                Tensor tmp_src = {.dtype = dtype, .data = col_buf, .ndim = 2, .size = (size_t)col_rows * col_cols,
                                  .shape = col_t_shape, .strides = col_t_strides, .device_id = dev,
                                  .owns_data = false, .base_tensor = NULL, .data_offset = 0, .metadata = NULL};
                Tensor* transposed = rp_transpose(&tmp_src, 0, 1);
                if (transposed) {
                    Tensor* contig = rp_contiguous(transposed);
                    if (contig) {
                        cudaMemcpy(col_t_buf, contig->data, (size_t)col_rows * col_cols * esz, cudaMemcpyDeviceToDevice);
                        free(contig->shape); free(contig->strides); free(contig);
                    }
                    free(transposed->shape); free(transposed->strides); free(transposed);
                }
            }

            rp_matmul(grad_w_tmp, grad_out_n, col_t_buf, s->C_out, col_cols, col_rows, dtype, dev);

            rp_add(weight->metadata->grad->data, weight->metadata->grad->data, grad_w_tmp,
                   (size_t)s->C_out * col_rows, dtype, dev);
        }

        if (dev >= 0) { cudaFree(col_buf); cudaFree(col_t_buf); cudaFree(grad_w_tmp); }
        else { free(col_buf); free(col_t_buf); free(grad_w_tmp); }
    }

    if (s->has_bias && self->num_inputs > 2) {
        Tensor* bias = self->inputs[2];
        if (bias && bias->metadata && bias->metadata->requires_grad) {
            if (!bias->metadata->grad) {
                bias->metadata->grad = zeros_tensor(dtype, dev, bias->ndim, bias->shape, NULL);
                if (!bias->metadata->grad) return;
            }

            Tensor* host_go = grad_output;
            bool free_host = false;
            if (dev >= 0) {
                host_go = tensor_to(grad_output, -1, dtype, false);
                if (!host_go) return;
                free_host = true;
            }

            float* bias_grad_f32 = (float*)calloc(s->C_out, sizeof(float));
            if (dtype == DTYPE_FLOAT32 || dtype == DTYPE_FLOAT16 || dtype == DTYPE_BFLOAT16) {
                const float* go_data;
                float* go_f32 = NULL;
                if (dtype != DTYPE_FLOAT32) {
                    go_f32 = (float*)malloc(grad_output->size * sizeof(float));
                    half_to_fp32_array(host_go->data, go_f32, grad_output->size, dtype);
                    go_data = go_f32;
                } else {
                    go_data = (const float*)host_go->data;
                }
                for (int n = 0; n < s->N; n++) {
                    for (int c = 0; c < s->C_out; c++) {
                        for (int hw = 0; hw < col_cols; hw++) {
                            bias_grad_f32[c] += go_data[(size_t)n * s->C_out * col_cols + c * col_cols + hw];
                        }
                    }
                }
                if (dev == -1) {
                    float* bg = (float*)bias->metadata->grad->data;
                    for (int c = 0; c < s->C_out; c++) bg[c] += bias_grad_f32[c];
                } else {
                    if (dtype == DTYPE_FLOAT32) {
                        float* bg_host = (float*)malloc(s->C_out * sizeof(float));
                        cudaMemcpy(bg_host, bias->metadata->grad->data, s->C_out * sizeof(float), cudaMemcpyDeviceToHost);
                        for (int c = 0; c < s->C_out; c++) bg_host[c] += bias_grad_f32[c];
                        cudaMemcpy(bias->metadata->grad->data, bg_host, s->C_out * sizeof(float), cudaMemcpyHostToDevice);
                        free(bg_host);
                    }
                }
                if (go_f32) free(go_f32);
            } else {
                const double* go_data = (const double*)host_go->data;
                double* bias_grad_f64 = (double*)calloc(s->C_out, sizeof(double));
                for (int n = 0; n < s->N; n++) {
                    for (int c = 0; c < s->C_out; c++) {
                        for (int hw = 0; hw < col_cols; hw++) {
                            bias_grad_f64[c] += go_data[(size_t)n * s->C_out * col_cols + c * col_cols + hw];
                        }
                    }
                }
                double* bg = (double*)bias->metadata->grad->data;
                for (int c = 0; c < s->C_out; c++) bg[c] += bias_grad_f64[c];
                free(bias_grad_f64);
            }
            free(bias_grad_f32);
            if (free_host) free_tensor(host_go);
        }
    }
}

Tensor* op_conv2d(Tensor* input, Tensor* weight, Tensor* bias,
                  int stride_h, int stride_w, int pad_h, int pad_w,
                  int dilation_h, int dilation_w) {
    if (!input || !weight) return NULL;
    if (input->ndim != 4 || weight->ndim != 4) return NULL;

    int N = input->shape[0];
    int C_in = input->shape[1];
    int H = input->shape[2];
    int W = input->shape[3];
    int C_out = weight->shape[0];
    int kH = weight->shape[2];
    int kW = weight->shape[3];

    if (weight->shape[1] != C_in) return NULL;

    int out_H = (H + 2 * pad_h - dilation_h * (kH - 1) - 1) / stride_h + 1;
    int out_W = (W + 2 * pad_w - dilation_w * (kW - 1) - 1) / stride_w + 1;
    if (out_H <= 0 || out_W <= 0) return NULL;

    DType dtype = input->dtype;
    int dev = input->device_id;
    size_t esz = dtype_size(dtype);
    int col_rows = C_in * kH * kW;
    int col_cols = out_H * out_W;

    int out_shape[] = {N, C_out, out_H, out_W};
    Tensor* out = zeros_tensor(dtype, dev, 4, out_shape, NULL);
    if (!out) return NULL;

    void* col_buf;
    if (dev >= 0) {
        if (cudaMalloc(&col_buf, (size_t)col_rows * col_cols * esz) != cudaSuccess) { free_tensor(out); return NULL; }
    } else {
        col_buf = malloc((size_t)col_rows * col_cols * esz);
        if (!col_buf) { free_tensor(out); return NULL; }
    }

    for (int n = 0; n < N; n++) {
        void* input_n = (char*)input->data + (size_t)n * C_in * H * W * esz;
        void* out_n = (char*)out->data + (size_t)n * C_out * col_cols * esz;

        rp_im2col_2d(col_buf, input_n, C_in, H, W, kH, kW,
                     stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
                     out_H, out_W, dtype, dev);

        rp_matmul(out_n, weight->data, col_buf, C_out, col_rows, col_cols, dtype, dev);
    }

    if (dev >= 0) cudaFree(col_buf);
    else free(col_buf);

    if (bias) {
        size_t spatial = (size_t)out_H * out_W;
        if (dev == -1) {
            if (dtype == DTYPE_FLOAT32) {
                float* od = (float*)out->data;
                const float* bd = (const float*)bias->data;
                for (int n = 0; n < N; n++)
                    for (int c = 0; c < C_out; c++)
                        for (size_t hw = 0; hw < spatial; hw++)
                            od[(size_t)n * C_out * spatial + c * spatial + hw] += bd[c];
            } else if (dtype == DTYPE_FLOAT64) {
                double* od = (double*)out->data;
                const double* bd = (const double*)bias->data;
                for (int n = 0; n < N; n++)
                    for (int c = 0; c < C_out; c++)
                        for (size_t hw = 0; hw < spatial; hw++)
                            od[(size_t)n * C_out * spatial + c * spatial + hw] += bd[c];
            }
        } else {
            Tensor* host_out = tensor_to(out, -1, dtype, false);
            Tensor* host_bias = tensor_to(bias, -1, dtype, false);
            if (host_out && host_bias) {
                if (dtype == DTYPE_FLOAT32) {
                    float* od = (float*)host_out->data;
                    const float* bd = (const float*)host_bias->data;
                    for (int n = 0; n < N; n++)
                        for (int c = 0; c < C_out; c++)
                            for (size_t hw = 0; hw < spatial; hw++)
                                od[(size_t)n * C_out * spatial + c * spatial + hw] += bd[c];
                }
                cudaMemcpy(out->data, host_out->data, out->size * esz, cudaMemcpyHostToDevice);
            }
            if (host_out) free_tensor(host_out);
            if (host_bias) free_tensor(host_bias);
        }
    }

    bool requires_grad = (input->metadata && input->metadata->requires_grad) ||
                          (weight->metadata && weight->metadata->requires_grad) ||
                          (bias && bias->metadata && bias->metadata->requires_grad);

    if (requires_grad) {
        if (!out->metadata) {
            out->metadata = (Meta*)calloc(1, sizeof(Meta));
            if (!out->metadata) { free_tensor(out); return NULL; }
        }
        out->metadata->requires_grad = true;
        out->metadata->is_leaf = false;

        int num_inputs = bias ? 3 : 2;
        GradFn* grad_fn = (GradFn*)calloc(1, sizeof(GradFn));
        if (!grad_fn) { free_tensor(out); return NULL; }
        grad_fn->backward = backward_conv2d_fn;
        grad_fn->num_inputs = num_inputs;
        grad_fn->inputs = (Tensor**)malloc(num_inputs * sizeof(Tensor*));
        if (!grad_fn->inputs) { free(grad_fn); free_tensor(out); return NULL; }
        grad_fn->inputs[0] = input;
        grad_fn->inputs[1] = weight;
        if (bias) grad_fn->inputs[2] = bias;

        Conv2dSavedData* saved = (Conv2dSavedData*)malloc(sizeof(Conv2dSavedData));
        if (!saved) { free(grad_fn->inputs); free(grad_fn); free_tensor(out); return NULL; }
        saved->stride_h = stride_h; saved->stride_w = stride_w;
        saved->pad_h = pad_h; saved->pad_w = pad_w;
        saved->dilation_h = dilation_h; saved->dilation_w = dilation_w;
        saved->N = N; saved->C_in = C_in; saved->H = H; saved->W = W;
        saved->C_out = C_out; saved->kH = kH; saved->kW = kW;
        saved->out_H = out_H; saved->out_W = out_W;
        saved->has_bias = bias ? 1 : 0;
        grad_fn->saved_data = saved;
        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

typedef struct {
    int reduction;
    size_t input_size;
} MSELossSavedData;

void backward_mse_loss_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* pred = self->inputs[0];
    Tensor* target = self->inputs[1];
    MSELossSavedData* saved = (MSELossSavedData*)self->saved_data;
    if (!saved || !pred->metadata || !pred->metadata->requires_grad) return;

    Tensor* diff = zeros_tensor(pred->dtype, pred->device_id, pred->ndim, pred->shape, NULL);
    if (!diff) return;
    rp_sub(diff->data, pred->data, target->data, pred->size, pred->dtype, pred->device_id);

    float scale_f32;
    double scale_f64;
    void* scale_ptr;

    if (saved->reduction == REDUCTION_MEAN) {
        scale_f32 = 2.0f / (float)saved->input_size;
        scale_f64 = 2.0 / (double)saved->input_size;
    } else {
        scale_f32 = 2.0f;
        scale_f64 = 2.0;
    }
    scale_ptr = (pred->dtype == DTYPE_FLOAT64) ? (void*)&scale_f64 : (void*)&scale_f32;

    Tensor* grad = zeros_tensor(pred->dtype, pred->device_id, pred->ndim, pred->shape, NULL);
    if (!grad) { free_tensor(diff); return; }
    rp_mul_scalar(grad->data, diff->data, scale_ptr, pred->size, pred->dtype, pred->device_id);
    free_tensor(diff);

    if (saved->reduction != REDUCTION_NONE) {
        float go_f32 = 0.0f; double go_f64 = 0.0;
        if (grad_output->device_id == -1) {
            if (grad_output->dtype == DTYPE_FLOAT64) go_f64 = *((double*)grad_output->data);
            else go_f32 = *((float*)grad_output->data);
        } else {
            if (grad_output->dtype == DTYPE_FLOAT64) cudaMemcpy(&go_f64, grad_output->data, sizeof(double), cudaMemcpyDeviceToHost);
            else cudaMemcpy(&go_f32, grad_output->data, sizeof(float), cudaMemcpyDeviceToHost);
        }
        void* go_ptr = (pred->dtype == DTYPE_FLOAT64) ? (void*)&go_f64 : (void*)&go_f32;
        Tensor* scaled = zeros_tensor(pred->dtype, pred->device_id, pred->ndim, pred->shape, NULL);
        if (scaled) {
            rp_mul_scalar(scaled->data, grad->data, go_ptr, pred->size, pred->dtype, pred->device_id);
            free_tensor(grad);
            grad = scaled;
        }
    }

    if (!pred->metadata->grad) {
        pred->metadata->grad = grad;
    } else {
        rp_add(pred->metadata->grad->data, pred->metadata->grad->data, grad->data,
               pred->size, pred->dtype, pred->device_id);
        free_tensor(grad);
    }
}

Tensor* op_mse_loss(Tensor* pred, Tensor* target, int reduction) {
    if (!pred || !target) return NULL;
    if (pred->size != target->size) return NULL;

    Tensor* diff = zeros_tensor(pred->dtype, pred->device_id, pred->ndim, pred->shape, NULL);
    if (!diff) return NULL;
    rp_sub(diff->data, pred->data, target->data, pred->size, pred->dtype, pred->device_id);

    Tensor* sq = zeros_tensor(pred->dtype, pred->device_id, pred->ndim, pred->shape, NULL);
    if (!sq) { free_tensor(diff); return NULL; }
    rp_mul(sq->data, diff->data, diff->data, pred->size, pred->dtype, pred->device_id);
    free_tensor(diff);

    Tensor* out;
    if (reduction == REDUCTION_MEAN) {
        int scalar_shape[] = {1};
        out = zeros_tensor(pred->dtype, pred->device_id, 1, scalar_shape, NULL);
        if (!out) { free_tensor(sq); return NULL; }
        rp_mean_all(out->data, sq->data, sq->size, pred->dtype, pred->device_id);
    } else if (reduction == REDUCTION_SUM) {
        int scalar_shape[] = {1};
        out = zeros_tensor(pred->dtype, pred->device_id, 1, scalar_shape, NULL);
        if (!out) { free_tensor(sq); return NULL; }
        rp_sum_all(out->data, sq->data, sq->size, pred->dtype, pred->device_id);
    } else {
        out = sq;
        sq = NULL;
    }
    if (sq) free_tensor(sq);

    bool requires_grad = (pred->metadata && pred->metadata->requires_grad);
    if (requires_grad) {
        if (!out->metadata) {
            out->metadata = (Meta*)calloc(1, sizeof(Meta));
            if (!out->metadata) { free_tensor(out); return NULL; }
        }
        out->metadata->requires_grad = true;
        out->metadata->is_leaf = false;

        GradFn* grad_fn = (GradFn*)calloc(1, sizeof(GradFn));
        if (!grad_fn) { free_tensor(out); return NULL; }
        grad_fn->backward = backward_mse_loss_fn;
        grad_fn->num_inputs = 2;
        grad_fn->inputs = (Tensor**)malloc(2 * sizeof(Tensor*));
        if (!grad_fn->inputs) { free(grad_fn); free_tensor(out); return NULL; }
        grad_fn->inputs[0] = pred;
        grad_fn->inputs[1] = target;

        MSELossSavedData* saved = (MSELossSavedData*)malloc(sizeof(MSELossSavedData));
        if (!saved) { free(grad_fn->inputs); free(grad_fn); free_tensor(out); return NULL; }
        saved->reduction = reduction;
        saved->input_size = pred->size;
        grad_fn->saved_data = saved;

        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

typedef struct {
    int reduction;
    size_t input_size;
    int from_logits;
} BCELossSavedData;

void backward_bce_loss_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* pred = self->inputs[0];
    Tensor* target = self->inputs[1];
    BCELossSavedData* saved = (BCELossSavedData*)self->saved_data;
    if (!saved || !pred->metadata || !pred->metadata->requires_grad) return;

    Tensor* probs = pred;
    bool free_probs = false;

    if (saved->from_logits) {
        probs = zeros_tensor(pred->dtype, pred->device_id, pred->ndim, pred->shape, NULL);
        if (!probs) return;
        rp_sigmoid(probs->data, pred->data, pred->size, pred->dtype, pred->device_id);
        free_probs = true;
    }

    Tensor* grad = zeros_tensor(pred->dtype, pred->device_id, pred->ndim, pred->shape, NULL);
    if (!grad) { if (free_probs) free_tensor(probs); return; }

    if (saved->from_logits) {
        Tensor* diff = zeros_tensor(pred->dtype, pred->device_id, pred->ndim, pred->shape, NULL);
        if (!diff) { free_tensor(grad); if (free_probs) free_tensor(probs); return; }
        rp_sub(diff->data, probs->data, target->data, pred->size, pred->dtype, pred->device_id);
        if (saved->reduction == REDUCTION_MEAN) {
            float s_f32 = 1.0f / (float)saved->input_size;
            double s_f64 = 1.0 / (double)saved->input_size;
            void* sp = (pred->dtype == DTYPE_FLOAT64) ? (void*)&s_f64 : (void*)&s_f32;
            rp_mul_scalar(grad->data, diff->data, sp, pred->size, pred->dtype, pred->device_id);
        } else {
            memcpy(grad->data, diff->data, pred->size * dtype_size(pred->dtype));
        }
        free_tensor(diff);
    } else {
        Tensor* one_minus_p = zeros_tensor(pred->dtype, pred->device_id, pred->ndim, pred->shape, NULL);
        Tensor* one_minus_t = zeros_tensor(pred->dtype, pred->device_id, pred->ndim, pred->shape, NULL);
        Tensor* term1 = zeros_tensor(pred->dtype, pred->device_id, pred->ndim, pred->shape, NULL);
        Tensor* term2 = zeros_tensor(pred->dtype, pred->device_id, pred->ndim, pred->shape, NULL);
        if (!one_minus_p || !one_minus_t || !term1 || !term2) {
            if (one_minus_p) free_tensor(one_minus_p);
            if (one_minus_t) free_tensor(one_minus_t);
            if (term1) free_tensor(term1);
            if (term2) free_tensor(term2);
            free_tensor(grad);
            return;
        }
        float one_f32 = 1.0f; double one_f64 = 1.0;
        void* one_ptr = (pred->dtype == DTYPE_FLOAT64) ? (void*)&one_f64 : (void*)&one_f32;
        rp_rsub_scalar(one_minus_p->data, one_ptr, probs->data, pred->size, pred->dtype, pred->device_id);
        rp_rsub_scalar(one_minus_t->data, one_ptr, target->data, pred->size, pred->dtype, pred->device_id);
        rp_divide(term1->data, target->data, probs->data, pred->size, pred->dtype, pred->device_id);
        rp_divide(term2->data, one_minus_t->data, one_minus_p->data, pred->size, pred->dtype, pred->device_id);
        rp_sub(grad->data, term2->data, term1->data, pred->size, pred->dtype, pred->device_id);

        if (saved->reduction == REDUCTION_MEAN) {
            float s_f32 = 1.0f / (float)saved->input_size;
            double s_f64 = 1.0 / (double)saved->input_size;
            void* sp = (pred->dtype == DTYPE_FLOAT64) ? (void*)&s_f64 : (void*)&s_f32;
            Tensor* temp = zeros_tensor(pred->dtype, pred->device_id, pred->ndim, pred->shape, NULL);
            if (temp) {
                rp_mul_scalar(temp->data, grad->data, sp, pred->size, pred->dtype, pred->device_id);
                free_tensor(grad);
                grad = temp;
            }
        }

        free_tensor(one_minus_p); free_tensor(one_minus_t);
        free_tensor(term1); free_tensor(term2);
    }

    if (free_probs) free_tensor(probs);

    if (saved->reduction != REDUCTION_NONE) {
        float go_f32 = 0.0f; double go_f64 = 0.0;
        if (grad_output->device_id == -1) {
            if (grad_output->dtype == DTYPE_FLOAT64) go_f64 = *((double*)grad_output->data);
            else go_f32 = *((float*)grad_output->data);
        } else {
            if (grad_output->dtype == DTYPE_FLOAT64) cudaMemcpy(&go_f64, grad_output->data, sizeof(double), cudaMemcpyDeviceToHost);
            else cudaMemcpy(&go_f32, grad_output->data, sizeof(float), cudaMemcpyDeviceToHost);
        }
        void* go_ptr = (pred->dtype == DTYPE_FLOAT64) ? (void*)&go_f64 : (void*)&go_f32;
        Tensor* scaled = zeros_tensor(pred->dtype, pred->device_id, pred->ndim, pred->shape, NULL);
        if (scaled) {
            rp_mul_scalar(scaled->data, grad->data, go_ptr, pred->size, pred->dtype, pred->device_id);
            free_tensor(grad);
            grad = scaled;
        }
    }

    if (!pred->metadata->grad) {
        pred->metadata->grad = grad;
    } else {
        rp_add(pred->metadata->grad->data, pred->metadata->grad->data, grad->data,
               pred->size, pred->dtype, pred->device_id);
        free_tensor(grad);
    }
}

Tensor* op_bce_loss(Tensor* pred, Tensor* target, int reduction, int from_logits) {
    if (!pred || !target) return NULL;
    if (pred->size != target->size) return NULL;

    Tensor* probs = pred;
    bool free_probs = false;

    int detected_logits = from_logits;
    if (from_logits == -1) {
        detected_logits = !(pred->metadata && pred->metadata->grad_fn &&
                           ((GradFn*)pred->metadata->grad_fn)->backward == backward_sigmoid_fn);
    }

    if (detected_logits) {
        probs = zeros_tensor(pred->dtype, pred->device_id, pred->ndim, pred->shape, NULL);
        if (!probs) return NULL;
        rp_sigmoid(probs->data, pred->data, pred->size, pred->dtype, pred->device_id);
        free_probs = true;
    }

    Tensor* log_p = zeros_tensor(pred->dtype, pred->device_id, pred->ndim, pred->shape, NULL);
    Tensor* log_1mp = zeros_tensor(pred->dtype, pred->device_id, pred->ndim, pred->shape, NULL);
    Tensor* one_minus_p = zeros_tensor(pred->dtype, pred->device_id, pred->ndim, pred->shape, NULL);
    Tensor* one_minus_t = zeros_tensor(pred->dtype, pred->device_id, pred->ndim, pred->shape, NULL);
    Tensor* term1 = zeros_tensor(pred->dtype, pred->device_id, pred->ndim, pred->shape, NULL);
    Tensor* term2 = zeros_tensor(pred->dtype, pred->device_id, pred->ndim, pred->shape, NULL);
    Tensor* bce = zeros_tensor(pred->dtype, pred->device_id, pred->ndim, pred->shape, NULL);

    if (!log_p || !log_1mp || !one_minus_p || !one_minus_t || !term1 || !term2 || !bce) {
        if (log_p) free_tensor(log_p); if (log_1mp) free_tensor(log_1mp);
        if (one_minus_p) free_tensor(one_minus_p); if (one_minus_t) free_tensor(one_minus_t);
        if (term1) free_tensor(term1); if (term2) free_tensor(term2);
        if (bce) free_tensor(bce);
        if (free_probs) free_tensor(probs);
        return NULL;
    }

    float one_f32 = 1.0f; double one_f64 = 1.0;
    void* one_ptr = (pred->dtype == DTYPE_FLOAT64) ? (void*)&one_f64 : (void*)&one_f32;
    float neg_f32 = -1.0f; double neg_f64 = -1.0;
    void* neg_ptr = (pred->dtype == DTYPE_FLOAT64) ? (void*)&neg_f64 : (void*)&neg_f32;

    rp_log(log_p->data, probs->data, pred->size, pred->dtype, pred->device_id);
    rp_rsub_scalar(one_minus_p->data, one_ptr, probs->data, pred->size, pred->dtype, pred->device_id);
    rp_log(log_1mp->data, one_minus_p->data, pred->size, pred->dtype, pred->device_id);
    rp_rsub_scalar(one_minus_t->data, one_ptr, target->data, pred->size, pred->dtype, pred->device_id);
    rp_mul(term1->data, target->data, log_p->data, pred->size, pred->dtype, pred->device_id);
    rp_mul(term2->data, one_minus_t->data, log_1mp->data, pred->size, pred->dtype, pred->device_id);
    rp_add(bce->data, term1->data, term2->data, pred->size, pred->dtype, pred->device_id);
    rp_mul_scalar(bce->data, bce->data, neg_ptr, pred->size, pred->dtype, pred->device_id);

    free_tensor(log_p); free_tensor(log_1mp); free_tensor(one_minus_p);
    free_tensor(one_minus_t); free_tensor(term1); free_tensor(term2);
    if (free_probs) free_tensor(probs);

    Tensor* out;
    if (reduction == REDUCTION_MEAN) {
        int s[] = {1};
        out = zeros_tensor(pred->dtype, pred->device_id, 1, s, NULL);
        if (!out) { free_tensor(bce); return NULL; }
        rp_mean_all(out->data, bce->data, bce->size, pred->dtype, pred->device_id);
        free_tensor(bce);
    } else if (reduction == REDUCTION_SUM) {
        int s[] = {1};
        out = zeros_tensor(pred->dtype, pred->device_id, 1, s, NULL);
        if (!out) { free_tensor(bce); return NULL; }
        rp_sum_all(out->data, bce->data, bce->size, pred->dtype, pred->device_id);
        free_tensor(bce);
    } else {
        out = bce;
    }

    bool requires_grad = (pred->metadata && pred->metadata->requires_grad);
    if (requires_grad) {
        if (!out->metadata) {
            out->metadata = (Meta*)calloc(1, sizeof(Meta));
            if (!out->metadata) { free_tensor(out); return NULL; }
        }
        out->metadata->requires_grad = true;
        out->metadata->is_leaf = false;

        GradFn* grad_fn = (GradFn*)calloc(1, sizeof(GradFn));
        if (!grad_fn) { free_tensor(out); return NULL; }
        grad_fn->backward = backward_bce_loss_fn;
        grad_fn->num_inputs = 2;
        grad_fn->inputs = (Tensor**)malloc(2 * sizeof(Tensor*));
        if (!grad_fn->inputs) { free(grad_fn); free_tensor(out); return NULL; }
        grad_fn->inputs[0] = pred;
        grad_fn->inputs[1] = target;

        BCELossSavedData* saved = (BCELossSavedData*)malloc(sizeof(BCELossSavedData));
        if (!saved) { free(grad_fn->inputs); free(grad_fn); free_tensor(out); return NULL; }
        saved->reduction = reduction;
        saved->input_size = pred->size;
        saved->from_logits = detected_logits;
        grad_fn->saved_data = saved;
        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

typedef struct {
    int reduction;
    int* targets;
    int batch_size;
    int num_classes;
    int from_logits;
} CELossSavedData;

void backward_nll_loss_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* input = self->inputs[0];
    CELossSavedData* saved = (CELossSavedData*)self->saved_data;
    if (!saved || !input->metadata || !input->metadata->requires_grad) return;

    if (!input->metadata->grad) {
        input->metadata->grad = zeros_tensor(input->dtype, input->device_id, input->ndim, input->shape, NULL);
        if (!input->metadata->grad) return;
    }

    Tensor* grad = zeros_tensor(input->dtype, input->device_id, input->ndim, input->shape, NULL);
    if (!grad) return;

    float neg_f32; double neg_f64;
    if (saved->reduction == REDUCTION_MEAN) {
        neg_f32 = -1.0f / (float)saved->batch_size;
        neg_f64 = -1.0 / (double)saved->batch_size;
    } else {
        neg_f32 = -1.0f;
        neg_f64 = -1.0;
    }

    size_t elem_sz = dtype_size(input->dtype);
    for (int b = 0; b < saved->batch_size; b++) {
        int cls = saved->targets[b];
        size_t offset = (size_t)b * saved->num_classes + cls;
        if (input->device_id == -1) {
            if (input->dtype == DTYPE_FLOAT32) ((float*)grad->data)[offset] = neg_f32;
            else if (input->dtype == DTYPE_FLOAT64) ((double*)grad->data)[offset] = neg_f64;
        } else {
            if (input->dtype == DTYPE_FLOAT32) cudaMemcpy((char*)grad->data + offset * elem_sz, &neg_f32, sizeof(float), cudaMemcpyHostToDevice);
            else if (input->dtype == DTYPE_FLOAT64) cudaMemcpy((char*)grad->data + offset * elem_sz, &neg_f64, sizeof(double), cudaMemcpyHostToDevice);
        }
    }

    if (saved->reduction != REDUCTION_NONE) {
        float go_f32 = 0.0f; double go_f64 = 0.0;
        if (grad_output->device_id == -1) {
            if (grad_output->dtype == DTYPE_FLOAT64) go_f64 = *((double*)grad_output->data);
            else go_f32 = *((float*)grad_output->data);
        } else {
            if (grad_output->dtype == DTYPE_FLOAT64) cudaMemcpy(&go_f64, grad_output->data, sizeof(double), cudaMemcpyDeviceToHost);
            else cudaMemcpy(&go_f32, grad_output->data, sizeof(float), cudaMemcpyDeviceToHost);
        }
        void* go_ptr = (input->dtype == DTYPE_FLOAT64) ? (void*)&go_f64 : (void*)&go_f32;
        Tensor* scaled = zeros_tensor(input->dtype, input->device_id, input->ndim, input->shape, NULL);
        if (scaled) {
            rp_mul_scalar(scaled->data, grad->data, go_ptr, input->size, input->dtype, input->device_id);
            free_tensor(grad);
            grad = scaled;
        }
    }

    rp_add(input->metadata->grad->data, input->metadata->grad->data, grad->data,
           input->size, input->dtype, input->device_id);
    free_tensor(grad);
}

Tensor* op_nll_loss(Tensor* input, const int* targets, int batch_size, int num_classes, int reduction) {
    if (!input || !targets) return NULL;
    if (input->ndim != 2 || input->shape[0] != batch_size || input->shape[1] != num_classes) return NULL;

    float sum_f32 = 0.0f;
    double sum_f64 = 0.0;

    Tensor* host_input = input;
    bool free_host = false;
    if (input->device_id >= 0) {
        host_input = tensor_to(input, -1, input->dtype, false);
        if (!host_input) return NULL;
        free_host = true;
    }

    for (int b = 0; b < batch_size; b++) {
        size_t offset = (size_t)b * num_classes + targets[b];
        if (input->dtype == DTYPE_FLOAT64) sum_f64 -= ((double*)host_input->data)[offset];
        else sum_f64 -= (double)((float*)host_input->data)[offset];
    }
    sum_f32 = (float)sum_f64;
    if (free_host) free_tensor(host_input);

    if (reduction == REDUCTION_MEAN) {
        sum_f32 /= (float)batch_size;
        sum_f64 /= (double)batch_size;
    }

    Tensor* out;
    if (reduction == REDUCTION_NONE) {
        int shape[] = {batch_size};
        out = zeros_tensor(input->dtype, input->device_id, 1, shape, NULL);
        if (!out) return NULL;
        if (input->device_id == -1) {
            for (int b = 0; b < batch_size; b++) {
                size_t offset = (size_t)b * num_classes + targets[b];
                if (input->dtype == DTYPE_FLOAT32) ((float*)out->data)[b] = -((float*)input->data)[offset];
                else ((double*)out->data)[b] = -((double*)input->data)[offset];
            }
        }
    } else {
        int shape[] = {1};
        out = zeros_tensor(input->dtype, input->device_id, 1, shape, NULL);
        if (!out) return NULL;
        if (input->device_id == -1) {
            if (input->dtype == DTYPE_FLOAT32) *((float*)out->data) = sum_f32;
            else *((double*)out->data) = sum_f64;
        } else {
            if (input->dtype == DTYPE_FLOAT32) cudaMemcpy(out->data, &sum_f32, sizeof(float), cudaMemcpyHostToDevice);
            else cudaMemcpy(out->data, &sum_f64, sizeof(double), cudaMemcpyHostToDevice);
        }
    }

    bool requires_grad = (input->metadata && input->metadata->requires_grad);
    if (requires_grad) {
        if (!out->metadata) {
            out->metadata = (Meta*)calloc(1, sizeof(Meta));
            if (!out->metadata) { free_tensor(out); return NULL; }
        }
        out->metadata->requires_grad = true;
        out->metadata->is_leaf = false;

        GradFn* grad_fn = (GradFn*)calloc(1, sizeof(GradFn));
        if (!grad_fn) { free_tensor(out); return NULL; }
        grad_fn->backward = backward_nll_loss_fn;
        grad_fn->num_inputs = 1;
        grad_fn->inputs = (Tensor**)malloc(1 * sizeof(Tensor*));
        if (!grad_fn->inputs) { free(grad_fn); free_tensor(out); return NULL; }
        grad_fn->inputs[0] = input;

        CELossSavedData* saved = (CELossSavedData*)malloc(sizeof(CELossSavedData));
        if (!saved) { free(grad_fn->inputs); free(grad_fn); free_tensor(out); return NULL; }
        saved->reduction = reduction;
        saved->batch_size = batch_size;
        saved->num_classes = num_classes;
        saved->from_logits = 0;
        saved->targets = (int*)malloc(batch_size * sizeof(int));
        memcpy(saved->targets, targets, batch_size * sizeof(int));
        grad_fn->saved_data = saved;
        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

void backward_cross_entropy_loss_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* input = self->inputs[0];
    Tensor* softmax_out = self->inputs[1];
    CELossSavedData* saved = (CELossSavedData*)self->saved_data;
    if (!saved || !input->metadata || !input->metadata->requires_grad) return;

    if (!input->metadata->grad) {
        input->metadata->grad = zeros_tensor(input->dtype, input->device_id, input->ndim, input->shape, NULL);
        if (!input->metadata->grad) return;
    }

    Tensor* grad = zeros_tensor(input->dtype, input->device_id, input->ndim, input->shape, NULL);
    if (!grad) return;

    size_t total = (size_t)saved->batch_size * saved->num_classes;

    Tensor* host_sm = softmax_out;
    bool free_sm = false;
    if (softmax_out->device_id >= 0) {
        host_sm = tensor_to(softmax_out, -1, softmax_out->dtype, false);
        if (!host_sm) { free_tensor(grad); return; }
        free_sm = true;
    }

    float* grad_f32 = NULL;
    if (input->dtype == DTYPE_FLOAT32 || input->dtype == DTYPE_FLOAT16 || input->dtype == DTYPE_BFLOAT16) {
        grad_f32 = (float*)malloc(total * sizeof(float));
        if (!grad_f32) { free_tensor(grad); if (free_sm) free_tensor(host_sm); return; }
        const float* sm_data;
        float* sm_f32_alloc = NULL;
        if (host_sm->dtype == DTYPE_FLOAT32) {
            sm_data = (const float*)host_sm->data;
        } else {
            sm_f32_alloc = (float*)malloc(total * sizeof(float));
            half_to_fp32_array(host_sm->data, sm_f32_alloc, total, host_sm->dtype);
            sm_data = sm_f32_alloc;
        }
        for (size_t i = 0; i < total; i++) grad_f32[i] = sm_data[i];
        for (int b = 0; b < saved->batch_size; b++) {
            grad_f32[(size_t)b * saved->num_classes + saved->targets[b]] -= 1.0f;
        }
        if (saved->reduction == REDUCTION_MEAN) {
            float scale = 1.0f / (float)saved->batch_size;
            for (size_t i = 0; i < total; i++) grad_f32[i] *= scale;
        }
        if (input->device_id == -1) {
            if (input->dtype == DTYPE_FLOAT32) memcpy(grad->data, grad_f32, total * sizeof(float));
            else fp32_to_half_array(grad_f32, grad->data, total, input->dtype);
        } else {
            if (input->dtype == DTYPE_FLOAT32) cudaMemcpy(grad->data, grad_f32, total * sizeof(float), cudaMemcpyHostToDevice);
            else {
                void* half_buf = malloc(total * dtype_size(input->dtype));
                fp32_to_half_array(grad_f32, half_buf, total, input->dtype);
                cudaMemcpy(grad->data, half_buf, total * dtype_size(input->dtype), cudaMemcpyHostToDevice);
                free(half_buf);
            }
        }
        free(grad_f32);
        if (sm_f32_alloc) free(sm_f32_alloc);
    } else {
        double* grad_f64 = (double*)malloc(total * sizeof(double));
        if (!grad_f64) { free_tensor(grad); if (free_sm) free_tensor(host_sm); return; }
        const double* sm_data = (const double*)host_sm->data;
        for (size_t i = 0; i < total; i++) grad_f64[i] = sm_data[i];
        for (int b = 0; b < saved->batch_size; b++) {
            grad_f64[(size_t)b * saved->num_classes + saved->targets[b]] -= 1.0;
        }
        if (saved->reduction == REDUCTION_MEAN) {
            double scale = 1.0 / (double)saved->batch_size;
            for (size_t i = 0; i < total; i++) grad_f64[i] *= scale;
        }
        if (input->device_id == -1) memcpy(grad->data, grad_f64, total * sizeof(double));
        else cudaMemcpy(grad->data, grad_f64, total * sizeof(double), cudaMemcpyHostToDevice);
        free(grad_f64);
    }
    if (free_sm) free_tensor(host_sm);

    if (saved->reduction != REDUCTION_NONE) {
        float go_f32 = 0.0f; double go_f64 = 0.0;
        if (grad_output->device_id == -1) {
            if (grad_output->dtype == DTYPE_FLOAT64) go_f64 = *((double*)grad_output->data);
            else go_f32 = *((float*)grad_output->data);
        } else {
            if (grad_output->dtype == DTYPE_FLOAT64) cudaMemcpy(&go_f64, grad_output->data, sizeof(double), cudaMemcpyDeviceToHost);
            else cudaMemcpy(&go_f32, grad_output->data, sizeof(float), cudaMemcpyDeviceToHost);
        }
        void* go_ptr = (input->dtype == DTYPE_FLOAT64) ? (void*)&go_f64 : (void*)&go_f32;
        Tensor* scaled = zeros_tensor(input->dtype, input->device_id, input->ndim, input->shape, NULL);
        if (scaled) {
            rp_mul_scalar(scaled->data, grad->data, go_ptr, input->size, input->dtype, input->device_id);
            free_tensor(grad);
            grad = scaled;
        }
    }

    rp_add(input->metadata->grad->data, input->metadata->grad->data, grad->data,
           input->size, input->dtype, input->device_id);
    free_tensor(grad);
}

Tensor* op_cross_entropy_loss(Tensor* input, const int* targets, int batch_size, int num_classes, int reduction, int from_logits) {
    if (!input || !targets) return NULL;
    if (input->ndim != 2 || input->shape[0] != batch_size || input->shape[1] != num_classes) return NULL;

    int detected_logits = from_logits;
    if (from_logits == -1) {
        detected_logits = !(input->metadata && input->metadata->grad_fn &&
                           ((GradFn*)input->metadata->grad_fn)->backward == backward_softmax_fn);
    }

    Tensor* log_probs;
    Tensor* softmax_result = NULL;
    bool free_log_probs = false;

    if (detected_logits) {
        int dim = input->ndim - 1;
        size_t outer = 1;
        for (int i = 0; i < dim; i++) outer *= input->shape[i];
        size_t dim_size = input->shape[dim];
        size_t inner = 1;

        log_probs = zeros_tensor(input->dtype, input->device_id, input->ndim, input->shape, NULL);
        if (!log_probs) return NULL;
        rp_log_softmax(log_probs->data, input->data, outer, dim_size, inner, input->dtype, input->device_id);
        free_log_probs = true;

        softmax_result = zeros_tensor(input->dtype, input->device_id, input->ndim, input->shape, NULL);
        if (!softmax_result) { free_tensor(log_probs); return NULL; }
        rp_softmax(softmax_result->data, input->data, outer, dim_size, inner, input->dtype, input->device_id);
    } else {
        log_probs = zeros_tensor(input->dtype, input->device_id, input->ndim, input->shape, NULL);
        if (!log_probs) return NULL;
        rp_log(log_probs->data, input->data, input->size, input->dtype, input->device_id);
        free_log_probs = true;

        softmax_result = input;
    }

    Tensor* nll_result = op_nll_loss(log_probs, targets, batch_size, num_classes, reduction);
    if (free_log_probs) free_tensor(log_probs);

    if (!nll_result) {
        if (softmax_result != input) free_tensor(softmax_result);
        return NULL;
    }

    Tensor* out = nll_result;

    bool requires_grad = (input->metadata && input->metadata->requires_grad);
    if (requires_grad) {
        if (out->metadata && out->metadata->grad_fn) {
            free_grad_fn(out->metadata->grad_fn);
        }
        if (!out->metadata) {
            out->metadata = (Meta*)calloc(1, sizeof(Meta));
            if (!out->metadata) {
                if (softmax_result != input) free_tensor(softmax_result);
                free_tensor(out);
                return NULL;
            }
        }
        out->metadata->requires_grad = true;
        out->metadata->is_leaf = false;

        GradFn* grad_fn = (GradFn*)calloc(1, sizeof(GradFn));
        if (!grad_fn) {
            if (softmax_result != input) free_tensor(softmax_result);
            free_tensor(out);
            return NULL;
        }
        grad_fn->backward = backward_cross_entropy_loss_fn;
        grad_fn->num_inputs = 2;
        grad_fn->inputs = (Tensor**)malloc(2 * sizeof(Tensor*));
        if (!grad_fn->inputs) { free(grad_fn); if (softmax_result != input) free_tensor(softmax_result); free_tensor(out); return NULL; }
        grad_fn->inputs[0] = input;
        grad_fn->inputs[1] = softmax_result;

        CELossSavedData* saved = (CELossSavedData*)malloc(sizeof(CELossSavedData));
        if (!saved) { free(grad_fn->inputs); free(grad_fn); if (softmax_result != input) free_tensor(softmax_result); free_tensor(out); return NULL; }
        saved->reduction = reduction;
        saved->batch_size = batch_size;
        saved->num_classes = num_classes;
        saved->from_logits = detected_logits;
        saved->targets = (int*)malloc(batch_size * sizeof(int));
        memcpy(saved->targets, targets, batch_size * sizeof(int));
        grad_fn->saved_data = saved;
        out->metadata->grad_fn = grad_fn;
    } else {
        if (softmax_result != input) free_tensor(softmax_result);
    }

    return out;
}

int backwards_softmax(const void* grad_c, const void* softmax_out, void* grad_x,
                      size_t outer_size, size_t dim_size, size_t inner_size,
                      DType dtype, int device_id) {
    if (device_id == -1) {
        return backwards_softmax_host(grad_c, softmax_out, grad_x, outer_size, dim_size, inner_size, dtype);
    } else {
        return backwards_softmax_device(grad_c, softmax_out, grad_x, outer_size, dim_size, inner_size, dtype);
    }
}

int backwards_log_softmax(const void* grad_c, const void* log_softmax_out, void* grad_x,
                           size_t outer_size, size_t dim_size, size_t inner_size,
                           DType dtype, int device_id) {
    if (device_id == -1) {
        return backwards_log_softmax_host(grad_c, log_softmax_out, grad_x, outer_size, dim_size, inner_size, dtype);
    } else {
        return backwards_log_softmax_device(grad_c, log_softmax_out, grad_x, outer_size, dim_size, inner_size, dtype);
    }
}

void backward_softmax_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];
    Tensor* out = self->inputs[1];
    SumDimSavedData* saved = (SumDimSavedData*)self->saved_data;
    if (!saved) return;

    if (x->metadata && x->metadata->requires_grad) {
        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (!x->metadata->grad) return;
            backwards_softmax(grad_output->data, out->data, x->metadata->grad->data,
                             saved->outer_size, saved->dim_size, saved->inner_size,
                             x->dtype, x->device_id);
        } else {
            Tensor* temp = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (!temp) return;
            backwards_softmax(grad_output->data, out->data, temp->data,
                             saved->outer_size, saved->dim_size, saved->inner_size,
                             x->dtype, x->device_id);
            rp_add(x->metadata->grad->data, x->metadata->grad->data, temp->data,
                   x->size, x->dtype, x->device_id);
            free_tensor(temp);
        }
    }
}

void backward_log_softmax_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];
    Tensor* out = self->inputs[1];
    SumDimSavedData* saved = (SumDimSavedData*)self->saved_data;
    if (!saved) return;

    if (x->metadata && x->metadata->requires_grad) {
        if (!x->metadata->grad) {
            x->metadata->grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (!x->metadata->grad) return;
            backwards_log_softmax(grad_output->data, out->data, x->metadata->grad->data,
                                  saved->outer_size, saved->dim_size, saved->inner_size,
                                  x->dtype, x->device_id);
        } else {
            Tensor* temp = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
            if (!temp) return;
            backwards_log_softmax(grad_output->data, out->data, temp->data,
                                  saved->outer_size, saved->dim_size, saved->inner_size,
                                  x->dtype, x->device_id);
            rp_add(x->metadata->grad->data, x->metadata->grad->data, temp->data,
                   x->size, x->dtype, x->device_id);
            free_tensor(temp);
        }
    }
}

Tensor* op_softmax(Tensor* x, int dim) {
    if (!x) return NULL;

    if (dim < 0) dim += x->ndim;
    if (dim < 0 || dim >= x->ndim) return NULL;

    Tensor* input = x;
    bool made_contiguous = false;
    if (!rp_is_contiguous(x)) {
        input = rp_contiguous(x);
        if (!input) return NULL;
        made_contiguous = true;
    }

    size_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= input->shape[i];
    size_t dim_size = input->shape[dim];
    size_t inner_size = 1;
    for (int i = dim + 1; i < input->ndim; i++) inner_size *= input->shape[i];

    Tensor* out = zeros_tensor(input->dtype, input->device_id, input->ndim, input->shape, NULL);
    if (!out) { if (made_contiguous) free_tensor(input); return NULL; }

    int result = rp_softmax(out->data, input->data, outer_size, dim_size, inner_size,
                            input->dtype, input->device_id);
    if (result != 0) {
        free_tensor(out);
        if (made_contiguous) free_tensor(input);
        return NULL;
    }

    if (made_contiguous) free_tensor(input);

    bool requires_grad = (x->metadata && x->metadata->requires_grad);
    if (requires_grad) {
        if (!out->metadata) {
            out->metadata = (Meta*)calloc(1, sizeof(Meta));
            if (!out->metadata) { free_tensor(out); return NULL; }
        }
        out->metadata->requires_grad = true;
        out->metadata->is_leaf = false;

        GradFn* grad_fn = (GradFn*)calloc(1, sizeof(GradFn));
        if (!grad_fn) { free_tensor(out); return NULL; }

        grad_fn->backward = backward_softmax_fn;
        grad_fn->num_inputs = 2;
        grad_fn->inputs = (Tensor**)malloc(2 * sizeof(Tensor*));
        if (!grad_fn->inputs) { free(grad_fn); free_tensor(out); return NULL; }
        grad_fn->inputs[0] = x;
        grad_fn->inputs[1] = out;

        SumDimSavedData* saved = (SumDimSavedData*)malloc(sizeof(SumDimSavedData));
        if (!saved) { free(grad_fn->inputs); free(grad_fn); free_tensor(out); return NULL; }
        saved->outer_size = outer_size;
        saved->dim_size = dim_size;
        saved->inner_size = inner_size;
        saved->input_ndim = x->ndim;
        saved->input_shape = (int*)malloc(x->ndim * sizeof(int));
        if (!saved->input_shape) { free(saved); free(grad_fn->inputs); free(grad_fn); free_tensor(out); return NULL; }
        memcpy(saved->input_shape, x->shape, x->ndim * sizeof(int));
        grad_fn->saved_data = saved;

        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

Tensor* op_log_softmax(Tensor* x, int dim) {
    if (!x) return NULL;

    if (dim < 0) dim += x->ndim;
    if (dim < 0 || dim >= x->ndim) return NULL;

    Tensor* input = x;
    bool made_contiguous = false;
    if (!rp_is_contiguous(x)) {
        input = rp_contiguous(x);
        if (!input) return NULL;
        made_contiguous = true;
    }

    size_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= input->shape[i];
    size_t dim_size = input->shape[dim];
    size_t inner_size = 1;
    for (int i = dim + 1; i < input->ndim; i++) inner_size *= input->shape[i];

    Tensor* out = zeros_tensor(input->dtype, input->device_id, input->ndim, input->shape, NULL);
    if (!out) { if (made_contiguous) free_tensor(input); return NULL; }

    int result = rp_log_softmax(out->data, input->data, outer_size, dim_size, inner_size,
                                input->dtype, input->device_id);
    if (result != 0) {
        free_tensor(out);
        if (made_contiguous) free_tensor(input);
        return NULL;
    }

    if (made_contiguous) free_tensor(input);

    bool requires_grad = (x->metadata && x->metadata->requires_grad);
    if (requires_grad) {
        if (!out->metadata) {
            out->metadata = (Meta*)calloc(1, sizeof(Meta));
            if (!out->metadata) { free_tensor(out); return NULL; }
        }
        out->metadata->requires_grad = true;
        out->metadata->is_leaf = false;

        GradFn* grad_fn = (GradFn*)calloc(1, sizeof(GradFn));
        if (!grad_fn) { free_tensor(out); return NULL; }

        grad_fn->backward = backward_log_softmax_fn;
        grad_fn->num_inputs = 2;
        grad_fn->inputs = (Tensor**)malloc(2 * sizeof(Tensor*));
        if (!grad_fn->inputs) { free(grad_fn); free_tensor(out); return NULL; }
        grad_fn->inputs[0] = x;
        grad_fn->inputs[1] = out;

        SumDimSavedData* saved = (SumDimSavedData*)malloc(sizeof(SumDimSavedData));
        if (!saved) { free(grad_fn->inputs); free(grad_fn); free_tensor(out); return NULL; }
        saved->outer_size = outer_size;
        saved->dim_size = dim_size;
        saved->inner_size = inner_size;
        saved->input_ndim = x->ndim;
        saved->input_shape = (int*)malloc(x->ndim * sizeof(int));
        if (!saved->input_shape) { free(saved); free(grad_fn->inputs); free(grad_fn); free_tensor(out); return NULL; }
        memcpy(saved->input_shape, x->shape, x->ndim * sizeof(int));
        grad_fn->saved_data = saved;

        out->metadata->grad_fn = grad_fn;
    }

    return out;
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
        void* scalar_ptr = (x->dtype == DTYPE_FLOAT64) ? (void*)&saved->scalar_f64 : (void*)&saved->scalar_f32;

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
        void* scalar_ptr = (x->dtype == DTYPE_FLOAT64) ? (void*)&saved->scalar_f64 : (void*)&saved->scalar_f32;

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
        if (x->dtype == DTYPE_FLOAT64) {
            saved->scalar_f64 = *(const double*)scalar;
            saved->scalar_f32 = (float)saved->scalar_f64;
        } else {
            saved->scalar_f32 = *(const float*)scalar;
            saved->scalar_f64 = (double)saved->scalar_f32;
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
        if (x->dtype == DTYPE_FLOAT64) {
            saved->scalar_f64 = *(const double*)scalar;
            saved->scalar_f32 = (float)saved->scalar_f64;
        } else {
            saved->scalar_f32 = *(const float*)scalar;
            saved->scalar_f64 = (double)saved->scalar_f32;
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
        void* exp_m1_ptr = (x->dtype == DTYPE_FLOAT64) ? (void*)&exp_m1_f64 : (void*)&exp_m1_f32;
        void* scalar_ptr = (x->dtype == DTYPE_FLOAT64) ? (void*)&saved->scalar_f64 : (void*)&saved->scalar_f32;

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
        if (x->dtype == DTYPE_FLOAT64) {
            saved->scalar_f64 = *(const double*)scalar;
            saved->scalar_f32 = (float)saved->scalar_f64;
        } else {
            saved->scalar_f32 = *(const float*)scalar;
            saved->scalar_f64 = (double)saved->scalar_f32;
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
        void* scalar_ptr = (x->dtype == DTYPE_FLOAT64) ? (void*)&saved->scalar_f64 : (void*)&saved->scalar_f32;

        Tensor* x_sq = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
        if (!x_sq) return;
        rp_mul(x_sq->data, x->data, x->data, x->size, x->dtype, x->device_id);

        Tensor* neg_c_over_x2 = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
        if (!neg_c_over_x2) { free_tensor(x_sq); return; }
        rp_rdiv_scalar(neg_c_over_x2->data, scalar_ptr, x_sq->data, x->size, x->dtype, x->device_id);
        free_tensor(x_sq);

        float neg1_f32 = -1.0f;
        double neg1_f64 = -1.0;
        void* neg1_ptr = (x->dtype == DTYPE_FLOAT64) ? (void*)&neg1_f64 : (void*)&neg1_f32;
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
        if (x->dtype == DTYPE_FLOAT64) {
            saved->scalar_f64 = *(const double*)scalar;
            saved->scalar_f32 = (float)saved->scalar_f64;
        } else {
            saved->scalar_f32 = *(const float*)scalar;
            saved->scalar_f64 = (double)saved->scalar_f32;
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
        void* ln_c_ptr = (x->dtype == DTYPE_FLOAT64) ? (void*)&ln_c_f64 : (void*)&ln_c_f32;

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
        if (x->dtype == DTYPE_FLOAT64) {
            saved->scalar_f64 = *(const double*)scalar;
            saved->scalar_f32 = (float)saved->scalar_f64;
        } else {
            saved->scalar_f32 = *(const float*)scalar;
            saved->scalar_f64 = (double)saved->scalar_f32;
        }
        grad_fn->saved_data = saved;

        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

void backward_logb_scalar_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];
    ScalarSavedData* saved = (ScalarSavedData*)self->saved_data;

    if (x->metadata && x->metadata->requires_grad) {
        float ln_c_f32 = logf(saved->scalar_f32);
        double ln_c_f64 = log(saved->scalar_f64);
        void* ln_c_ptr = (x->dtype == DTYPE_FLOAT64) ? (void*)&ln_c_f64 : (void*)&ln_c_f32;

        Tensor* x_ln_c = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
        if (!x_ln_c) return;
        rp_mul_scalar(x_ln_c->data, x->data, ln_c_ptr, x->size, x->dtype, x->device_id);

        Tensor* local_grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
        if (!local_grad) { free_tensor(x_ln_c); return; }
        rp_divide(local_grad->data, grad_output->data, x_ln_c->data, x->size, x->dtype, x->device_id);
        free_tensor(x_ln_c);

        if (!x->metadata->grad) {
            x->metadata->grad = local_grad;
        } else {
            rp_add(x->metadata->grad->data, x->metadata->grad->data, local_grad->data,
                   x->size, x->dtype, x->device_id);
            free_tensor(local_grad);
        }
    }
}

Tensor* op_logb_scalar(Tensor* x, const void* scalar) {
    if (!x || !scalar) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    int result = rp_logb_scalar(out->data, x->data, scalar, x->size, x->dtype, x->device_id);
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

        grad_fn->backward = backward_logb_scalar_fn;
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
        if (x->dtype == DTYPE_FLOAT64) {
            saved->scalar_f64 = *(const double*)scalar;
            saved->scalar_f32 = (float)saved->scalar_f64;
        } else {
            saved->scalar_f32 = *(const float*)scalar;
            saved->scalar_f64 = (double)saved->scalar_f32;
        }
        grad_fn->saved_data = saved;

        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

void backward_rlogb_scalar_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* x = self->inputs[0];
    ScalarSavedData* saved = (ScalarSavedData*)self->saved_data;

    if (x->metadata && x->metadata->requires_grad) {
        float neg_ln_c_f32 = -logf(saved->scalar_f32);
        double neg_ln_c_f64 = -log(saved->scalar_f64);
        void* neg_ln_c_ptr = (x->dtype == DTYPE_FLOAT64) ? (void*)&neg_ln_c_f64 : (void*)&neg_ln_c_f32;

        Tensor* ln_x = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
        if (!ln_x) return;
        rp_log(ln_x->data, x->data, x->size, x->dtype, x->device_id);

        Tensor* ln_x_sq = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
        if (!ln_x_sq) { free_tensor(ln_x); return; }
        rp_mul(ln_x_sq->data, ln_x->data, ln_x->data, x->size, x->dtype, x->device_id);
        free_tensor(ln_x);

        Tensor* x_ln_x_sq = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
        if (!x_ln_x_sq) { free_tensor(ln_x_sq); return; }
        rp_mul(x_ln_x_sq->data, x->data, ln_x_sq->data, x->size, x->dtype, x->device_id);
        free_tensor(ln_x_sq);

        Tensor* deriv = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
        if (!deriv) { free_tensor(x_ln_x_sq); return; }
        rp_rdiv_scalar(deriv->data, neg_ln_c_ptr, x_ln_x_sq->data, x->size, x->dtype, x->device_id);
        free_tensor(x_ln_x_sq);

        Tensor* local_grad = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
        if (!local_grad) { free_tensor(deriv); return; }
        rp_mul(local_grad->data, grad_output->data, deriv->data, x->size, x->dtype, x->device_id);
        free_tensor(deriv);

        if (!x->metadata->grad) {
            x->metadata->grad = local_grad;
        } else {
            rp_add(x->metadata->grad->data, x->metadata->grad->data, local_grad->data,
                   x->size, x->dtype, x->device_id);
            free_tensor(local_grad);
        }
    }
}

Tensor* op_rlogb_scalar(Tensor* x, const void* scalar) {
    if (!x || !scalar) return NULL;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) return NULL;

    int result = rp_rlogb_scalar(out->data, scalar, x->data, x->size, x->dtype, x->device_id);
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

        grad_fn->backward = backward_rlogb_scalar_fn;
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
        if (x->dtype == DTYPE_FLOAT64) {
            saved->scalar_f64 = *(const double*)scalar;
            saved->scalar_f32 = (float)saved->scalar_f64;
        } else {
            saved->scalar_f32 = *(const float*)scalar;
            saved->scalar_f64 = (double)saved->scalar_f32;
        }
        grad_fn->saved_data = saved;

        out->metadata->grad_fn = grad_fn;
    }

    return out;
}

void backward_matmul_fn(GradFn* self, Tensor* grad_output) {
    if (!self || !grad_output) return;

    Tensor* a = self->inputs[0];
    Tensor* b = self->inputs[1];

    int m = a->shape[0];
    int k = a->shape[1];
    int n = b->shape[1];

    if (a->metadata && a->metadata->requires_grad) {
        Tensor* bt = rp_transpose(b, 0, 1);
        Tensor* bt_c = rp_contiguous(bt);
        free_tensor(bt);

        Tensor* grad_a = zeros_tensor(a->dtype, a->device_id, a->ndim, a->shape, NULL);
        if (grad_a && bt_c) {
            rp_matmul(grad_a->data, grad_output->data, bt_c->data, m, n, k,
                     a->dtype, a->device_id);
        }
        if (bt_c) free_tensor(bt_c);

        if (grad_a) {
            if (!a->metadata->grad) {
                a->metadata->grad = grad_a;
            } else {
                rp_add(a->metadata->grad->data, a->metadata->grad->data, grad_a->data,
                       a->size, a->dtype, a->device_id);
                free_tensor(grad_a);
            }
        }
    }

    if (b->metadata && b->metadata->requires_grad) {
        Tensor* at = rp_transpose(a, 0, 1);
        Tensor* at_c = rp_contiguous(at);
        free_tensor(at);

        Tensor* grad_b = zeros_tensor(b->dtype, b->device_id, b->ndim, b->shape, NULL);
        if (grad_b && at_c) {
            rp_matmul(grad_b->data, at_c->data, grad_output->data, k, m, n,
                     b->dtype, b->device_id);
        }
        if (at_c) free_tensor(at_c);

        if (grad_b) {
            if (!b->metadata->grad) {
                b->metadata->grad = grad_b;
            } else {
                rp_add(b->metadata->grad->data, b->metadata->grad->data, grad_b->data,
                       b->size, b->dtype, b->device_id);
                free_tensor(grad_b);
            }
        }
    }
}

Tensor* op_matmul(Tensor* a, Tensor* b) {
    if (!a || !b) return NULL;
    if (a->ndim != 2 || b->ndim != 2) return NULL;
    if (a->shape[1] != b->shape[0]) return NULL;
    if (a->dtype != b->dtype || a->device_id != b->device_id) return NULL;

    int m = a->shape[0];
    int k = a->shape[1];
    int n = b->shape[1];

    int out_shape[2] = {m, n};
    Tensor* out = zeros_tensor(a->dtype, a->device_id, 2, out_shape, NULL);
    if (!out) return NULL;

    int result = rp_matmul(out->data, a->data, b->data, m, k, n, a->dtype, a->device_id);
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

        grad_fn->backward = backward_matmul_fn;
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

int broadcast_shapes(int ndim1, const int* shape1, int ndim2, const int* shape2,
                     int* out_ndim, int** out_shape) {
    int max_ndim = (ndim1 > ndim2) ? ndim1 : ndim2;
    int* result = (int*)malloc(max_ndim * sizeof(int));
    if (!result) return -1;

    for (int i = 0; i < max_ndim; i++) {
        int d1_idx = ndim1 - max_ndim + i;
        int d2_idx = ndim2 - max_ndim + i;
        int s1 = (d1_idx >= 0) ? shape1[d1_idx] : 1;
        int s2 = (d2_idx >= 0) ? shape2[d2_idx] : 1;

        if (s1 == s2) {
            result[i] = s1;
        } else if (s1 == 1) {
            result[i] = s2;
        } else if (s2 == 1) {
            result[i] = s1;
        } else {
            free(result);
            return -1;
        }
    }

    *out_ndim = max_ndim;
    *out_shape = result;
    return 0;
}

Tensor* reduce_grad_to_shape(Tensor* grad, int target_ndim, const int* target_shape) {
    if (!grad || !target_shape || target_ndim <= 0) return NULL;

    int grad_ndim = grad->ndim;
    int target_device = grad->device_id;

    Tensor* grad_host = grad;
    bool need_free_grad_host = false;
    if (grad->device_id != -1) {
        grad_host = tensor_to(grad, -1, grad->dtype, false);
        if (!grad_host) return NULL;
        need_free_grad_host = true;
    }

    int padded_ndim = grad_ndim;
    int* padded_shape = (int*)malloc(padded_ndim * sizeof(int));
    if (!padded_shape) {
        if (need_free_grad_host) free_tensor(grad_host);
        return NULL;
    }
    int offset = grad_ndim - target_ndim;
    for (int i = 0; i < padded_ndim; i++) {
        padded_shape[i] = (i < offset) ? 1 : target_shape[i - offset];
    }

    Tensor* result = zeros_tensor(grad->dtype, -1, target_ndim, (int*)target_shape, NULL);
    if (!result) {
        free(padded_shape);
        if (need_free_grad_host) free_tensor(grad_host);
        return NULL;
    }

    int* input_strides = (int*)malloc(target_ndim * sizeof(int));
    int* output_strides = (int*)malloc(padded_ndim * sizeof(int));
    if (!input_strides || !output_strides) {
        free(input_strides);
        free(output_strides);
        free(padded_shape);
        free_tensor(result);
        if (need_free_grad_host) free_tensor(grad_host);
        return NULL;
    }

    input_strides[target_ndim - 1] = 1;
    for (int d = target_ndim - 2; d >= 0; d--) {
        input_strides[d] = input_strides[d + 1] * target_shape[d + 1];
    }

    output_strides[padded_ndim - 1] = 1;
    for (int d = padded_ndim - 2; d >= 0; d--) {
        output_strides[d] = output_strides[d + 1] * grad_host->shape[d + 1];
    }

    size_t output_size = grad_host->size;

    bool is_half = (grad->dtype == DTYPE_FLOAT16 || grad->dtype == DTYPE_BFLOAT16);

    if (is_half || grad->dtype == DTYPE_FLOAT32) {
        float* res = NULL;
        const float* g = NULL;
        float* res_alloc = NULL;
        float* g_alloc = NULL;

        if (is_half) {
            res_alloc = (float*)calloc(result->size, sizeof(float));
            g_alloc = (float*)malloc(output_size * sizeof(float));
            if (!res_alloc || !g_alloc) {
                free(res_alloc); free(g_alloc);
                goto reduce_cleanup;
            }
            half_to_fp32_array(grad_host->data, g_alloc, output_size, grad->dtype);
            res = res_alloc;
            g = g_alloc;
        } else {
            res = (float*)result->data;
            g = (const float*)grad_host->data;
        }

        for (size_t i = 0; i < output_size; i++) {
            size_t out_idx = i;
            size_t in_idx = 0;

            for (int d = 0; d < padded_ndim; d++) {
                int coord = out_idx / output_strides[d];
                out_idx %= output_strides[d];
                if (d >= offset) {
                    int in_coord = (padded_shape[d] == 1) ? 0 : coord;
                    in_idx += in_coord * input_strides[d - offset];
                }
            }
            res[in_idx] += g[i];
        }

        if (is_half) {
            fp32_to_half_array(res_alloc, result->data, result->size, grad->dtype);
            free(res_alloc);
            free(g_alloc);
        }
    } else {
        double* res = (double*)result->data;
        const double* g = (const double*)grad_host->data;

        for (size_t i = 0; i < output_size; i++) {
            size_t out_idx = i;
            size_t in_idx = 0;

            for (int d = 0; d < padded_ndim; d++) {
                int coord = out_idx / output_strides[d];
                out_idx %= output_strides[d];
                if (d >= offset) {
                    int in_coord = (padded_shape[d] == 1) ? 0 : coord;
                    in_idx += in_coord * input_strides[d - offset];
                }
            }
            res[in_idx] += g[i];
        }
    }

reduce_cleanup:
    free(input_strides);
    free(output_strides);
    free(padded_shape);
    if (need_free_grad_host) free_tensor(grad_host);

    if (target_device != -1) {
        Tensor* result_device = tensor_to(result, target_device, result->dtype, false);
        free_tensor(result);
        return result_device;
    }

    return result;
}

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
        } else if (grad_fn->backward == backward_sum_dim_fn ||
                   grad_fn->backward == backward_mean_dim_fn ||
                   grad_fn->backward == backward_softmax_fn ||
                   grad_fn->backward == backward_log_softmax_fn) {
            SumDimSavedData* saved = (SumDimSavedData*)grad_fn->saved_data;
            if (saved->input_shape) free(saved->input_shape);
        } else if (grad_fn->backward == backward_nll_loss_fn ||
                   grad_fn->backward == backward_cross_entropy_loss_fn) {
            CELossSavedData* saved = (CELossSavedData*)grad_fn->saved_data;
            if (saved->targets) free(saved->targets);
        } else if (grad_fn->backward == backward_gather_fn) {
            GatherSavedData* saved = (GatherSavedData*)grad_fn->saved_data;
            if (saved->indices) free(saved->indices);
            if (saved->d_indices) cudaFree(saved->d_indices);
            if (saved->index_shape) free(saved->index_shape);
            if (saved->input_shape) free(saved->input_shape);
        } else if (grad_fn->backward == backward_repeat_fn) {
            RepeatSavedData* saved = (RepeatSavedData*)grad_fn->saved_data;
            if (saved->repeats) free(saved->repeats);
            if (saved->input_shape) free(saved->input_shape);
        } else if (grad_fn->backward == backward_add_fn ||
                   grad_fn->backward == backward_sub_fn ||
                   grad_fn->backward == backward_mul_fn ||
                   grad_fn->backward == backward_div_fn ||
                   grad_fn->backward == backward_power_fn ||
                   grad_fn->backward == backward_logb_fn) {
            BroadcastSavedData* saved = (BroadcastSavedData*)grad_fn->saved_data;
            if (saved->x1_shape) free(saved->x1_shape);
            if (saved->x2_shape) free(saved->x2_shape);
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
            size_t elem_size = dtype_size(input->dtype);
            bool is_half = (input->dtype == DTYPE_FLOAT16 || input->dtype == DTYPE_BFLOAT16);

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

                float* grad_f32 = NULL;
                float* input_grad_f32 = NULL;
                if (is_half) {
                    grad_f32 = (float*)malloc(grad_output->size * sizeof(float));
                    input_grad_f32 = (float*)malloc(input->metadata->grad->size * sizeof(float));
                    if (!grad_f32 || !input_grad_f32) {
                        free(grad_f32); free(input_grad_f32); free(indices);
                        offset += slice_size;
                        continue;
                    }
                    half_to_fp32_array(grad_output->data, grad_f32, grad_output->size, input->dtype);
                    half_to_fp32_array(input->metadata->grad->data, input_grad_f32, input->metadata->grad->size, input->dtype);
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

                    if (is_half) {
                        input_grad_f32[input_idx] += grad_f32[grad_idx];
                    } else if (input->dtype == DTYPE_FLOAT32) {
                        ((float*)input->metadata->grad->data)[input_idx] += ((float*)grad_output->data)[grad_idx];
                    } else {
                        ((double*)input->metadata->grad->data)[input_idx] += ((double*)grad_output->data)[grad_idx];
                    }

                    free(grad_shape);
                }

                if (is_half) {
                    fp32_to_half_array(input_grad_f32, input->metadata->grad->data, input->metadata->grad->size, input->dtype);
                    free(grad_f32);
                    free(input_grad_f32);
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

    size_t elem_size = dtype_size(input->dtype);
    bool is_half = (input->dtype == DTYPE_FLOAT16 || input->dtype == DTYPE_BFLOAT16);

    float* grad_f32 = NULL;
    float* input_grad_f32 = NULL;
    if (is_half) {
        grad_f32 = (float*)malloc(grad_output->size * sizeof(float));
        input_grad_f32 = (float*)malloc(input->metadata->grad->size * sizeof(float));
        if (!grad_f32 || !input_grad_f32) {
            free(grad_f32); free(input_grad_f32);
            return;
        }
        half_to_fp32_array(grad_output->data, grad_f32, grad_output->size, input->dtype);
        half_to_fp32_array(input->metadata->grad->data, input_grad_f32, input->metadata->grad->size, input->dtype);
    }

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

        if (is_half) {
            input_grad_f32[input_idx] += grad_f32[idx];
        } else if (input->dtype == DTYPE_FLOAT32) {
            ((float*)input->metadata->grad->data)[input_idx] += ((float*)grad_output->data)[idx];
        } else {
            ((double*)input->metadata->grad->data)[input_idx] += ((double*)grad_output->data)[idx];
        }

        free(grad_indices);
        free(input_indices);
    }

    if (is_half) {
        fp32_to_half_array(input_grad_f32, input->metadata->grad->data, input->metadata->grad->size, input->dtype);
        free(grad_f32);
        free(input_grad_f32);
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

    bool is_half = (input->dtype == DTYPE_FLOAT16 || input->dtype == DTYPE_BFLOAT16);

    if (is_half || input->dtype == DTYPE_FLOAT32) {
        float* grad_in = NULL;
        const float* grad_out = NULL;
        float* grad_in_alloc = NULL;
        float* grad_out_alloc = NULL;

        if (is_half) {
            grad_in_alloc = (float*)calloc(grad_input->size, sizeof(float));
            grad_out_alloc = (float*)malloc(output_size * sizeof(float));
            if (!grad_in_alloc || !grad_out_alloc) {
                free(grad_in_alloc); free(grad_out_alloc);
                goto expand_cleanup;
            }
            half_to_fp32_array(grad_out_host->data, grad_out_alloc, output_size, input->dtype);
            grad_in = grad_in_alloc;
            grad_out = grad_out_alloc;
        } else {
            grad_in = (float*)grad_input->data;
            grad_out = (const float*)grad_out_host->data;
        }

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

        if (is_half) {
            fp32_to_half_array(grad_in_alloc, grad_input->data, grad_input->size, input->dtype);
            free(grad_in_alloc);
            free(grad_out_alloc);
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

expand_cleanup:
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

    bool is_half = (input->dtype == DTYPE_FLOAT16 || input->dtype == DTYPE_BFLOAT16);

    if (is_half || input->dtype == DTYPE_FLOAT32) {
        float* grad_in = NULL;
        const float* grad_out = NULL;
        float* grad_in_alloc = NULL;
        float* grad_out_alloc = NULL;

        if (is_half) {
            grad_in_alloc = (float*)malloc(input->metadata->grad->size * sizeof(float));
            grad_out_alloc = (float*)malloc(grad_size * sizeof(float));
            if (!grad_in_alloc || !grad_out_alloc) {
                free(grad_in_alloc); free(grad_out_alloc);
                return;
            }
            half_to_fp32_array(input->metadata->grad->data, grad_in_alloc, input->metadata->grad->size, input->dtype);
            half_to_fp32_array(grad_output->data, grad_out_alloc, grad_size, input->dtype);
            grad_in = grad_in_alloc;
            grad_out = grad_out_alloc;
        } else {
            grad_in = (float*)input->metadata->grad->data;
            grad_out = (const float*)grad_output->data;
        }

        for (size_t i = 0; i < grad_size; i++) {
            size_t after_idx = i % after_size;
            size_t chunk_idx = (i / after_size) % chunk_size;
            size_t before_idx = i / (after_size * chunk_size);

            size_t input_idx = before_idx * (input_dim_size * after_size) +
                              (offset + chunk_idx) * after_size + after_idx;

            grad_in[input_idx] += grad_out[i];
        }

        if (is_half) {
            fp32_to_half_array(grad_in_alloc, input->metadata->grad->data, input->metadata->grad->size, input->dtype);
            free(grad_in_alloc);
            free(grad_out_alloc);
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

    bool is_half = (input->dtype == DTYPE_FLOAT16 || input->dtype == DTYPE_BFLOAT16);

    if (is_half || input->dtype == DTYPE_FLOAT32) {
        float* grad_in = NULL;
        const float* grad_out = NULL;
        float* grad_in_alloc = NULL;
        float* grad_out_alloc = NULL;

        if (is_half) {
            grad_in_alloc = (float*)calloc(grad_input->size, sizeof(float));
            grad_out_alloc = (float*)malloc(output_size * sizeof(float));
            if (!grad_in_alloc || !grad_out_alloc) {
                free(grad_in_alloc); free(grad_out_alloc);
                goto repeat_cleanup;
            }
            half_to_fp32_array(grad_out_host->data, grad_out_alloc, output_size, input->dtype);
            grad_in = grad_in_alloc;
            grad_out = grad_out_alloc;
        } else {
            grad_in = (float*)grad_input->data;
            grad_out = (const float*)grad_out_host->data;
        }

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

        if (is_half) {
            fp32_to_half_array(grad_in_alloc, grad_input->data, grad_input->size, input->dtype);
            free(grad_in_alloc);
            free(grad_out_alloc);
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

repeat_cleanup:
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

Tensor** op_split(Tensor* src, int* sizes, int num_splits, int dim) {
    if (!src) return NULL;

    Tensor** result = rp_split(src, sizes, num_splits, dim);
    if (!result) return NULL;

    bool requires_grad = (src->metadata && src->metadata->requires_grad);

    if (requires_grad) {
        int offset = 0;
        for (int i = 0; i < num_splits; i++) {
            GradFn* grad_fn = (GradFn*)calloc(1, sizeof(GradFn));
            if (!grad_fn) {
                for (int j = 0; j < num_splits; j++) {
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
                for (int j = 0; j < num_splits; j++) {
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
                for (int j = 0; j < num_splits; j++) {
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
                for (int j = 0; j < num_splits; j++) {
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
                    for (int j = 0; j < num_splits; j++) {
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

Tensor** op_split_equal(Tensor* src, int num_splits, int dim) {
    if (!src) return NULL;

    Tensor** result = rp_split_equal(src, num_splits, dim);
    if (!result) return NULL;

    bool requires_grad = (src->metadata && src->metadata->requires_grad);

    if (requires_grad) {
        int offset = 0;
        for (int i = 0; i < num_splits; i++) {
            GradFn* grad_fn = (GradFn*)calloc(1, sizeof(GradFn));
            if (!grad_fn) {
                for (int j = 0; j < num_splits; j++) {
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
                for (int j = 0; j < num_splits; j++) {
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
                for (int j = 0; j < num_splits; j++) {
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
                for (int j = 0; j < num_splits; j++) {
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
                    for (int j = 0; j < num_splits; j++) {
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
