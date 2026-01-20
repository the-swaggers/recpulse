#include "tensor.h"
#include "../ops/ops.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <cuda_runtime.h>


bool validate_device_id(int device_id) {
    if (device_id == -1) return true;

    int count;
    cudaGetDeviceCount(&count);
    return device_id >= 0 && device_id < count;
}

Tensor* zeros_tensor(DType dtype, int device_id, int ndim, int* shape, Meta* metadata) {
    if (!validate_device_id(device_id)) return NULL;

    if (device_id == -1) {
        return zeros_host_tensor(dtype, ndim, shape, metadata);
    }
    return zeros_device_tensor(dtype, device_id, ndim, shape, metadata);
}

Tensor* ones_tensor(DType dtype, int device_id, int ndim, int* shape, Meta* metadata) {
    if (!validate_device_id(device_id)) return NULL;

    if (device_id == -1) {
        return ones_host_tensor(dtype, ndim, shape, metadata);
    }
    return ones_device_tensor(dtype, device_id, ndim, shape, metadata);
}

Tensor* values_tensor(void* vals, DType vals_dtype, DType target_dtype, int source_device_id, int device_id, int ndim, int* shape, Meta* metadata) {
    if (!validate_device_id(device_id)) return NULL;

    if (device_id == -1) {
        return values_host_tensor(vals, vals_dtype, target_dtype, ndim, shape, metadata);
    }
    return values_device_tensor(vals, vals_dtype, target_dtype, source_device_id, ndim, shape, device_id, metadata);
}

Tensor* fill_value_tensor(double value, Tensor* tensor) {
    if (!tensor) return NULL;

    if (tensor->device_id == -1) {
        return fill_value_host_tensor(value, tensor);
    }
    return fill_value_device_tensor(value, tensor);
}

Tensor* tensor_copy(Tensor* tensor) {
    if (!tensor) return NULL;

    if (tensor->device_id == -1) {
        return tensor_copy_host(tensor, tensor->dtype);
    }
    if (tensor->device_id >= 0) {
        return tensor_copy_device(tensor, tensor->device_id, tensor->dtype);
    }

    fprintf(stderr, "Error: Invalid device_id %d in tensor_copy\n", tensor->device_id);
    return NULL;
}

Tensor* tensor_to(Tensor* src, int target_device_id, DType target_dtype, bool inplace) {
    if (!src) return NULL;

    if (!validate_device_id(target_device_id)) {
        fprintf(stderr, "Error: Invalid target_device_id %d\n", target_device_id);
        return NULL;
    }

    bool same_device = (src->device_id == target_device_id);
    bool same_dtype = (src->dtype == target_dtype);

    if (same_device && same_dtype) {
        if (inplace) {
            return src;
        } else {
            return tensor_copy(src);
        }
    }

    Tensor* result = NULL;

    bool src_is_host = (src->device_id == -1);
    bool target_is_host = (target_device_id == -1);

    if (src_is_host && target_is_host) {
        result = tensor_copy_host(src, target_dtype);
    } else if (!src_is_host && !target_is_host) {
        result = tensor_copy_device(src, target_device_id, target_dtype);
    } else if (src_is_host && !target_is_host) {
        result = move_host_to_device(src, target_device_id, target_dtype);
    } else if (!src_is_host && target_is_host) {
        result = move_device_to_host(src, target_dtype);
    } else {
        fprintf(stderr, "Error: Invalid device combination in tensor_to\n");
        return NULL;
    }

    if (!result) {
        return NULL;
    }

    if (inplace) {
        free_tensor(src);
    }

    return result;
}

Tensor* tensor_reshape(Tensor* tensor, int new_ndim, int* new_shape) {
    if (!tensor || !new_shape || new_ndim <= 0) {
        fprintf(stderr, "Error: Invalid arguments to tensor_reshape\n");
        return NULL;
    }

    int* inferred_shape = (int*)malloc(new_ndim * sizeof(int));
    if (!inferred_shape) {
        fprintf(stderr, "Error: Failed to allocate memory for inferred_shape\n");
        return NULL;
    }

    int infer_dim = -1;
    size_t known_elements = 1;

    for (int i = 0; i < new_ndim; i++) {
        if (new_shape[i] == -1) {
            if (infer_dim != -1) {
                fprintf(stderr, "Error: Only one dimension can be -1\n");
                free(inferred_shape);
                return NULL;
            }
            infer_dim = i;
            inferred_shape[i] = -1;
        } else if (new_shape[i] <= 0) {
            fprintf(stderr, "Error: Invalid dimension size %d\n", new_shape[i]);
            free(inferred_shape);
            return NULL;
        } else {
            inferred_shape[i] = new_shape[i];
            known_elements *= new_shape[i];
        }
    }

    if (infer_dim != -1) {
        if (tensor->size % known_elements != 0) {
            fprintf(stderr, "Error: Cannot infer dimension, size mismatch\n");
            free(inferred_shape);
            return NULL;
        }
        inferred_shape[infer_dim] = tensor->size / known_elements;
    } else {
        if (known_elements != tensor->size) {
            fprintf(stderr, "Error: New shape has %zu elements but tensor has %zu\n",
                    known_elements, tensor->size);
            free(inferred_shape);
            return NULL;
        }
    }

    Tensor* result = NULL;
    if (tensor->device_id == -1) {
        result = tensor_reshape_host(tensor, new_ndim, inferred_shape);
    } else {
        result = tensor_reshape_device(tensor, new_ndim, inferred_shape);
    }

    free(inferred_shape);
    return result;
}

int tensor_backward(Tensor* tensor) {
    if (!tensor) return -1;
    if (!tensor->metadata || !tensor->metadata->requires_grad) return -1;

    if (!tensor->metadata->grad) {
        tensor->metadata->grad = ones_tensor(tensor->dtype, tensor->device_id, tensor->ndim, tensor->shape, NULL);
        if (!tensor->metadata->grad) {
            return -1;
        }
    }

    if (tensor->metadata->grad_fn) {
        tensor->metadata->grad_fn->backward(tensor->metadata->grad_fn, tensor->metadata->grad);
    }

    return 0;
}

void free_tensor(Tensor* tensor){
    if (!tensor) return;
    if (tensor->device_id == -1) return free_tensor_host(tensor);
    if (tensor->device_id >= 0) return free_tensor_device(tensor);

    fprintf(stderr, "Error: Invalid device_id %d in free_tensor\n", tensor->device_id);
    exit(1);
};

