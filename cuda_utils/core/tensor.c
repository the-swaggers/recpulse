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

