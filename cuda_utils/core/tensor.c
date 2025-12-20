#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <cuda_runtime.h>


bool validate_device_id(DeviceType device, int device_id) {
    if (device == HOST) return device_id == 0;

    int count;
    cudaGetDeviceCount(&count);
    return device_id >= 0 && device_id < count;
}

Tensor* zeros_tensor(DType dtype, DeviceType device, int device_id, int ndim, int* shape, Meta* metadata) {
    if (!validate_device_id(device, device_id)) return NULL;

    if (device == HOST) {
        return zeros_host_tensor(dtype, ndim, shape, metadata);
    }
    return zeros_device_tensor(dtype, device_id, ndim, shape, metadata);
}

Tensor* ones_tensor(DType dtype, DeviceType device, int device_id, int ndim, int* shape, Meta* metadata) {
    if (!validate_device_id(device, device_id)) return NULL;

    if (device == HOST) {
        return ones_host_tensor(dtype, ndim, shape, metadata);
    }
    return ones_device_tensor(dtype, device_id, ndim, shape, metadata);
}

Tensor* values_tensor(void* vals, DType vals_dtype, DType target_dtype, DeviceType source_device, DeviceType target_device, int device_id, int ndim, int* shape, Meta* metadata) {
    if (!validate_device_id(target_device, device_id)) return NULL;

    if (target_device == HOST) {
        return values_host_tensor(vals, vals_dtype, target_dtype, ndim, shape, metadata);
    }
    return values_device_tensor(vals, vals_dtype, target_dtype, source_device, ndim, shape, device_id, metadata);
}

Tensor* fill_value_tensor(double value, Tensor* tensor) {
    if (!tensor) return NULL;

    if (tensor->device == HOST) {
        return fill_value_host_tensor(value, tensor);
    }
    return fill_value_device_tensor(value, tensor);
}

Tensor* tensor_copy(Tensor* tensor) {
    if (!tensor) return NULL;

    if (tensor->device == HOST) {
        return tensor_copy_host(tensor, tensor->dtype);
    }
    if (tensor->device == DEVICE) {
        return tensor_copy_device(tensor, tensor->device_id, tensor->dtype);
    }

    fprintf(stderr, "Error: Invalid device type %d in tensor_copy\n", tensor->device);
    return NULL;
}

Tensor* tensor_to(Tensor* src, DeviceType target_device, int target_device_id, DType target_dtype, bool inplace) {
    if (!src) return NULL;

    if (!validate_device_id(target_device, target_device_id)) {
        fprintf(stderr, "Error: Invalid device_id %d for device type %d\n", target_device_id, target_device);
        return NULL;
    }

    bool same_device_type = (src->device == target_device);
    bool same_device_id = (src->device_id == target_device_id);
    bool same_dtype = (src->dtype == target_dtype);

    if (same_device_type && same_device_id && same_dtype) {
        if (inplace) {
            return src;
        } else {
            return tensor_copy(src);
        }
    }

    Tensor* result = NULL;

    if (src->device == HOST && target_device == HOST) {
        result = tensor_copy_host(src, target_dtype);
    } else if (src->device == DEVICE && target_device == DEVICE) {
        result = tensor_copy_device(src, target_device_id, target_dtype);
    } else if (src->device == HOST && target_device == DEVICE) {
        result = move_host_to_device(src, target_device_id, target_dtype);
    } else if (src->device == DEVICE && target_device == HOST) {
        result = move_device_to_host(src, target_dtype);
    } else {
        fprintf(stderr, "Error: Invalid device type combination in tensor_to\n");
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

void free_tensor(Tensor* tensor){
    if (!tensor) return;
    if (tensor->device == HOST) return free_tensor_host(tensor);
    if (tensor->device == DEVICE) return free_tensor_device(tensor);

    fprintf(stderr, "Error: Invalid device type %d in free_tensor\n", tensor->device);
    exit(1);
};
