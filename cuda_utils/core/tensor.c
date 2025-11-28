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
        fprintf(stderr, "Error: tensor_copy for DEVICE not yet implemented\n");
        return NULL;
    }

    fprintf(stderr, "Error: Invalid device type %d in tensor_copy\n", tensor->device);
    return NULL;
}

void free_tensor(Tensor* tensor){
    if (!tensor) return;
    if (tensor->device == HOST) return free_tensor_host(tensor);
    if (tensor->device == DEVICE) return free_tensor_device(tensor);

    fprintf(stderr, "Error: Invalid device type %d in free_tensor\n", tensor->device);
    exit(1);
};
