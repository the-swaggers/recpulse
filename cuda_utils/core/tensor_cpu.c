#include "tensor.h"
#include <stdlib.h>
#include <string.h>

Tensor* zeros_cpu_tensor(DType dtype, int ndim, int* shape, Meta* metadata) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) return NULL;  
    size_t total_elements = 1;
    for (int i = 0; i < ndim; i++) {
        total_elements *= shape[i];
    }

    tensor->shape = (int*)malloc(ndim * sizeof(int));
    if (!tensor->shape) {
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, ndim * sizeof(int));

    tensor->strides = (int*)malloc(ndim * sizeof(int));
    if (!tensor->strides) {
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        tensor->strides[i] = stride;
        stride *= shape[i];
    }

    size_t dtype_size = (dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);
    size_t data_size = total_elements * dtype_size;

    tensor->data = calloc(total_elements, dtype_size);
    if (!tensor->data) {
        free(tensor->strides);
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    tensor->dtype = dtype;
    tensor->ndim = ndim;
    tensor->size = total_elements;
    tensor->device = DEVICE_CPU;
    tensor->device_id = 0;
    tensor->owns_data = true;
    tensor->metadata = metadata;

    return tensor;
}

Tensor* ones_cpu_tensor(DType dtype, int ndim, int* shape, Meta* metadata) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) return NULL;

    size_t total_elements = 1;
    for (int i = 0; i < ndim; i++) {
        total_elements *= shape[i];
    }

    tensor->shape = (int*)malloc(ndim * sizeof(int));
    if (!tensor->shape) {
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, ndim * sizeof(int));

    tensor->strides = (int*)malloc(ndim * sizeof(int));
    if (!tensor->strides) {
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        tensor->strides[i] = stride;
        stride *= shape[i];
    }

    size_t dtype_size = (dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);

    tensor->data = malloc(total_elements * dtype_size);
    if (!tensor->data) {
        free(tensor->strides);
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    if (dtype == DTYPE_FLOAT32) {
        float* data = (float*)tensor->data;
        for (size_t i = 0; i < total_elements; i++) {
            data[i] = 1.0f;
        }
    } else {
        double* data = (double*)tensor->data;
        for (size_t i = 0; i < total_elements; i++) {
            data[i] = 1.0;
        }
    }

    tensor->dtype = dtype;
    tensor->ndim = ndim;
    tensor->size = total_elements;
    tensor->device = DEVICE_CPU;
    tensor->device_id = 0;
    tensor->owns_data = true;
    tensor->metadata = metadata;

    return tensor;
}

Tensor* values_cpu_tensor(void* vals, DType dtype, int ndim, int* shape, Meta* metadata) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) return NULL;

    size_t total_elements = 1;
    for (int i = 0; i < ndim; i++) {
        total_elements *= shape[i];
    }

    tensor->shape = (int*)malloc(ndim * sizeof(int));
    if (!tensor->shape) {
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, ndim * sizeof(int));

    tensor->strides = (int*)malloc(ndim * sizeof(int));
    if (!tensor->strides) {
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        tensor->strides[i] = stride;
        stride *= shape[i];
    }

    size_t dtype_size = (dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);
    size_t data_size = total_elements * dtype_size;

    tensor->data = malloc(data_size);
    if (!tensor->data) {
        free(tensor->strides);
        free(tensor->shape);
        free(tensor);
        return NULL;
    }
    memcpy(tensor->data, vals, data_size);

    tensor->dtype = dtype;
    tensor->ndim = ndim;
    tensor->size = total_elements;
    tensor->device = DEVICE_CPU;
    tensor->device_id = 0;
    tensor->owns_data = true;
    tensor->metadata = metadata;

    return tensor;
}

Tensor* tensor_copy_cpu(Tensor* tensor) {
    if (tensor->device != DEVICE_CPU) return NULL;

    Tensor* copy = (Tensor*)malloc(sizeof(Tensor));
    if (!copy) return NULL;

    copy->shape = (int*)malloc(tensor->ndim * sizeof(int));
    if (!copy->shape) {
        free(copy);
        return NULL;
    }
    memcpy(copy->shape, tensor->shape, tensor->ndim * sizeof(int));

    copy->strides = (int*)malloc(tensor->ndim * sizeof(int));
    if (!copy->strides) {
        free(copy->shape);
        free(copy);
        return NULL;
    }
    memcpy(copy->strides, tensor->strides, tensor->ndim * sizeof(int));

    size_t dtype_size = (tensor->dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);
    size_t data_size = tensor->size * dtype_size;

    copy->data = malloc(data_size);
    if (!copy->data) {
        free(copy->strides);
        free(copy->shape);
        free(copy);
        return NULL;
    }
    memcpy(copy->data, tensor->data, data_size);

    copy->dtype = tensor->dtype;
    copy->ndim = tensor->ndim;
    copy->size = tensor->size;
    copy->device = DEVICE_CPU;
    copy->device_id = 0;
    copy->owns_data = true;

    if (tensor->metadata) {
        copy->metadata = (Meta*)malloc(sizeof(Meta));
        if (copy->metadata) {
            memcpy(copy->metadata, tensor->metadata, sizeof(Meta));
            copy->metadata->grad = NULL;
            copy->metadata->grad_fn = NULL;
        }
    } else {
        copy->metadata = NULL;
    }

    return copy;
}

void free_tensor_cpu(Tensor* tensor) {
    if (!tensor) return;
    if (tensor->device != DEVICE_CPU) return;

    if (tensor->owns_data && tensor->data) {
        free(tensor->data);
    }

    if (tensor->shape) {
        free(tensor->shape);
    }

    if (tensor->strides) {
        free(tensor->strides);
    }

    if (tensor->metadata) {
        if (tensor->metadata->grad) {
            free_tensor_cpu(tensor->metadata->grad);
        }
        free(tensor->metadata);
    }

    free(tensor);
}
