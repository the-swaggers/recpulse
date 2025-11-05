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
