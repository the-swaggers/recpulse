#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


Tensor* tensor_new(uint32_t ndim, int* shape, void* vals, DType dtype, DeviceType device_type, uint32_t device_id, bool requires_grad, bool owns_data){
    assert(!(dtype == DTYPE_FLOAT8 && device_type == DEVICE_CPU) && "CPU does not support float8 operations!");

    int nvals = 1;
    for (int i = 0; i < ndim; i++){
        nvals = nvals * shape[i];
    }
    size_t size = get_size(nvals, dtype);

    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) return NULL;

    tensor->size = size;
    tensor->ndim = ndim;
    tensor->dtype = dtype;
    tensor->device_type = device_type;
    tensor->device_id = device_id;
    tensor->requires_grad = requires_grad;
    tensor->owns_data = owns_data;

    tensor->shape = (int*)malloc(ndim * sizeof(int));
    if (!tensor->shape) {
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, ndim * sizeof(int));

    if (device_type == DEVICE_CPU) {
        if (vals && !owns_data) {
            tensor->vals = vals;
        }
        else {
            tensor->vals = malloc(size);
            if (!tensor->vals) {
                free(tensor->shape);
                free(tensor);
                return NULL;
            }
            if (vals && tensor->vals) {
                memcpy(tensor->vals, vals, size);
            }
        }
    }
    else {
        // implement CUDA here
    }

    return tensor;
}


size_t get_size(int number_of_parameters, DType dtype){
    switch (dtype) {
        case DTYPE_FLOAT8:
            return number_of_parameters;
        case DTYPE_FLOAT16:
            return number_of_parameters * 2;
        case DTYPE_FLOAT32:
            return number_of_parameters * 4;
        case DTYPE_FLOAT64:
            return number_of_parameters * 8;
        default:
            return 0;
    }
}

