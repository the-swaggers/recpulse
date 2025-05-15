#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef TENSOR_H
#define TENSOR_H

typedef enum {
    DTYPE_FLOAT8,
    DTYPE_FLOAT16,
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
} DType;

typedef enum {
    DEVICE_CPU,
    DEVICE_CUDA,
} DeviceType;

typedef struct {
    size_t size;
    uint ndim;
    int* shape;
    void* vals;
    DType dtype;
    DeviceType device_type;
    uint32_t device_id;
    requires_grad bool;
    owns_data bool;
} Tensor;


#endif
