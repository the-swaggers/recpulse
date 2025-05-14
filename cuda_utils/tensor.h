#include <stdbool.h>
#include <stddef.h>

#ifndef TENSOR_H
#define TENSOR_H

typedef enum {
    float8,
    float16,
    float32,
    float64,
} DType;

typedef enum {
    DEVICE_CPU,
    DEVICE_CUDA,
} DeviceType;

struct Tensor{
    size_t size;
    uint ndim;
    *int shape;
    *void vals;
    DType dtype;
    DeviceType device_type;
    uint device_id;
    requires_grad bool;
    owns_data bool;
}


#endif
