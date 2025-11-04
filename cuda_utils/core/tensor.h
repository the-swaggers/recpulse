#ifndef TENSOR_H
#define TENSOR_H

#include <stdbool.h>
#include <stddef.h>

typedef enum {
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
} DType;


typedef enum {
    DEVICE_CPU,
    DEVICE_CUDA,
} DeviceType;


typedef struct Tensor Tensor;

typedef struct {
    bool is_leaf;
    bool requires_grad;
    bool is_frozen;
    void* grad_fn;
    Tensor* grad;
} Meta;


struct Tensor {
    DType dtype;
    void* data;
    int ndim;
    size_t size;
    int* shape;
    int* strides;
    DeviceType device;
    int device_id;
    bool owns_data;
    
    Meta* metadata;
}; 

#endif
