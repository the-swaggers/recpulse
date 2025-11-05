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

Tensor* zeros_cpu_tensor(DType dtype, int ndim, int* shape, Meta* metadata);
Tensor* ones_cpu_tensor(DType dtype, int ndim, int* shape, Meta* metadata);
Tensor* values_cpu_tensor(void* vals, DType dtype, int ndim, int* shape, Meta* metadata);

Tensor* zeros_cuda_tensor(DType dtype, int ndim, int* shape, Meta* metadata);
Tensor* ones_cuda_tensor(DType dtype, int ndim, int* shape, Meta* metadata);
Tensor* values_cuda_tensor(void* vals, DType dtype, int ndim, int* shape, Meta* metadata);

Tensor* move_cpu_to_cuda(Tensor* tensor, int device_id);
Tensor* move_cuda_to_cpu(Tensor* tensor);
Tensor* move_cuda_to_cuda(Tensor* tensor, int device_id);
Tensor* tensor_copy_cpu(Tensor* tensor);


Tensor* tensor_copy(Tensor* tensor);

void free_tensor(Tensor* tensor);

void free_tensor_cpu(Tensor* tensor);
void free_tensor_cuda(Tensor* tensor);

#endif
