#ifndef TENSOR_H
#define TENSOR_H

#include <stdbool.h>
#include <stddef.h>

typedef enum {
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    DTYPE_PRESERVE = -1,
} DType;


typedef enum {
    HOST,
    DEVICE,
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

#ifdef __cplusplus
extern "C" {
#endif

Tensor* zeros_host_tensor(DType dtype, int ndim, int* shape, Meta* metadata);
Tensor* ones_host_tensor(DType dtype, int ndim, int* shape, Meta* metadata);
Tensor* values_host_tensor(void* vals, DType vals_dtype, DType target_dtype, int ndim, int* shape, Meta* metadata);
Tensor* fill_value_host_tensor(double value, Tensor* tensor);

Tensor* zeros_device_tensor(DType dtype, int device_id, int ndim, int* shape, Meta* metadata);
Tensor* ones_device_tensor(DType dtype, int device_id, int ndim, int* shape, Meta* metadata);
Tensor* values_device_tensor(void* vals, DType vals_dtype, DType target_dtype, DeviceType source_device, int ndim, int* shape, int device_id, Meta* metadata);
Tensor* fill_value_device_tensor(double value, Tensor* tensor);

Tensor* zeros_tensor(DType dtype, DeviceType device, int device_id, int ndim, int* shape, Meta* metadata);
Tensor* ones_tensor(DType dtype, DeviceType device, int device_id, int ndim, int* shape, Meta* metadata);
Tensor* values_tensor(void* vals, DType vals_dtype, DType target_dtype, DeviceType source_device, DeviceType target_device, int device_id, int ndim, int* shape, Meta* metadata);
Tensor* fill_value_tensor(double value, Tensor* tensor);

Tensor* move_host_to_device(Tensor* tensor, int device_id, DType target_dtype);
Tensor* move_device_to_host(Tensor* tensor, DType target_dtype);
Tensor* tensor_copy_host(Tensor* tensor, DType target_dtype);
Tensor* tensor_copy_device(Tensor* tensor, int device_id, DType target_dtype);

Tensor* tensor_copy(Tensor* tensor);
Tensor* tensor_to(Tensor* src, DeviceType target_device, int target_device_id, DType target_dtype, bool inplace);

bool validate_device_id(DeviceType device, int device_id);

void free_tensor(Tensor* tensor);

void free_tensor_host(Tensor* tensor);
void free_tensor_device(Tensor* tensor);

#ifdef __cplusplus
}
#endif

#endif
