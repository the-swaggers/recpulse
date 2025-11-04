#ifndef TENSOR_H
#define TENSOR_H

typedef enum {
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
} DType;


typedef enum {
    DEVICE_CPU,
    DEVICE_CUDA,
} DeviceType;

typedef struct {
    DType dtype;
    void* data;
    int ndim;
    int size;
    int* shape;
    int* strides;
    DeviceType device;
    int device_id;
    bool owns_data;

}

#endif
