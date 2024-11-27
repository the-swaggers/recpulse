#ifndef TENSOR_CUH
#define TENSOR_CUH


// supports float32 only for simplicity
// TODO - add support for other types
typedef struct {
    float* vals;
    int* shape;
    int ndim;
    size_t size;
} Tensor32;

extern "C" {
    Tensor32* create_tensor(int* shape, int ndim);
    void free_tensor(Tensor32* tensor);
    void fill_tensor(Tensor32* tensor, float value);
}

__global__ void fill_kernel(float* vals, float value, size_t size);

Tensor32* create_tensor(int* shape, int ndim);
void free_tensor(Tensor32* tensor);
void fill_tensor(Tensor32* tensor, float value);

#endif
