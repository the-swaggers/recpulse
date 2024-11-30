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
    void fill_tensor_scalar(Tensor32* tensor, float value);
    void fill_tensor_vals(Tensor32* tensor, float* new_vals);
    float element_from_tensor(Tensor32* tensor, int* idx);
    void vals_from_tensor(Tensor32* tensor, float* array);
    void free_array(float* arr);
}

__global__ void fill_kernel(float* vals, float value, size_t size);

#endif
