#ifndef TENSOR_CUH
#define TENSOR_CUH


typedef struct {
    void* vals;
    int* shape;
    int ndim;
    size_t size;
} Tensor;

extern "C" {
    Tensor* create_tensor(int* shape, int ndim);
    void free_tensor(Tensor* tensor);
    void fill_tensor_scalar(Tensor* tensor, void value);
    void fill_tensor_vals(Tensor* tensor, void* new_vals);
    float element_from_tensor(Tensor* tensor, int* idx);
    void vals_from_tensor(Tensor* tensor, void* array);
    void free_array(void* arr);
}

__global__ void fill_kernel(void* vals, void value, size_t size);

#endif
