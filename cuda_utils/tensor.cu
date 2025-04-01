#include "tensor.cuh"
#include <cuda_runtime.h>
#include <stdio.h>


__global__ void fill_kernel_scalar(void* vals, float value, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        vals[idx] = value;
    }
}

void fill_kernel_vals(Tensor* tensor, void* new_vals, size_t size) {
    cudaMemcpy(tensor->vals, new_vals, tensor->size * sizeof(void), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize(); 
}


Tensor* create_tensor(int* shape, int ndim) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    tensor->ndim = ndim;
    tensor->shape = (int*)malloc(ndim * sizeof(int));
    
    size_t size = 1;
    for (int i = 0; i < ndim; i++) {
        tensor->shape[i] = shape[i];
        size *= shape[i];
    }
    tensor->size = size;
    
    cudaMalloc(&tensor->vals, size * sizeof(void));
    return tensor;
}

void free_tensor(Tensor* tensor) {
    cudaFree(tensor->vals);
    free(tensor->shape);
    free(tensor);
}

void fill_tensor_scalar(Tensor* tensor, void value) {
    int block_size = 256;
    int num_blocks = (tensor->size + block_size - 1) / block_size;
    
    fill_kernel_scalar<<<num_blocks, block_size>>>(tensor->vals, value, tensor->size);
    cudaDeviceSynchronize();
}

void fill_tensor_vals(Tensor* tensor, void* new_vals) {
    char str[50];
    int size = tensor->size;

    int block_size = 256;
    int num_blocks = (tensor->size + block_size - 1) / block_size;

    fill_kernel_vals(tensor, new_vals, tensor->size);

    void arr[tensor->size];
}

void vals_from_tensor(Tensor* tensor, void* array) {
    cudaMemcpy(array, tensor->vals, tensor->size * sizeof(void), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize(); 
}

void element_from_tensor(Tensor* tensor, int* idx) {
    int index = 0;
    int n = 1;
    for (int i = tensor->ndim - 1; i >= 0; i--) {
        index += n * idx[i];
        n *= tensor->shape[i];
    }
    void result;
    
    cudaError_t err = cudaMemcpy(&result, &(tensor->vals[index]), sizeof(void), cudaMemcpyDeviceToHost);
    
    return result;
}

void free_array(void* arr) {
    free(arr);
}

//  Tensor cut_tensor(Tensor* tensor, int* shape, int* slice) {
//      
//  }

