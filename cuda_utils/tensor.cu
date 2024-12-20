#include "tensor.cuh"
#include <cuda_runtime.h>
#include <stdio.h>


void log_message(const char* message) {
    FILE* log_file = fopen("/home/drexon/projects/recpulse/cuda_utils/cuda_debug.log", "a");
    
    fprintf(log_file, "%s\n", message);
    fflush(log_file);
    fclose(log_file);
}

__global__ void fill_kernel_scalar(float* vals, float value, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        vals[idx] = value;
    }
}

void fill_kernel_vals(Tensor32* tensor, float* new_vals, size_t size) {
    cudaMemcpy(tensor->vals, new_vals, tensor->size * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize(); 
}


Tensor32* create_tensor(int* shape, int ndim) {
    Tensor32* tensor = (Tensor32*)malloc(sizeof(Tensor32));
    tensor->ndim = ndim;
    tensor->shape = (int*)malloc(ndim * sizeof(int));
    
    size_t size = 1;
    for (int i = 0; i < ndim; i++) {
        tensor->shape[i] = shape[i];
        size *= shape[i];
    }
    tensor->size = size;
    
    cudaMalloc(&tensor->vals, size * sizeof(float));
    return tensor;
}

void free_tensor(Tensor32* tensor) {
    cudaFree(tensor->vals);
    free(tensor->shape);
    free(tensor);
}

void fill_tensor_scalar(Tensor32* tensor, float value) {
    int block_size = 256;
    int num_blocks = (tensor->size + block_size - 1) / block_size;
    
    fill_kernel_scalar<<<num_blocks, block_size>>>(tensor->vals, value, tensor->size);
    cudaDeviceSynchronize();
}

void fill_tensor_vals(Tensor32* tensor, float* new_vals) {
    char str[50];
    int size = tensor->size;

    str[0] = '\0';
    char temp[10];
    for (int i = 0; i < size; i++) {
        sprintf(temp, "%.6f ", new_vals[i]);
        strcat(str, temp);
    }

    log_message(str);

    int block_size = 256;
    int num_blocks = (tensor->size + block_size - 1) / block_size;
    
    sprintf(temp, "%d ", num_blocks);
    log_message(temp);

    fill_kernel_vals(tensor, new_vals, tensor->size);

    float arr[tensor->size];
    
    sprintf(temp, "%.4f ", arr[1]);
    log_message(temp);
  
    arr[1] = 1.1111;

    vals_from_tensor(tensor, arr);
    
    log_message("haha");
    sprintf(temp, "%.4f ", arr[1]);
    log_message(temp);
}

void vals_from_tensor(Tensor32* tensor, float* array) {
    cudaMemcpy(array, tensor->vals, tensor->size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize(); 
}

float element_from_tensor(Tensor32* tensor, int* idx) {
    int index = 0;
    int n = 1;
    for (int i = tensor->ndim - 1; i >= 0; i--) {
        index += n * idx[i];
        n *= tensor->shape[i];
    }
    float result;
    
    cudaError_t err = cudaMemcpy(&result, &(tensor->vals[index]), sizeof(float), cudaMemcpyDeviceToHost);
    
    return result;
}

void free_array(float* arr) {
    free(arr);
}

//  Tensor32 cut_tensor(Tensor32* tensor, int* shape, int* slice) {
//      
//  }

