#include "cuda_config.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

static int current_block_size = 0;


int get_optimal_block_size() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    int optimal_block_size;
    
    optimal_block_size = min(256, prop.maxThreadsPerBlock);
    optimal_block_size = (optimal_block_size / 32) * 32;
    
    return optimal_block_size;
}

void set_block_size(int size) {
    current_block_size = (size / 32) * 32;
}

int get_current_block_size() {
    if (current_block_size == 0) {
        const char* env_block_size = getenv("CUDA_BLOCK_SIZE");
        if (env_block_size != NULL) {
            int size = atoi(env_block_size);
            if (size > 0) {
                current_block_size = (size / 32) * 32;
            }
        }
        
        if (current_block_size == 0) {
            current_block_size = get_optimal_block_size();
        }
    }
    return current_block_size;
}

