#ifndef CUDA_CONFIG_CUH
#define CUDA_CONFIG_CUH

#include <cuda_runtime.h>

extern "C" {
    int get_optimal_block_size();
    void set_block_size(int size);
    int get_current_block_size();
}

#endif

