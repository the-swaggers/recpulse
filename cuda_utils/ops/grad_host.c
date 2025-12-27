#include "ops.h"
#include <string.h>

int backwards_add_x1_host(const void* grad_c, void* grad_x1, size_t size, DType dtype) {
    if (!grad_c || !grad_x1) return -1;

    size_t element_size = (dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);
    size_t nbytes = size * element_size;

    memcpy(grad_x1, grad_c, nbytes);
    return 0;
}

int backwards_add_x2_host(const void* grad_c, void* grad_x2, size_t size, DType dtype) {
    if (!grad_c || !grad_x2) return -1;

    size_t element_size = (dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);
    size_t nbytes = size * element_size;

    memcpy(grad_x2, grad_c, nbytes);
    return 0;
}
