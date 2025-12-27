#include "ops.h"
#include <string.h>

int backwards_add_a_host(const void* grad_c, void* grad_a, size_t size, DType dtype) {
    if (!grad_c || !grad_a) return -1;

    size_t element_size = (dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);
    size_t nbytes = size * element_size;

    memcpy(grad_a, grad_c, nbytes);
    return 0;
}

int backwards_add_b_host(const void* grad_c, void* grad_b, size_t size, DType dtype) {
    if (!grad_c || !grad_b) return -1;

    size_t element_size = (dtype == DTYPE_FLOAT32) ? sizeof(float) : sizeof(double);
    size_t nbytes = size * element_size;

    memcpy(grad_b, grad_c, nbytes);
    return 0;
}
