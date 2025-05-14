#include <stdbool.h>
#include <stddef.h>

#ifndef TENSOR_H
#define TENSOR_H


struct Tensor{
    size_t size;
    ndim uint;
    shape *int;
    vals *void;
    <idk what here> dtype;
    device *char;
    requires_grad bool;
    owns_data bool;
}


#endif
