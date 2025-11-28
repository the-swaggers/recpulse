#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


void free_tensor(Tensor* tensor){
    if (!tensor) return;
    if (tensor->device == HOST) return free_tensor_host(tensor);
    if (tensor->device == DEVICE) return free_tensor_device(tensor);

    fprintf(stderr, "Error: Invalid device type %d in free_tensor\n", tensor->device);
    exit(1);
};
