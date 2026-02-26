#include "tensor.h"
#include "half_precision.h"
#include "cuda_helpers.h"
#include <stdlib.h>
#include <string.h>

Tensor* zeros_host_tensor(DType dtype, int ndim, int* shape, Meta* metadata) {
    if (!check_shape_valid(ndim, shape)) return NULL;

    size_t total_elements;
    if (!calculate_total_elements(ndim, shape, &total_elements)) return NULL;

    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) return NULL;

    tensor->shape = NULL;
    tensor->strides = NULL;
    tensor->data = NULL;
    tensor->metadata = metadata;

    tensor->shape = (int*)malloc(ndim * sizeof(int));
    if (!tensor->shape) goto cleanup;
    memcpy(tensor->shape, shape, ndim * sizeof(int));

    tensor->strides = (int*)malloc(ndim * sizeof(int));
    if (!tensor->strides) goto cleanup;

    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        tensor->strides[i] = stride;
        stride *= shape[i];
    }

    tensor->data = calloc(total_elements, dtype_size(dtype));
    if (!tensor->data) goto cleanup;

    tensor->dtype = dtype;
    tensor->ndim = ndim;
    tensor->size = total_elements;
    tensor->device_id = -1;
    tensor->owns_data = true;
    tensor->base_tensor = NULL;

    return tensor;

cleanup:
    if (tensor->data) free(tensor->data);
    if (tensor->strides) free(tensor->strides);
    if (tensor->shape) free(tensor->shape);
    free(tensor);
    return NULL;
}

Tensor* ones_host_tensor(DType dtype, int ndim, int* shape, Meta* metadata) {
    if (!check_shape_valid(ndim, shape)) return NULL;

    size_t total_elements;
    if (!calculate_total_elements(ndim, shape, &total_elements)) return NULL;

    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) return NULL;

    tensor->shape = NULL;
    tensor->strides = NULL;
    tensor->data = NULL;
    tensor->metadata = metadata;

    tensor->shape = (int*)malloc(ndim * sizeof(int));
    if (!tensor->shape) goto cleanup;
    memcpy(tensor->shape, shape, ndim * sizeof(int));

    tensor->strides = (int*)malloc(ndim * sizeof(int));
    if (!tensor->strides) goto cleanup;

    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        tensor->strides[i] = stride;
        stride *= shape[i];
    }

    tensor->data = malloc(total_elements * dtype_size(dtype));
    if (!tensor->data) goto cleanup;

    if (dtype == DTYPE_FLOAT32) {
        float* data = (float*)tensor->data;
        for (size_t i = 0; i < total_elements; i++) {
            data[i] = 1.0f;
        }
    } else if (dtype == DTYPE_FLOAT64) {
        double* data = (double*)tensor->data;
        for (size_t i = 0; i < total_elements; i++) {
            data[i] = 1.0;
        }
    } else if (dtype == DTYPE_FLOAT16) {
        uint16_t* data = (uint16_t*)tensor->data;
        uint16_t one = fp32_to_fp16(1.0f);
        for (size_t i = 0; i < total_elements; i++) {
            data[i] = one;
        }
    } else if (dtype == DTYPE_BFLOAT16) {
        uint16_t* data = (uint16_t*)tensor->data;
        uint16_t one = fp32_to_bf16(1.0f);
        for (size_t i = 0; i < total_elements; i++) {
            data[i] = one;
        }
    }

    tensor->dtype = dtype;
    tensor->ndim = ndim;
    tensor->size = total_elements;
    tensor->device_id = -1;
    tensor->owns_data = true;
    tensor->base_tensor = NULL;

    return tensor;

cleanup:
    if (tensor->data) free(tensor->data);
    if (tensor->strides) free(tensor->strides);
    if (tensor->shape) free(tensor->shape);
    free(tensor);
    return NULL;
}

Tensor* values_host_tensor(void* vals, DType vals_dtype, DType target_dtype, int ndim, int* shape, Meta* metadata) {
    if (!check_shape_valid(ndim, shape)) return NULL;

    size_t total_elements;
    if (!calculate_total_elements(ndim, shape, &total_elements)) return NULL;

    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) return NULL;

    tensor->shape = NULL;
    tensor->strides = NULL;
    tensor->data = NULL;
    tensor->metadata = metadata;

    tensor->shape = (int*)malloc(ndim * sizeof(int));
    if (!tensor->shape) goto cleanup;
    memcpy(tensor->shape, shape, ndim * sizeof(int));

    tensor->strides = (int*)malloc(ndim * sizeof(int));
    if (!tensor->strides) goto cleanup;

    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        tensor->strides[i] = stride;
        stride *= shape[i];
    }

    size_t data_size = total_elements * dtype_size(target_dtype);

    tensor->data = malloc(data_size);
    if (!tensor->data) goto cleanup;

    if (vals_dtype == target_dtype) {
        memcpy(tensor->data, vals, data_size);
    } else {
        float* tmp = NULL;
        if (vals_dtype == DTYPE_FLOAT32) {
            tmp = (float*)vals;
        } else if (vals_dtype == DTYPE_FLOAT64) {
            tmp = (float*)malloc(total_elements * sizeof(float));
            if (!tmp) goto cleanup;
            double* src = (double*)vals;
            for (size_t i = 0; i < total_elements; i++) tmp[i] = (float)src[i];
        } else {
            tmp = (float*)malloc(total_elements * sizeof(float));
            if (!tmp) goto cleanup;
            half_to_fp32_array(vals, tmp, total_elements, vals_dtype);
        }

        if (target_dtype == DTYPE_FLOAT32) {
            memcpy(tensor->data, tmp, total_elements * sizeof(float));
        } else if (target_dtype == DTYPE_FLOAT64) {
            double* dst = (double*)tensor->data;
            for (size_t i = 0; i < total_elements; i++) dst[i] = (double)tmp[i];
        } else {
            fp32_to_half_array(tmp, tensor->data, total_elements, target_dtype);
        }

        if (tmp != (float*)vals) free(tmp);
    }

    tensor->dtype = target_dtype;
    tensor->ndim = ndim;
    tensor->size = total_elements;
    tensor->device_id = -1;
    tensor->owns_data = true;

    return tensor;

cleanup:
    if (tensor->data) free(tensor->data);
    if (tensor->strides) free(tensor->strides);
    if (tensor->shape) free(tensor->shape);
    free(tensor);
    return NULL;
}

Tensor* tensor_copy_host(Tensor* tensor, DType target_dtype) {
    if (tensor->device_id != -1) return NULL;

    Tensor* copy = (Tensor*)malloc(sizeof(Tensor));
    if (!copy) return NULL;

    copy->shape = NULL;
    copy->strides = NULL;
    copy->data = NULL;
    copy->metadata = NULL;

    copy->shape = (int*)malloc(tensor->ndim * sizeof(int));
    if (!copy->shape) goto cleanup;
    memcpy(copy->shape, tensor->shape, tensor->ndim * sizeof(int));

    copy->strides = (int*)malloc(tensor->ndim * sizeof(int));
    if (!copy->strides) goto cleanup;
    memcpy(copy->strides, tensor->strides, tensor->ndim * sizeof(int));

    size_t data_size = tensor->size * dtype_size(target_dtype);

    copy->data = malloc(data_size);
    if (!copy->data) goto cleanup;

    if (tensor->dtype == target_dtype) {
        memcpy(copy->data, tensor->data, data_size);
    } else {
        float* tmp = NULL;
        if (tensor->dtype == DTYPE_FLOAT32) {
            tmp = (float*)tensor->data;
        } else if (tensor->dtype == DTYPE_FLOAT64) {
            tmp = (float*)malloc(tensor->size * sizeof(float));
            if (!tmp) goto cleanup;
            double* src = (double*)tensor->data;
            for (size_t i = 0; i < tensor->size; i++) tmp[i] = (float)src[i];
        } else {
            tmp = (float*)malloc(tensor->size * sizeof(float));
            if (!tmp) goto cleanup;
            half_to_fp32_array(tensor->data, tmp, tensor->size, tensor->dtype);
        }

        if (target_dtype == DTYPE_FLOAT32) {
            memcpy(copy->data, tmp, tensor->size * sizeof(float));
        } else if (target_dtype == DTYPE_FLOAT64) {
            double* dst = (double*)copy->data;
            for (size_t i = 0; i < tensor->size; i++) dst[i] = (double)tmp[i];
        } else {
            fp32_to_half_array(tmp, copy->data, tensor->size, target_dtype);
        }

        if (tmp != (float*)tensor->data) free(tmp);
    }

    copy->dtype = target_dtype;
    copy->ndim = tensor->ndim;
    copy->size = tensor->size;
    copy->device_id = -1;
    copy->owns_data = true;

    if (tensor->metadata) {
        copy->metadata = (Meta*)malloc(sizeof(Meta));
        if (copy->metadata) {
            memcpy(copy->metadata, tensor->metadata, sizeof(Meta));
            copy->metadata->grad = NULL;
            copy->metadata->grad_fn = NULL;
        }
    }

    return copy;

cleanup:
    if (copy->data) free(copy->data);
    if (copy->strides) free(copy->strides);
    if (copy->shape) free(copy->shape);
    free(copy);
    return NULL;
}

void free_tensor_host(Tensor* tensor) {
    if (!tensor) return;
    if (tensor->device_id != -1) return;

    if (tensor->owns_data && tensor->data) {
        free(tensor->data);
    }

    if (tensor->shape) {
        free(tensor->shape);
    }

    if (tensor->strides) {
        free(tensor->strides);
    }

    if (tensor->metadata) {
        if (tensor->metadata->grad) {
            free_tensor(tensor->metadata->grad);
        }
        free(tensor->metadata);
    }

    free(tensor);
}


Tensor* fill_value_host_tensor(double value, Tensor* tensor){
    if (!tensor || !tensor->data) return NULL;

    if (tensor->dtype == DTYPE_FLOAT32) {
        float* data = (float*)tensor->data;
        float f_value = (float)value;
        for (size_t i = 0; i < tensor->size; i++) {
            data[i] = f_value;
        }
    } else if (tensor->dtype == DTYPE_FLOAT64) {
        double* data = (double*)tensor->data;
        for (size_t i = 0; i < tensor->size; i++) {
            data[i] = value;
        }
    } else if (tensor->dtype == DTYPE_FLOAT16) {
        uint16_t* data = (uint16_t*)tensor->data;
        uint16_t h_value = fp32_to_fp16((float)value);
        for (size_t i = 0; i < tensor->size; i++) {
            data[i] = h_value;
        }
    } else if (tensor->dtype == DTYPE_BFLOAT16) {
        uint16_t* data = (uint16_t*)tensor->data;
        uint16_t b_value = fp32_to_bf16((float)value);
        for (size_t i = 0; i < tensor->size; i++) {
            data[i] = b_value;
        }
    }
    return tensor;
}

Tensor* tensor_reshape_host(Tensor* tensor, int new_ndim, int* new_shape) {
    if (tensor->device_id != -1) return NULL;

    Tensor* reshaped = (Tensor*)malloc(sizeof(Tensor));
    if (!reshaped) return NULL;

    reshaped->shape = NULL;
    reshaped->strides = NULL;
    reshaped->data = NULL;
    reshaped->metadata = NULL;

    reshaped->shape = (int*)malloc(new_ndim * sizeof(int));
    if (!reshaped->shape) goto cleanup;
    memcpy(reshaped->shape, new_shape, new_ndim * sizeof(int));

    reshaped->strides = (int*)malloc(new_ndim * sizeof(int));
    if (!reshaped->strides) goto cleanup;

    int stride = 1;
    for (int i = new_ndim - 1; i >= 0; i--) {
        reshaped->strides[i] = stride;
        stride *= new_shape[i];
    }

    size_t data_size = tensor->size * dtype_size(tensor->dtype);

    reshaped->data = malloc(data_size);
    if (!reshaped->data) goto cleanup;

    memcpy(reshaped->data, tensor->data, data_size);

    reshaped->dtype = tensor->dtype;
    reshaped->ndim = new_ndim;
    reshaped->size = tensor->size;
    reshaped->device_id = -1;
    reshaped->owns_data = true;
    reshaped->base_tensor = NULL;

    if (tensor->metadata) {
        reshaped->metadata = (Meta*)malloc(sizeof(Meta));
        if (reshaped->metadata) {
            memcpy(reshaped->metadata, tensor->metadata, sizeof(Meta));
            reshaped->metadata->grad = NULL;
            reshaped->metadata->grad_fn = NULL;
        }
    }

    return reshaped;

cleanup:
    if (reshaped->data) free(reshaped->data);
    if (reshaped->strides) free(reshaped->strides);
    if (reshaped->shape) free(reshaped->shape);
    free(reshaped);
    return NULL;
}
