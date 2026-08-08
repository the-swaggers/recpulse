#include "serialize.h"
#include "half_precision.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <cuda_runtime.h>

TensorDict* tensor_dict_create(int capacity) {
    TensorDict* dict = (TensorDict*)calloc(1, sizeof(TensorDict));
    if (!dict) return NULL;
    dict->names = (char**)calloc(capacity, sizeof(char*));
    dict->tensors = (Tensor**)calloc(capacity, sizeof(Tensor*));
    if (!dict->names || !dict->tensors) {
        free(dict->names);
        free(dict->tensors);
        free(dict);
        return NULL;
    }
    dict->count = 0;
    return dict;
}

void tensor_dict_free(TensorDict* dict) {
    if (!dict) return;
    for (int i = 0; i < dict->count; i++) {
        free(dict->names[i]);
    }
    free(dict->names);
    free(dict->tensors);
    free(dict);
}

int tensor_dict_add(TensorDict* dict, const char* name, Tensor* tensor) {
    if (!dict || !name || !tensor) return -1;
    dict->names[dict->count] = (char*)malloc(strlen(name) + 1);
    strcpy(dict->names[dict->count], name);
    dict->tensors[dict->count] = tensor;
    dict->count++;
    return 0;
}

int rpt_save(TensorDict* dict, const char* path) {
    if (!dict || !path) return -1;
    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    unsigned int magic = RPT_MAGIC;
    int count = dict->count;
    fwrite(&magic, sizeof(unsigned int), 1, f);
    fwrite(&count, sizeof(int), 1, f);

    for (int i = 0; i < count; i++) {
        Tensor* t = dict->tensors[i];
        const char* name = dict->names[i];

        Tensor* host_t = t;
        int need_free = 0;
        if (t->device_id >= 0) {
            host_t = tensor_to(t, -1, t->dtype, 0);
            if (!host_t) { fclose(f); return -1; }
            need_free = 1;
        }

        int name_len = (int)strlen(name);
        fwrite(&name_len, sizeof(int), 1, f);
        fwrite(name, 1, name_len, f);

        int dtype = (int)host_t->dtype;
        fwrite(&dtype, sizeof(int), 1, f);
        fwrite(&host_t->ndim, sizeof(int), 1, f);
        fwrite(host_t->shape, sizeof(int), host_t->ndim, f);

        size_t data_bytes = host_t->size * dtype_size(host_t->dtype);
        fwrite(&data_bytes, sizeof(size_t), 1, f);
        fwrite(host_t->data, 1, data_bytes, f);

        if (need_free) free_tensor(host_t);
    }

    fclose(f);
    return 0;
}

TensorDict* rpt_load(const char* path) {
    if (!path) return NULL;
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    unsigned int magic;
    int count;
    if (fread(&magic, sizeof(unsigned int), 1, f) != 1 || magic != RPT_MAGIC) {
        fclose(f);
        return NULL;
    }
    if (fread(&count, sizeof(int), 1, f) != 1) { fclose(f); return NULL; }

    TensorDict* dict = tensor_dict_create(count);
    if (!dict) { fclose(f); return NULL; }

    for (int i = 0; i < count; i++) {
        int name_len;
        if (fread(&name_len, sizeof(int), 1, f) != 1) { tensor_dict_free(dict); fclose(f); return NULL; }
        char* name = (char*)malloc(name_len + 1);
        if ((int)fread(name, 1, name_len, f) != name_len) { free(name); tensor_dict_free(dict); fclose(f); return NULL; }
        name[name_len] = '\0';

        int dtype_int, ndim;
        if (fread(&dtype_int, sizeof(int), 1, f) != 1) { free(name); tensor_dict_free(dict); fclose(f); return NULL; }
        if (fread(&ndim, sizeof(int), 1, f) != 1) { free(name); tensor_dict_free(dict); fclose(f); return NULL; }

        int* shape = (int*)malloc(ndim * sizeof(int));
        if ((int)fread(shape, sizeof(int), ndim, f) != ndim) { free(shape); free(name); tensor_dict_free(dict); fclose(f); return NULL; }

        size_t data_bytes;
        if (fread(&data_bytes, sizeof(size_t), 1, f) != 1) { free(shape); free(name); tensor_dict_free(dict); fclose(f); return NULL; }

        DType dtype = (DType)dtype_int;
        Tensor* tensor = zeros_tensor(dtype, -1, ndim, shape, NULL);
        free(shape);
        if (!tensor) { free(name); tensor_dict_free(dict); fclose(f); return NULL; }

        if (fread(tensor->data, 1, data_bytes, f) != data_bytes) {
            free_tensor(tensor);
            free(name);
            tensor_dict_free(dict);
            fclose(f);
            return NULL;
        }

        dict->names[dict->count] = name;
        dict->tensors[dict->count] = tensor;
        dict->count++;
    }

    fclose(f);
    return dict;
}
