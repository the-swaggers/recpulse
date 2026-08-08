#ifndef RECPULSE_SERIALIZE_H
#define RECPULSE_SERIALIZE_H

#include "tensor.h"

#define RPT_MAGIC 0x52505431

typedef struct {
    char** names;
    Tensor** tensors;
    int count;
} TensorDict;

TensorDict* tensor_dict_create(int capacity);
void tensor_dict_free(TensorDict* dict);
int tensor_dict_add(TensorDict* dict, const char* name, Tensor* tensor);

int rpt_save(TensorDict* dict, const char* path);
TensorDict* rpt_load(const char* path);

#endif
