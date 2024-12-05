#ifndef DTYPE
#define DTYPE

#include <cuda_fp16.h>

typedef union {
    __half        fp16;
    float         fp32;
    double        fp64;
    long double   fp128;
} Number;

typedef enum {
    float16,
    float32,
    float64,
    float128,
} dtype;

#endif

