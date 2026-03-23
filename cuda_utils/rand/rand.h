#ifndef RECPULSE_RAND_H
#define RECPULSE_RAND_H

#include "../core/tensor.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

typedef enum {
    NORMAL_BOX_MULLER = 0,
    NORMAL_ZIGGURAT = 1,
    NORMAL_INVERSE_CDF = 2,
} NormalMethod;

typedef struct {
    uint64_t s[4];
    bool has_cached_normal;
    double cached_normal;
} Xoshiro256State;

typedef struct {
    uint64_t counter;
    uint64_t key;
} PhiloxState;

void rp_manual_seed(uint64_t seed);

Tensor* rp_rand_tensor(DType dtype, int device_id, int ndim, int* shape);
Tensor* rp_randn_tensor(DType dtype, int device_id, int ndim, int* shape, NormalMethod method);
Tensor* rp_randint_tensor(double low, double high, DType dtype, int device_id, int ndim, int* shape);

int rp_rand_fill(void* out, size_t size, DType dtype, int device_id);
int rp_randn_fill(void* out, size_t size, DType dtype, int device_id, NormalMethod method);
int rp_randint_fill(void* out, size_t size, double low, double high, DType dtype, int device_id);

int rand_fill_host_f32(float* out, size_t size);
int rand_fill_host_f64(double* out, size_t size);
int randn_fill_host_f32(float* out, size_t size, NormalMethod method);
int randn_fill_host_f64(double* out, size_t size, NormalMethod method);
int randint_fill_host_f32(float* out, size_t size, float low, float high);
int randint_fill_host_f64(double* out, size_t size, double low, double high);

#ifdef __cplusplus
extern "C" {
#endif

int rand_fill_device(void* out, size_t size, DType dtype, uint64_t counter, uint64_t key);
int randn_fill_device(void* out, size_t size, DType dtype, NormalMethod method, uint64_t counter, uint64_t key);
int randint_fill_device(void* out, size_t size, DType dtype, double low, double high, uint64_t counter, uint64_t key);

#ifdef __cplusplus
}
#endif

Xoshiro256State* rp_get_cpu_state(void);

#endif
