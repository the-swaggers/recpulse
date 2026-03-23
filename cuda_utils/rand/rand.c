#define _POSIX_C_SOURCE 199309L
#include "rand.h"
#include "../core/half_precision.h"
#include <stdlib.h>
#include <time.h>

#define MAX_DEVICES 16

static Xoshiro256State g_cpu_state;
static PhiloxState g_philox_states[MAX_DEVICES];
static bool g_initialized = false;

static uint64_t splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

void rp_manual_seed(uint64_t seed) {
    uint64_t sm = seed;
    for (int i = 0; i < 4; i++) {
        g_cpu_state.s[i] = splitmix64(&sm);
    }
    g_cpu_state.has_cached_normal = false;
    g_cpu_state.cached_normal = 0.0;

    for (int d = 0; d < MAX_DEVICES; d++) {
        g_philox_states[d].counter = 0;
        uint64_t dk = seed ^ ((uint64_t)(d + 1) * 0x9e3779b97f4a7c15ULL);
        g_philox_states[d].key = dk;
    }

    g_initialized = true;
}

static void ensure_initialized(void) {
    if (!g_initialized) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        uint64_t seed = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
        rp_manual_seed(seed);
    }
}

Xoshiro256State* rp_get_cpu_state(void) {
    ensure_initialized();
    return &g_cpu_state;
}

int rp_rand_fill(void* out, size_t size, DType dtype, int device_id) {
    if (!out || size == 0) return -1;
    ensure_initialized();

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return rand_fill_host_f32((float*)out, size);
        } else if (dtype == DTYPE_FLOAT64) {
            return rand_fill_host_f64((double*)out, size);
        } else if (dtype == DTYPE_FLOAT16 || dtype == DTYPE_BFLOAT16) {
            float* tmp = (float*)malloc(size * sizeof(float));
            if (!tmp) return -1;
            int ret = rand_fill_host_f32(tmp, size);
            if (ret == 0) fp32_to_half_array(tmp, out, size, dtype);
            free(tmp);
            return ret;
        }
        return -1;
    }

    if (device_id < 0 || device_id >= MAX_DEVICES) return -1;
    PhiloxState* ps = &g_philox_states[device_id];
    uint64_t counter = ps->counter;
    int ret = rand_fill_device(out, size, dtype, counter, ps->key);
    ps->counter += (size + 3) / 4;
    return ret;
}

int rp_randn_fill(void* out, size_t size, DType dtype, int device_id, NormalMethod method) {
    if (!out || size == 0) return -1;
    ensure_initialized();

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return randn_fill_host_f32((float*)out, size, method);
        } else if (dtype == DTYPE_FLOAT64) {
            return randn_fill_host_f64((double*)out, size, method);
        } else if (dtype == DTYPE_FLOAT16 || dtype == DTYPE_BFLOAT16) {
            float* tmp = (float*)malloc(size * sizeof(float));
            if (!tmp) return -1;
            int ret = randn_fill_host_f32(tmp, size, method);
            if (ret == 0) fp32_to_half_array(tmp, out, size, dtype);
            free(tmp);
            return ret;
        }
        return -1;
    }

    if (device_id < 0 || device_id >= MAX_DEVICES) return -1;
    PhiloxState* ps = &g_philox_states[device_id];
    uint64_t counter = ps->counter;
    int ret = randn_fill_device(out, size, dtype, method, counter, ps->key);
    ps->counter += (size + 1) / 2;
    return ret;
}

int rp_randint_fill(void* out, size_t size, double low, double high, DType dtype, int device_id) {
    if (!out || size == 0) return -1;
    if (low >= high) return -1;
    ensure_initialized();

    if (device_id == -1) {
        if (dtype == DTYPE_FLOAT32) {
            return randint_fill_host_f32((float*)out, size, (float)low, (float)high);
        } else if (dtype == DTYPE_FLOAT64) {
            return randint_fill_host_f64((double*)out, size, low, high);
        } else if (dtype == DTYPE_FLOAT16 || dtype == DTYPE_BFLOAT16) {
            float* tmp = (float*)malloc(size * sizeof(float));
            if (!tmp) return -1;
            int ret = randint_fill_host_f32(tmp, size, (float)low, (float)high);
            if (ret == 0) fp32_to_half_array(tmp, out, size, dtype);
            free(tmp);
            return ret;
        }
        return -1;
    }

    if (device_id < 0 || device_id >= MAX_DEVICES) return -1;
    PhiloxState* ps = &g_philox_states[device_id];
    uint64_t counter = ps->counter;
    int ret = randint_fill_device(out, size, dtype, low, high, counter, ps->key);
    ps->counter += (size + 3) / 4;
    return ret;
}

Tensor* rp_rand_tensor(DType dtype, int device_id, int ndim, int* shape) {
    Tensor* out = zeros_tensor(dtype, device_id, ndim, shape, NULL);
    if (!out) return NULL;

    int ret = rp_rand_fill(out->data, out->size, dtype, device_id);
    if (ret != 0) {
        free_tensor(out);
        return NULL;
    }
    return out;
}

Tensor* rp_randn_tensor(DType dtype, int device_id, int ndim, int* shape, NormalMethod method) {
    Tensor* out = zeros_tensor(dtype, device_id, ndim, shape, NULL);
    if (!out) return NULL;

    int ret = rp_randn_fill(out->data, out->size, dtype, device_id, method);
    if (ret != 0) {
        free_tensor(out);
        return NULL;
    }
    return out;
}

Tensor* rp_randint_tensor(double low, double high, DType dtype, int device_id, int ndim, int* shape) {
    Tensor* out = zeros_tensor(dtype, device_id, ndim, shape, NULL);
    if (!out) return NULL;

    int ret = rp_randint_fill(out->data, out->size, low, high, dtype, device_id);
    if (ret != 0) {
        free_tensor(out);
        return NULL;
    }
    return out;
}
