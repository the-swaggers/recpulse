#ifndef HALF_PRECISION_H
#define HALF_PRECISION_H

#include "tensor.h"
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

static inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DTYPE_FLOAT32:  return 4;
        case DTYPE_FLOAT64:  return 8;
        case DTYPE_FLOAT16:  return 2;
        case DTYPE_BFLOAT16: return 2;
        default: return 0;
    }
}

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            uint32_t result = sign;
            float f;
            memcpy(&f, &result, 4);
            return f;
        }
        float f = (float)mant / 1024.0f;
        f *= (1.0f / 16384.0f);
        return (sign ? -f : f);
    }

    if (exp == 0x1F) {
        uint32_t result = sign | 0x7F800000 | ((uint32_t)mant << 13);
        float f;
        memcpy(&f, &result, 4);
        return f;
    }

    uint32_t result = sign | ((exp + 112) << 23) | ((uint32_t)mant << 13);
    float f;
    memcpy(&f, &result, 4);
    return f;
}

static inline uint16_t fp32_to_fp16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);

    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;

    if (((bits >> 23) & 0xFF) == 0xFF) {
        if (mant) {
            return (uint16_t)(sign | 0x7C00 | (mant >> 13) | 1);
        }
        return (uint16_t)(sign | 0x7C00);
    }

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000;
        int shift = 14 - exp;
        uint32_t round_bit = (mant >> (shift - 1)) & 1;
        uint16_t result = (uint16_t)(sign | (mant >> shift));
        result += (uint16_t)round_bit;
        return result;
    }

    if (exp >= 0x1F) {
        return (uint16_t)(sign | 0x7C00);
    }

    uint32_t round_bit = (mant >> 12) & 1;
    uint16_t result = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
    result += (uint16_t)round_bit;
    return result;
}

static inline float bf16_to_fp32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static inline uint16_t fp32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    uint32_t round_bit = (bits >> 15) & 1;
    return (uint16_t)((bits >> 16) + round_bit);
}

static inline void fp16_array_to_fp32(const uint16_t* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = fp16_to_fp32(src[i]);
}

static inline void fp32_array_to_fp16(const float* src, uint16_t* dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = fp32_to_fp16(src[i]);
}

static inline void bf16_array_to_fp32(const uint16_t* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = bf16_to_fp32(src[i]);
}

static inline void fp32_array_to_bf16(const float* src, uint16_t* dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = fp32_to_bf16(src[i]);
}

static inline void half_to_fp32_array(const void* src, float* dst, size_t n, DType dtype) {
    if (dtype == DTYPE_FLOAT16)
        fp16_array_to_fp32((const uint16_t*)src, dst, n);
    else
        bf16_array_to_fp32((const uint16_t*)src, dst, n);
}

static inline void fp32_to_half_array(const float* src, void* dst, size_t n, DType dtype) {
    if (dtype == DTYPE_FLOAT16)
        fp32_array_to_fp16(src, (uint16_t*)dst, n);
    else
        fp32_array_to_bf16(src, (uint16_t*)dst, n);
}

#endif
