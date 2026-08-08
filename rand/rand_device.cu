#include "rand.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define PHILOX_M0 0xD2511F53U
#define PHILOX_M1 0xCD9E8D57U
#define PHILOX_W0 0x9E3779B9U
#define PHILOX_W1 0xBB67AE85U

struct Philox4x32 {
    unsigned int v[4];
};

__device__ static inline Philox4x32 philox4x32_10(unsigned long long counter, unsigned long long key, unsigned int thread_offset) {
    unsigned int c0 = (unsigned int)(counter + thread_offset);
    unsigned int c1 = (unsigned int)((counter + thread_offset) >> 32);
    unsigned int c2 = (unsigned int)thread_offset;
    unsigned int c3 = 0;
    unsigned int k0 = (unsigned int)key;
    unsigned int k1 = (unsigned int)(key >> 32);

    for (int round = 0; round < 10; round++) {
        unsigned int hi0 = __umulhi(PHILOX_M0, c0);
        unsigned int lo0 = PHILOX_M0 * c0;
        unsigned int hi1 = __umulhi(PHILOX_M1, c2);
        unsigned int lo1 = PHILOX_M1 * c2;

        c0 = hi1 ^ c1 ^ k0;
        c1 = lo1;
        c2 = hi0 ^ c3 ^ k1;
        c3 = lo0;

        k0 += PHILOX_W0;
        k1 += PHILOX_W1;
    }

    Philox4x32 result;
    result.v[0] = c0;
    result.v[1] = c1;
    result.v[2] = c2;
    result.v[3] = c3;
    return result;
}

__device__ static inline float uint32_to_float(unsigned int x) {
    return (float)(x >> 8) * (1.0f / 16777216.0f);
}

template<typename T>
__global__ void rand_uniform_kernel(T* out, size_t size, unsigned long long counter, unsigned long long key) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t base = idx * 4;
    if (base >= size) return;

    Philox4x32 rng = philox4x32_10(counter, key, (unsigned int)idx);

    for (int i = 0; i < 4 && base + i < size; i++) {
        out[base + i] = T(uint32_to_float(rng.v[i]));
    }
}

template<typename T>
__global__ void randn_box_muller_kernel(T* out, size_t size, unsigned long long counter, unsigned long long key) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t base = idx * 2;
    if (base >= size) return;

    Philox4x32 rng = philox4x32_10(counter, key, (unsigned int)idx);

    float u1 = uint32_to_float(rng.v[0]);
    float u2 = uint32_to_float(rng.v[1]);
    if (u1 <= 0.0f) u1 = 1e-7f;

    float mag = sqrtf(-2.0f * logf(u1));
    float z0 = mag * cosf(2.0f * (float)M_PI * u2);
    float z1 = mag * sinf(2.0f * (float)M_PI * u2);

    out[base] = T(z0);
    if (base + 1 < size) out[base + 1] = T(z1);
}

__device__ static inline float inverse_cdf_device(float u) {
    const float a[] = {
        -3.969683028665376e+01f,  2.209460984245205e+02f,
        -2.759285104469687e+02f,  1.383577518672690e+02f,
        -3.066479806614716e+01f,  2.506628277459239e+00f
    };
    const float b[] = {
        -5.447609879822406e+01f,  1.615858368580409e+02f,
        -1.556989798598866e+02f,  6.680131188771972e+01f,
        -1.328068155288572e+01f
    };
    const float c[] = {
        -7.784894002430293e-03f, -3.223964580411365e-01f,
        -2.400758277161838e+00f, -2.549732539343734e+00f,
         4.374664141464968e+00f,  2.938163982698783e+00f
    };
    const float d[] = {
         7.784695709041462e-03f,  3.224671290700398e-01f,
         2.445134137142996e+00f,  3.754408661907416e+00f
    };

    const float p_low = 0.02425f;
    const float p_high = 1.0f - 0.02425f;

    float q, r;

    if (u < p_low) {
        q = sqrtf(-2.0f * logf(u));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0f);
    } else if (u <= p_high) {
        q = u - 0.5f;
        r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0f);
    } else {
        q = sqrtf(-2.0f * logf(1.0f - u));
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0f);
    }
}

template<typename T>
__global__ void randn_inverse_cdf_kernel(T* out, size_t size, unsigned long long counter, unsigned long long key) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t base = idx * 4;
    if (base >= size) return;

    Philox4x32 rng = philox4x32_10(counter, key, (unsigned int)idx);

    for (int i = 0; i < 4 && base + i < size; i++) {
        float u = uint32_to_float(rng.v[i]);
        if (u <= 0.0f) u = 1e-7f;
        if (u >= 1.0f) u = 1.0f - 1e-7f;
        out[base + i] = T(inverse_cdf_device(u));
    }
}

__constant__ float d_zig_ytab[128];
__constant__ float d_zig_wtab[128];
__constant__ unsigned int d_zig_ktab[128];
static bool zig_tables_uploaded = false;

static void upload_ziggurat_tables() {
    if (zig_tables_uploaded) return;

    const double R = 3.44428647676;
    const double V = 0.00991256303526217;

    float ytab[128], wtab[128];
    unsigned int ktab[128];

    double dn = R;
    double tn = dn;
    double q = V / exp(-0.5 * dn * dn);

    ktab[0] = (unsigned int)((dn / q) * (double)(1U << 31));
    ktab[1] = 0;
    wtab[127] = (float)(q / (double)(1U << 31));
    wtab[0] = (float)(dn / (double)(1U << 31));
    ytab[0] = 1.0f;
    ytab[127] = (float)exp(-0.5 * dn * dn);

    for (int i = 126; i >= 1; i--) {
        dn = sqrt(-2.0 * log(V / dn + exp(-0.5 * dn * dn)));
        ktab[i + 1] = (unsigned int)((dn / tn) * (double)(1U << 31));
        tn = dn;
        ytab[i] = (float)exp(-0.5 * dn * dn);
        wtab[i] = (float)(dn / (double)(1U << 31));
    }

    cudaMemcpyToSymbol(d_zig_ytab, ytab, sizeof(ytab));
    cudaMemcpyToSymbol(d_zig_wtab, wtab, sizeof(wtab));
    cudaMemcpyToSymbol(d_zig_ktab, ktab, sizeof(ktab));
    zig_tables_uploaded = true;
}

template<typename T>
__global__ void randn_ziggurat_kernel(T* out, size_t size, unsigned long long counter, unsigned long long key) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    const float R = 3.44428647676f;
    unsigned int extra_counter = 0;

    while (1) {
        Philox4x32 rng = philox4x32_10(counter + extra_counter * (size + 255) / 256, key, (unsigned int)idx);
        unsigned int r = rng.v[0];
        int j = r & 0x7F;
        float x = (float)((int)r) * d_zig_wtab[j];

        if ((unsigned int)fabsf((float)((int)r)) < d_zig_ktab[j]) {
            out[idx] = T(x);
            return;
        }

        if (j == 0) {
            float u1 = uint32_to_float(rng.v[1]);
            float u2 = uint32_to_float(rng.v[2]);
            if (u1 <= 0.0f) u1 = 1e-7f;
            float tail = -logf(u1) / R;
            if (-2.0f * logf(u2) >= tail * tail) {
                out[idx] = T(((int)r > 0) ? tail + R : -(tail + R));
                return;
            }
        } else {
            float u = uint32_to_float(rng.v[1]);
            if (d_zig_ytab[j - 1] + u * (d_zig_ytab[j] - d_zig_ytab[j - 1]) < expf(-0.5f * x * x)) {
                out[idx] = T(x);
                return;
            }
        }

        extra_counter++;
    }
}

template<typename T>
__global__ void randint_kernel(T* out, size_t size, unsigned long long counter, unsigned long long key, float low, float range) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t base = idx * 4;
    if (base >= size) return;

    Philox4x32 rng = philox4x32_10(counter, key, (unsigned int)idx);

    for (int i = 0; i < 4 && base + i < size; i++) {
        float u = uint32_to_float(rng.v[i]);
        out[base + i] = T(floorf(low + u * range));
    }
}

static bool check_cuda_kernel() {
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess;
}

extern "C" {

int rand_fill_device(void* out, size_t size, DType dtype, uint64_t counter, uint64_t key) {
    int threads = 256;
    int elements_per_thread = 4;
    int blocks = ((int)size + threads * elements_per_thread - 1) / (threads * elements_per_thread);

    if (dtype == DTYPE_FLOAT32) {
        rand_uniform_kernel<float><<<blocks, threads>>>((float*)out, size, counter, key);
    } else if (dtype == DTYPE_FLOAT64) {
        rand_uniform_kernel<double><<<blocks, threads>>>((double*)out, size, counter, key);
    } else if (dtype == DTYPE_FLOAT16) {
        rand_uniform_kernel<__half><<<blocks, threads>>>((__half*)out, size, counter, key);
    } else if (dtype == DTYPE_BFLOAT16) {
        rand_uniform_kernel<__nv_bfloat16><<<blocks, threads>>>((__nv_bfloat16*)out, size, counter, key);
    } else {
        return -1;
    }
    return check_cuda_kernel() ? 0 : -1;
}

int randn_fill_device(void* out, size_t size, DType dtype, NormalMethod method, uint64_t counter, uint64_t key) {
    int threads = 256;

    if (method == NORMAL_BOX_MULLER) {
        int elements_per_thread = 2;
        int blocks = ((int)size + threads * elements_per_thread - 1) / (threads * elements_per_thread);
        if (dtype == DTYPE_FLOAT32) {
            randn_box_muller_kernel<float><<<blocks, threads>>>((float*)out, size, counter, key);
        } else if (dtype == DTYPE_FLOAT64) {
            randn_box_muller_kernel<double><<<blocks, threads>>>((double*)out, size, counter, key);
        } else if (dtype == DTYPE_FLOAT16) {
            randn_box_muller_kernel<__half><<<blocks, threads>>>((__half*)out, size, counter, key);
        } else if (dtype == DTYPE_BFLOAT16) {
            randn_box_muller_kernel<__nv_bfloat16><<<blocks, threads>>>((__nv_bfloat16*)out, size, counter, key);
        } else {
            return -1;
        }
    } else if (method == NORMAL_ZIGGURAT) {
        upload_ziggurat_tables();
        int blocks = ((int)size + threads - 1) / threads;
        if (dtype == DTYPE_FLOAT32) {
            randn_ziggurat_kernel<float><<<blocks, threads>>>((float*)out, size, counter, key);
        } else if (dtype == DTYPE_FLOAT64) {
            randn_ziggurat_kernel<double><<<blocks, threads>>>((double*)out, size, counter, key);
        } else if (dtype == DTYPE_FLOAT16) {
            randn_ziggurat_kernel<__half><<<blocks, threads>>>((__half*)out, size, counter, key);
        } else if (dtype == DTYPE_BFLOAT16) {
            randn_ziggurat_kernel<__nv_bfloat16><<<blocks, threads>>>((__nv_bfloat16*)out, size, counter, key);
        } else {
            return -1;
        }
    } else if (method == NORMAL_INVERSE_CDF) {
        int elements_per_thread = 4;
        int blocks = ((int)size + threads * elements_per_thread - 1) / (threads * elements_per_thread);
        if (dtype == DTYPE_FLOAT32) {
            randn_inverse_cdf_kernel<float><<<blocks, threads>>>((float*)out, size, counter, key);
        } else if (dtype == DTYPE_FLOAT64) {
            randn_inverse_cdf_kernel<double><<<blocks, threads>>>((double*)out, size, counter, key);
        } else if (dtype == DTYPE_FLOAT16) {
            randn_inverse_cdf_kernel<__half><<<blocks, threads>>>((__half*)out, size, counter, key);
        } else if (dtype == DTYPE_BFLOAT16) {
            randn_inverse_cdf_kernel<__nv_bfloat16><<<blocks, threads>>>((__nv_bfloat16*)out, size, counter, key);
        } else {
            return -1;
        }
    } else {
        return -1;
    }

    return check_cuda_kernel() ? 0 : -1;
}

int randint_fill_device(void* out, size_t size, DType dtype, double low, double high, uint64_t counter, uint64_t key) {
    int threads = 256;
    int elements_per_thread = 4;
    int blocks = ((int)size + threads * elements_per_thread - 1) / (threads * elements_per_thread);
    float f_low = (float)low;
    float f_range = (float)(high - low);

    if (dtype == DTYPE_FLOAT32) {
        randint_kernel<float><<<blocks, threads>>>((float*)out, size, counter, key, f_low, f_range);
    } else if (dtype == DTYPE_FLOAT64) {
        randint_kernel<double><<<blocks, threads>>>((double*)out, size, counter, key, f_low, f_range);
    } else if (dtype == DTYPE_FLOAT16) {
        randint_kernel<__half><<<blocks, threads>>>((__half*)out, size, counter, key, f_low, f_range);
    } else if (dtype == DTYPE_BFLOAT16) {
        randint_kernel<__nv_bfloat16><<<blocks, threads>>>((__nv_bfloat16*)out, size, counter, key, f_low, f_range);
    } else {
        return -1;
    }
    return check_cuda_kernel() ? 0 : -1;
}

}
