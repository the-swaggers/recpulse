#include "rand.h"
#include <math.h>
#include <stdlib.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t xoshiro256pp_next(Xoshiro256State* state) {
    const uint64_t result = rotl(state->s[0] + state->s[3], 23) + state->s[0];
    const uint64_t t = state->s[1] << 17;

    state->s[2] ^= state->s[0];
    state->s[3] ^= state->s[1];
    state->s[1] ^= state->s[2];
    state->s[0] ^= state->s[3];

    state->s[2] ^= t;
    state->s[3] = rotl(state->s[3], 45);

    return result;
}

static inline float xoshiro_next_f32(Xoshiro256State* state) {
    return (float)(xoshiro256pp_next(state) >> 40) * 0x1.0p-24f;
}

static inline double xoshiro_next_f64(Xoshiro256State* state) {
    return (double)(xoshiro256pp_next(state) >> 11) * 0x1.0p-53;
}

int rand_fill_host_f32(float* out, size_t size) {
    if (!out) return -1;
    Xoshiro256State* state = rp_get_cpu_state();
    for (size_t i = 0; i < size; i++) {
        out[i] = xoshiro_next_f32(state);
    }
    return 0;
}

int rand_fill_host_f64(double* out, size_t size) {
    if (!out) return -1;
    Xoshiro256State* state = rp_get_cpu_state();
    for (size_t i = 0; i < size; i++) {
        out[i] = xoshiro_next_f64(state);
    }
    return 0;
}

static void box_muller_f32(Xoshiro256State* state, float* z0, float* z1) {
    float u1, u2;
    do {
        u1 = xoshiro_next_f32(state);
    } while (u1 <= 0.0f);
    u2 = xoshiro_next_f32(state);

    float mag = sqrtf(-2.0f * logf(u1));
    *z0 = mag * cosf(2.0f * (float)M_PI * u2);
    *z1 = mag * sinf(2.0f * (float)M_PI * u2);
}

static void box_muller_f64(Xoshiro256State* state, double* z0, double* z1) {
    double u1, u2;
    do {
        u1 = xoshiro_next_f64(state);
    } while (u1 <= 0.0);
    u2 = xoshiro_next_f64(state);

    double mag = sqrt(-2.0 * log(u1));
    *z0 = mag * cos(2.0 * M_PI * u2);
    *z1 = mag * sin(2.0 * M_PI * u2);
}

static int randn_box_muller_f32(float* out, size_t size) {
    Xoshiro256State* state = rp_get_cpu_state();

    size_t i = 0;
    if (state->has_cached_normal && size > 0) {
        out[0] = (float)state->cached_normal;
        state->has_cached_normal = false;
        i = 1;
    }

    for (; i + 1 < size; i += 2) {
        box_muller_f32(state, &out[i], &out[i + 1]);
    }

    if (i < size) {
        float z0, z1;
        box_muller_f32(state, &z0, &z1);
        out[i] = z0;
        state->cached_normal = (double)z1;
        state->has_cached_normal = true;
    }

    return 0;
}

static int randn_box_muller_f64(double* out, size_t size) {
    Xoshiro256State* state = rp_get_cpu_state();

    size_t i = 0;
    if (state->has_cached_normal && size > 0) {
        out[0] = state->cached_normal;
        state->has_cached_normal = false;
        i = 1;
    }

    for (; i + 1 < size; i += 2) {
        box_muller_f64(state, &out[i], &out[i + 1]);
    }

    if (i < size) {
        double z0, z1;
        box_muller_f64(state, &z0, &z1);
        out[i] = z0;
        state->cached_normal = z1;
        state->has_cached_normal = true;
    }

    return 0;
}

static const double ZIG_R = 3.44428647676;
static const double ZIG_V = 0.00991256303526217;

static double zig_x[129];
static double zig_ratio[128];
static bool ziggurat_initialized = false;

static void ziggurat_init(void) {
    if (ziggurat_initialized) return;

    zig_x[0] = ZIG_V / exp(-0.5 * ZIG_R * ZIG_R);
    zig_x[1] = ZIG_R;
    zig_x[128] = 0.0;

    for (int i = 2; i < 128; i++) {
        zig_x[i] = sqrt(-2.0 * log(ZIG_V / zig_x[i - 1] + exp(-0.5 * zig_x[i - 1] * zig_x[i - 1])));
    }

    for (int i = 0; i < 128; i++) {
        zig_ratio[i] = zig_x[i + 1] / zig_x[i];
    }

    ziggurat_initialized = true;
}

static double ziggurat_sample(Xoshiro256State* state) {
    while (1) {
        double u = 2.0 * xoshiro_next_f64(state) - 1.0;
        int i = (int)(xoshiro256pp_next(state) & 0x7F);

        if (fabs(u) < zig_ratio[i]) {
            return u * zig_x[i];
        }

        if (i == 0) {
            double x, y;
            do {
                double u1 = xoshiro_next_f64(state);
                double u2 = xoshiro_next_f64(state);
                if (u1 <= 0.0) u1 = 1e-15;
                x = -log(u1) / ZIG_R;
                y = -log(u2);
            } while (2.0 * y < x * x);
            return (u > 0) ? x + ZIG_R : -(x + ZIG_R);
        }

        double x = u * zig_x[i];
        double f_x = exp(-0.5 * x * x);
        double f_i = exp(-0.5 * zig_x[i] * zig_x[i]);
        double f_i1 = exp(-0.5 * zig_x[i + 1] * zig_x[i + 1]);

        if (f_i1 + xoshiro_next_f64(state) * (f_i - f_i1) < f_x) {
            return x;
        }
    }
}

static int randn_ziggurat_f32(float* out, size_t size) {
    ziggurat_init();
    Xoshiro256State* state = rp_get_cpu_state();
    for (size_t i = 0; i < size; i++) {
        out[i] = (float)ziggurat_sample(state);
    }
    return 0;
}

static int randn_ziggurat_f64(double* out, size_t size) {
    ziggurat_init();
    Xoshiro256State* state = rp_get_cpu_state();
    for (size_t i = 0; i < size; i++) {
        out[i] = ziggurat_sample(state);
    }
    return 0;
}

static double inverse_cdf_rational(double u) {
    static const double a[] = {
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00
    };
    static const double b[] = {
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01
    };
    static const double c[] = {
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00
    };
    static const double d[] = {
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00
    };

    static const double p_low  = 0.02425;
    static const double p_high = 1.0 - 0.02425;

    double q, r;

    if (u < p_low) {
        q = sqrt(-2.0 * log(u));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (u <= p_high) {
        q = u - 0.5;
        r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        q = sqrt(-2.0 * log(1.0 - u));
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
}

static int randn_inverse_cdf_f32(float* out, size_t size) {
    Xoshiro256State* state = rp_get_cpu_state();
    for (size_t i = 0; i < size; i++) {
        double u;
        do {
            u = xoshiro_next_f64(state);
        } while (u <= 0.0 || u >= 1.0);
        out[i] = (float)inverse_cdf_rational(u);
    }
    return 0;
}

static int randn_inverse_cdf_f64(double* out, size_t size) {
    Xoshiro256State* state = rp_get_cpu_state();
    for (size_t i = 0; i < size; i++) {
        double u;
        do {
            u = xoshiro_next_f64(state);
        } while (u <= 0.0 || u >= 1.0);
        out[i] = inverse_cdf_rational(u);
    }
    return 0;
}

int randn_fill_host_f32(float* out, size_t size, NormalMethod method) {
    if (!out) return -1;
    switch (method) {
        case NORMAL_BOX_MULLER:   return randn_box_muller_f32(out, size);
        case NORMAL_ZIGGURAT:     return randn_ziggurat_f32(out, size);
        case NORMAL_INVERSE_CDF:  return randn_inverse_cdf_f32(out, size);
        default: return -1;
    }
}

int randn_fill_host_f64(double* out, size_t size, NormalMethod method) {
    if (!out) return -1;
    switch (method) {
        case NORMAL_BOX_MULLER:   return randn_box_muller_f64(out, size);
        case NORMAL_ZIGGURAT:     return randn_ziggurat_f64(out, size);
        case NORMAL_INVERSE_CDF:  return randn_inverse_cdf_f64(out, size);
        default: return -1;
    }
}

int randint_fill_host_f32(float* out, size_t size, float low, float high) {
    if (!out) return -1;
    Xoshiro256State* state = rp_get_cpu_state();
    float range = high - low;
    for (size_t i = 0; i < size; i++) {
        out[i] = floorf(low + xoshiro_next_f32(state) * range);
    }
    return 0;
}

int randint_fill_host_f64(double* out, size_t size, double low, double high) {
    if (!out) return -1;
    Xoshiro256State* state = rp_get_cpu_state();
    double range = high - low;
    for (size_t i = 0; i < size; i++) {
        out[i] = floor(low + xoshiro_next_f64(state) * range);
    }
    return 0;
}
