#include "tensor_backend.hpp"

#include <bit>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <type_traits>

namespace recpulse {

namespace {

uint16_t float_to_half_bits(float value) {
    uint32_t bits = std::bit_cast<uint32_t>(value);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    bits &= 0x7fffffffu;
    if (bits >= (143u << 23)) {
        return static_cast<uint16_t>(sign | (bits > 0x7f800000u ? 0x7e00u : 0x7c00u));
    }
    if (bits < (113u << 23)) {
        const float denorm_magic = std::bit_cast<float>(126u << 23);
        const float shifted = std::bit_cast<float>(bits) + denorm_magic;
        return static_cast<uint16_t>(sign | (std::bit_cast<uint32_t>(shifted) - (126u << 23)));
    }
    const uint32_t mant_odd = (bits >> 13) & 1u;
    bits += 0xc8000fffu + mant_odd;
    return static_cast<uint16_t>(sign | (bits >> 13));
}

float half_bits_to_float(uint16_t half) {
    constexpr uint32_t shifted_exp = 0x7c00u << 13;
    uint32_t bits = (static_cast<uint32_t>(half) & 0x7fffu) << 13;
    const uint32_t exp = bits & shifted_exp;
    bits += (127u - 15u) << 23;
    if (exp == shifted_exp) {
        bits += (128u - 16u) << 23;
    } else if (exp == 0) {
        bits += 1u << 23;
        bits = std::bit_cast<uint32_t>(std::bit_cast<float>(bits) - std::bit_cast<float>(113u << 23));
    }
    return std::bit_cast<float>(bits | ((static_cast<uint32_t>(half) & 0x8000u) << 16));
}

uint16_t float_to_bfloat_bits(float value) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((bits >> 16) | 0x0040u);
    return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

float bfloat_bits_to_float(uint16_t bfloat) {
    return std::bit_cast<float>(static_cast<uint32_t>(bfloat) << 16);
}

template <typename F>
void dispatch(DType dtype, F&& f) {
    switch (dtype) {
    case DType::Float32: f(std::type_identity<float>{}); return;
    case DType::Float64: f(std::type_identity<double>{}); return;
    case DType::Float16: f(std::type_identity<Float16>{}); return;
    case DType::BFloat16: f(std::type_identity<BFloat16>{}); return;
    }
    throw std::invalid_argument("unknown dtype");
}

template <typename Dst, typename Src>
Dst convert(Src value) {
    if constexpr (std::is_same_v<Dst, Src>) {
        return value;
    } else if constexpr (std::is_arithmetic_v<Dst>) {
        return static_cast<Dst>(value);
    } else {
        return Dst(static_cast<float>(value));
    }
}

}

Float16::Float16(float value) : bits(float_to_half_bits(value)) {}
Float16::operator float() const { return half_bits_to_float(bits); }

Float16 Float16::from_bits(uint16_t raw) {
    Float16 result;
    result.bits = raw;
    return result;
}

BFloat16::BFloat16(float value) : bits(float_to_bfloat_bits(value)) {}
BFloat16::operator float() const { return bfloat_bits_to_float(bits); }

BFloat16 BFloat16::from_bits(uint16_t raw) {
    BFloat16 result;
    result.bits = raw;
    return result;
}

void* alloc_host(std::size_t nbytes) {
    void* data = std::malloc(nbytes);
    if (!data) throw std::bad_alloc();
    return data;
}

void free_host(void* data) noexcept {
    std::free(data);
}

void fill_host(void* data, DType dtype, const Layout& layout, double value) {
    if (std::bit_cast<uint64_t>(value) == 0 && layout.contiguous) {
        std::memset(data, 0, static_cast<std::size_t>(layout.numel) * dtype_size(dtype));
        return;
    }
    dispatch(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        const T filler = convert<T>(value);
        T* out = static_cast<T*>(data);
        for (int64_t i = 0; i < layout.numel; ++i) out[offset_of(i, layout)] = filler;
    });
}

void copy_host(void* dst, DType dst_dtype, const Layout& dst_layout,
               const void* src, DType src_dtype, const Layout& src_layout) {
    if (dst_dtype == src_dtype && dst_layout.contiguous && src_layout.contiguous) {
        std::memcpy(dst, src, static_cast<std::size_t>(dst_layout.numel) * dtype_size(dst_dtype));
        return;
    }
    dispatch(dst_dtype, [&](auto dst_tag) {
        dispatch(src_dtype, [&](auto src_tag) {
            using D = typename decltype(dst_tag)::type;
            using S = typename decltype(src_tag)::type;
            D* out = static_cast<D*>(dst);
            const S* in = static_cast<const S*>(src);
            for (int64_t i = 0; i < dst_layout.numel; ++i) {
                out[offset_of(i, dst_layout)] = convert<D>(in[offset_of(i, src_layout)]);
            }
        });
    });
}

}
