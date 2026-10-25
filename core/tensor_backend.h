#pragma once

#include "tensor.h"

#if defined(__CUDACC__)
#define RP_HOST_DEVICE __host__ __device__
#else
#define RP_HOST_DEVICE
#endif

namespace recpulse {

inline constexpr int32_t kMaxDims = 16;

struct Layout {
    int32_t ndim;
    bool contiguous;
    int64_t numel;
    int64_t shape[kMaxDims];
    int64_t strides[kMaxDims];
};

RP_HOST_DEVICE inline int64_t offset_of(int64_t index, const Layout& layout) {
    if (layout.contiguous) return index;
    int64_t offset = 0;
    for (int32_t d = layout.ndim - 1; d >= 0; --d) {
        offset += (index % layout.shape[d]) * layout.strides[d];
        index /= layout.shape[d];
    }
    return offset;
}

void* alloc_host(std::size_t nbytes);
void free_host(void* data) noexcept;
void fill_host(void* data, DType dtype, const Layout& layout, double value);
void copy_host(void* dst, DType dst_dtype, const Layout& dst_layout,
               const void* src, DType src_dtype, const Layout& src_layout);

void* alloc_device(std::size_t nbytes, int32_t device);
void free_device(void* data, int32_t device) noexcept;
void fill_device(void* data, DType dtype, const Layout& layout, double value, int32_t device);
void copy_device(void* dst, DType dst_dtype, const Layout& dst_layout,
                 const void* src, DType src_dtype, const Layout& src_layout, int32_t device);
void memcpy_across(void* dst, Device dst_device, const void* src, Device src_device, std::size_t nbytes);

}
