#include "tensor_backend.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <bit>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace recpulse {

namespace {

constexpr int kThreads = 256;
constexpr int64_t kMaxBlocks = 65535;

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

class DeviceGuard {
public:
    explicit DeviceGuard(int32_t device) {
        check(cudaGetDevice(&previous_), "cudaGetDevice");
        check(cudaSetDevice(device), "cudaSetDevice");
    }
    ~DeviceGuard() { cudaSetDevice(previous_); }

    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

private:
    int previous_ = 0;
};

int blocks_for(int64_t numel) {
    return static_cast<int>(std::min<int64_t>((numel + kThreads - 1) / kThreads, kMaxBlocks));
}

template <typename F>
void dispatch(DType dtype, F&& f) {
    switch (dtype) {
    case DType::Float32: f(std::type_identity<float>{}); return;
    case DType::Float64: f(std::type_identity<double>{}); return;
    case DType::Float16: f(std::type_identity<__half>{}); return;
    case DType::BFloat16: f(std::type_identity<__nv_bfloat16>{}); return;
    }
    throw std::invalid_argument("unknown dtype");
}

template <typename Dst, typename Src>
__host__ __device__ Dst convert(Src value) {
    if constexpr (std::is_same_v<Dst, Src>) {
        return value;
    } else if constexpr (std::is_arithmetic_v<Dst> && std::is_arithmetic_v<Src>) {
        return static_cast<Dst>(value);
    } else if constexpr (std::is_arithmetic_v<Dst>) {
        return static_cast<Dst>(static_cast<float>(value));
    } else {
        return Dst(static_cast<float>(value));
    }
}

template <typename T>
__global__ void fill_kernel(T* data, Layout layout, T value) {
    const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < layout.numel; i += step) {
        data[offset_of(i, layout)] = value;
    }
}

template <typename D, typename S>
__global__ void copy_kernel(D* dst, Layout dst_layout, const S* src, Layout src_layout) {
    const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < dst_layout.numel; i += step) {
        dst[offset_of(i, dst_layout)] = convert<D>(src[offset_of(i, src_layout)]);
    }
}

}

void* alloc_device(std::size_t nbytes, int32_t device) {
    DeviceGuard guard(device);
    void* data = nullptr;
    check(cudaMalloc(&data, nbytes), "cudaMalloc");
    return data;
}

void free_device(void* data, int32_t device) noexcept {
    int previous = 0;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    cudaFree(data);
    cudaSetDevice(previous);
}

void fill_device(void* data, DType dtype, const Layout& layout, double value, int32_t device) {
    DeviceGuard guard(device);
    if (std::bit_cast<uint64_t>(value) == 0 && layout.contiguous) {
        check(cudaMemset(data, 0, static_cast<std::size_t>(layout.numel) * dtype_size(dtype)), "cudaMemset");
        return;
    }
    dispatch(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        fill_kernel<T><<<blocks_for(layout.numel), kThreads>>>(static_cast<T*>(data), layout, convert<T>(value));
    });
    check(cudaGetLastError(), "fill_kernel");
}

void copy_device(void* dst, DType dst_dtype, const Layout& dst_layout,
                 const void* src, DType src_dtype, const Layout& src_layout, int32_t device) {
    DeviceGuard guard(device);
    if (dst_dtype == src_dtype && dst_layout.contiguous && src_layout.contiguous) {
        const std::size_t nbytes = static_cast<std::size_t>(dst_layout.numel) * dtype_size(dst_dtype);
        check(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToDevice), "cudaMemcpy");
        return;
    }
    dispatch(dst_dtype, [&](auto dst_tag) {
        dispatch(src_dtype, [&](auto src_tag) {
            using D = typename decltype(dst_tag)::type;
            using S = typename decltype(src_tag)::type;
            copy_kernel<D, S><<<blocks_for(dst_layout.numel), kThreads>>>(
                static_cast<D*>(dst), dst_layout, static_cast<const S*>(src), src_layout);
        });
    });
    check(cudaGetLastError(), "copy_kernel");
}

void memcpy_across(void* dst, Device dst_device, const void* src, Device src_device, std::size_t nbytes) {
    if (dst_device.is_cuda() && src_device.is_cuda()) {
        DeviceGuard guard(dst_device.index);
        cudaError_t err = cudaDeviceEnablePeerAccess(src_device.index, 0);
        if (err != cudaSuccess) cudaGetLastError();
        check(cudaMemcpyPeer(dst, dst_device.index, src, src_device.index, nbytes), "cudaMemcpyPeer");
    } else if (dst_device.is_cuda()) {
        DeviceGuard guard(dst_device.index);
        check(cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice), "cudaMemcpy");
    } else {
        DeviceGuard guard(src_device.index);
        check(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToHost), "cudaMemcpy");
    }
}

}
