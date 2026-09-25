#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace recpulse {

enum class DType : int8_t {
    Float32,
    Float64,
    Float16,
    BFloat16,
};

std::size_t dtype_size(DType dtype);

struct Float16 {
    uint16_t bits;

    Float16() = default;
    explicit Float16(float value);
    operator float() const;

    static Float16 from_bits(uint16_t raw);
};

struct BFloat16 {
    uint16_t bits;

    BFloat16() = default;
    explicit BFloat16(float value);
    operator float() const;

    static BFloat16 from_bits(uint16_t raw);
};

static_assert(sizeof(Float16) == 2);
static_assert(sizeof(BFloat16) == 2);

struct Device {
    int32_t index = -1;

    static constexpr Device cpu() { return Device{-1}; }
    static constexpr Device cuda(int32_t device_index = 0) { return Device{device_index}; }

    constexpr bool is_cpu() const { return index < 0; }
    constexpr bool is_cuda() const { return index >= 0; }

    bool operator==(const Device&) const = default;
};

using Shape = std::vector<int64_t>;
using Strides = std::vector<int64_t>;

class Storage {
public:
    Storage(std::size_t nbytes, Device device);
    ~Storage();

    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    void* data() { return data_; }
    const void* data() const { return data_; }
    std::size_t nbytes() const { return nbytes_; }
    Device device() const { return device_; }

    uint64_t version() const { return version_; }
    void bump_version() { ++version_; }

private:
    void* data_ = nullptr;
    std::size_t nbytes_ = 0;
    Device device_;
    uint64_t version_ = 0;
};

class GradFn;
struct AutogradMeta;

class TensorImpl {
public:
    TensorImpl(std::shared_ptr<Storage> storage, DType dtype, Shape shape, Strides strides, int64_t offset);
    ~TensorImpl();

    DType dtype() const { return dtype_; }
    Device device() const { return storage_->device(); }
    const Shape& shape() const { return shape_; }
    const Strides& strides() const { return strides_; }
    int64_t offset() const { return offset_; }
    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
    int64_t numel() const { return numel_; }
    bool is_contiguous() const;

    const std::shared_ptr<Storage>& storage() const { return storage_; }
    const void* data_ptr() const;
    void* mutable_data_ptr();

    const AutogradMeta* autograd_meta() const { return autograd_meta_.get(); }
    AutogradMeta* autograd_meta() { return autograd_meta_.get(); }
    AutogradMeta& ensure_autograd_meta();

private:
    std::shared_ptr<Storage> storage_;
    DType dtype_;
    Shape shape_;
    Strides strides_;
    int64_t offset_ = 0;
    int64_t numel_ = 0;
    std::unique_ptr<AutogradMeta> autograd_meta_;
};

class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::shared_ptr<TensorImpl> impl);

    static Tensor empty(const Shape& shape, DType dtype = DType::Float32, Device device = Device::cpu());
    static Tensor zeros(const Shape& shape, DType dtype = DType::Float32, Device device = Device::cpu());
    static Tensor ones(const Shape& shape, DType dtype = DType::Float32, Device device = Device::cpu());
    static Tensor full(const Shape& shape, double value, DType dtype = DType::Float32, Device device = Device::cpu());
    static Tensor from_data(const void* src, DType src_dtype, Device src_device,
                            const Shape& shape, DType dtype, Device device);

    bool defined() const { return impl_ != nullptr; }
    bool is_same(const Tensor& other) const { return impl_ == other.impl_; }
    TensorImpl* impl() const { return impl_.get(); }

    DType dtype() const;
    Device device() const;
    const Shape& shape() const;
    const Strides& strides() const;
    int64_t offset() const;
    int64_t ndim() const;
    int64_t numel() const;
    int64_t size(int64_t dim) const;
    bool is_contiguous() const;

    const void* data_ptr() const;
    void* mutable_data_ptr() const;
    template <typename T> const T* data() const { return static_cast<const T*>(data_ptr()); }
    template <typename T> T* mutable_data() const { return static_cast<T*>(mutable_data_ptr()); }

    Tensor to(Device device) const;
    Tensor to(DType dtype) const;
    Tensor to(Device device, DType dtype) const;
    Tensor clone() const;
    Tensor contiguous() const;
    Tensor as_strided(const Shape& shape, const Strides& strides, int64_t offset) const;

    const Tensor& copy_(const Tensor& src) const;
    const Tensor& fill_(double value) const;

    bool requires_grad() const;
    const Tensor& requires_grad_(bool requires_grad = true) const;
    bool is_leaf() const;
    Tensor grad() const;
    std::shared_ptr<GradFn> grad_fn() const;
    uint64_t version() const;
    void backward() const;
    void zero_grad() const;
    Tensor detach() const;

private:
    std::shared_ptr<TensorImpl> impl_;
};

struct AutogradMeta {
    bool requires_grad = false;
    std::shared_ptr<GradFn> grad_fn;
    Tensor grad;
};

class GradMode {
public:
    static bool is_enabled();
    static void set_enabled(bool enabled);
};

class NoGradGuard {
public:
    NoGradGuard();
    ~NoGradGuard();

    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

private:
    bool previous_;
};

Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator/(const Tensor& a, const Tensor& b);

Tensor operator+(const Tensor& a, double s);
Tensor operator-(const Tensor& a, double s);
Tensor operator*(const Tensor& a, double s);
Tensor operator/(const Tensor& a, double s);

Tensor operator+(double s, const Tensor& a);
Tensor operator-(double s, const Tensor& a);
Tensor operator*(double s, const Tensor& a);
Tensor operator/(double s, const Tensor& a);

Tensor operator-(const Tensor& a);

}
