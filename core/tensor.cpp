#include "tensor_backend.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace recpulse {

namespace {

thread_local bool grad_enabled = true;

TensorImpl& get(const Tensor& tensor) {
    if (!tensor.defined()) throw std::logic_error("accessing an undefined tensor");
    return *tensor.impl();
}

int64_t checked_numel(const Shape& shape) {
    if (shape.size() == 0) throw std::invalid_argument("tensor must have at least one dimension");
    int64_t numel = 1;
    for (int64_t dim : shape) {
        if (dim < 0) throw std::invalid_argument("shape of a tensor can't have a negative dimension");
        if (dim == 0) throw std::invalid_argument("total size of tensor must not be zero");
        if (__builtin_mul_overflow(numel, dim, &numel)) throw std::overflow_error("tensor size overflow");
    }
    return numel;
}

std::size_t checked_nbytes(const Shape& shape, DType dtype) {
    std::size_t nbytes = 0;
    if (__builtin_mul_overflow(static_cast<std::size_t>(checked_numel(shape)), dtype_size(dtype), &nbytes)) {
        throw std::overflow_error("tensor size overflow");
    }
    return nbytes;
}

std::string format_dims(const std::vector<int64_t>& dims) {
    std::string out = "[";
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) out += ", ";
        out += std::to_string(dims[i]);
    }
    return out + "]";
}

std::string describe_view(const Shape& shape, const Strides& strides, int64_t offset) {
    return "shape " + format_dims(shape) + ", strides " + format_dims(strides) + ", offset " + std::to_string(offset);
}

Strides contiguous_strides(const Shape& shape) {
    Strides strides(shape.size());
    int64_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

Layout contiguous_layout(int64_t numel) {
    Layout layout{};
    layout.ndim = 1;
    layout.contiguous = true;
    layout.numel = numel;
    layout.shape[0] = numel;
    layout.strides[0] = 1;
    return layout;
}

Layout make_layout(const TensorImpl& impl) {
    if (impl.is_contiguous()) return contiguous_layout(impl.numel());  // will only be used for element-wise ops
    if (impl.ndim() > kMaxDims) {
        throw std::invalid_argument("non-contiguous tensors support at most " + std::to_string(kMaxDims) + " dimensions");
    }
    Layout layout{};
    layout.ndim = static_cast<int32_t>(impl.ndim());
    layout.contiguous = false;
    layout.numel = impl.numel();
    for (int32_t d = 0; d < layout.ndim; ++d) {
        layout.shape[d] = impl.shape()[d];
        layout.strides[d] = impl.strides()[d];
    }
    return layout;
}

void copy_on_device(void* dst, DType dst_dtype, const Layout& dst_layout,
                    const void* src, DType src_dtype, const Layout& src_layout, Device device) {
    if (device.is_cpu()) {
        copy_host(dst, dst_dtype, dst_layout, src, src_dtype, src_layout);
    } else {
        copy_device(dst, dst_dtype, dst_layout, src, src_dtype, src_layout, device.index);
    }
}

void copy_raw(void* dst, DType dst_dtype, const Layout& dst_layout, Device dst_device,
              const void* src, DType src_dtype, const Layout& src_layout, Device src_device) {
    const int64_t numel = dst_layout.numel;
    if (dst_device == src_device) {
        copy_on_device(dst, dst_dtype, dst_layout, src, src_dtype, src_layout, dst_device);
        return;
    }

    Tensor dense;
    if (!src_layout.contiguous) {
        dense = Tensor::empty({numel}, src_dtype, src_device);
        copy_on_device(dense.mutable_data_ptr(), src_dtype, contiguous_layout(numel),
                       src, src_dtype, src_layout, src_device);
        src = dense.data_ptr();
    }

    const std::size_t nbytes = static_cast<std::size_t>(numel) * dtype_size(src_dtype);
    if (dst_layout.contiguous && dst_dtype == src_dtype) {
        memcpy_across(dst, dst_device, src, src_device, nbytes);
        return;
    }

    Tensor staged = Tensor::empty({numel}, src_dtype, dst_device);
    memcpy_across(staged.mutable_data_ptr(), dst_device, src, src_device, nbytes);
    copy_on_device(dst, dst_dtype, dst_layout, staged.data_ptr(), src_dtype, contiguous_layout(numel), dst_device);
}

Tensor copy_to(const TensorImpl& src, Device device, DType dtype) {
    Tensor result = Tensor::empty(src.shape(), dtype, device);
    copy_raw(result.mutable_data_ptr(), dtype, contiguous_layout(src.numel()), device,
             src.data_ptr(), src.dtype(), make_layout(src), src.device());
    return result;
}

void check_inplace(const Tensor& tensor) {
    for (int64_t d = 0; d < tensor.ndim(); ++d) {
        if (tensor.shape()[d] > 1 && tensor.strides()[d] == 0) {
            throw std::invalid_argument("in-place write to a tensor whose elements share memory (stride 0); clone() it first");
        }
    }
    if (GradMode::is_enabled() && tensor.requires_grad() && tensor.is_leaf()) {
        throw std::runtime_error("in-place operation on a leaf tensor that requires grad; use NoGradGuard");
    }
}

}

std::size_t dtype_size(DType dtype) {
    switch (dtype) {
    case DType::Float32: return 4;
    case DType::Float64: return 8;
    case DType::Float16: return 2;
    case DType::BFloat16: return 2;
    }
    throw std::invalid_argument("unknown dtype");
}

Storage::Storage(std::size_t nbytes, Device device) : nbytes_(nbytes), device_(device) {
    if (nbytes_ == 0) return;
    data_ = device_.is_cpu() ? alloc_host(nbytes_) : alloc_device(nbytes_, device_.index);
}

Storage::~Storage() {
    if (!data_) return;
    if (device_.is_cpu()) {
        free_host(data_);
    } else {
        free_device(data_, device_.index);
    }
}

TensorImpl::TensorImpl(std::shared_ptr<Storage> storage, DType dtype, Shape shape, Strides strides, int64_t offset)
    : storage_(std::move(storage)),
      dtype_(dtype),
      shape_(std::move(shape)),
      strides_(std::move(strides)),
      offset_(offset),
      numel_(checked_numel(shape_)) {
    if (!storage_) throw std::invalid_argument("tensor needs a storage");
    if (shape_.size() != strides_.size()) throw std::invalid_argument("shape and strides differ in length");
    if (offset_ < 0) throw std::invalid_argument("negative offset");
    int64_t end = offset_;
    for (std::size_t d = 0; d < shape_.size(); ++d) {
        if (strides_[d] < 0) throw std::invalid_argument("negative stride");
        int64_t span = 0;
        if (__builtin_mul_overflow(shape_[d] - 1, strides_[d], &span) || __builtin_add_overflow(end, span, &end)) {
            throw std::out_of_range("view size overflows int64: " + describe_view(shape_, strides_, offset_));
        }
    }
    std::size_t needed = 0;
    if (__builtin_add_overflow(end, 1, &end) ||
        __builtin_mul_overflow(static_cast<std::size_t>(end), dtype_size(dtype_), &needed)) {
        throw std::out_of_range("view size overflows int64: " + describe_view(shape_, strides_, offset_));
    }
    if (needed > storage_->nbytes()) {
        throw std::out_of_range("view exceeds its storage: " + describe_view(shape_, strides_, offset_) + " needs " +
                                std::to_string(needed) + " bytes, storage has " + std::to_string(storage_->nbytes()));
    }
}

TensorImpl::~TensorImpl() = default;

bool TensorImpl::is_contiguous() const {
    int64_t expected = 1;
    for (std::size_t d = shape_.size(); d-- > 0;) {
        if (shape_[d] == 1) continue;
        if (strides_[d] != expected) return false;
        expected *= shape_[d];
    }
    return true;
}

const void* TensorImpl::data_ptr() const {
    return static_cast<const char*>(storage_->data()) + offset_ * static_cast<int64_t>(dtype_size(dtype_));
}

void* TensorImpl::mutable_data_ptr() {
    storage_->bump_version();
    return static_cast<char*>(storage_->data()) + offset_ * static_cast<int64_t>(dtype_size(dtype_));
}

AutogradMeta& TensorImpl::ensure_autograd_meta() {
    if (!autograd_meta_) autograd_meta_ = std::make_unique<AutogradMeta>();
    return *autograd_meta_;
}

Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}

Tensor Tensor::empty(const Shape& shape, DType dtype, Device device) {
    auto storage = std::make_shared<Storage>(checked_nbytes(shape, dtype), device);
    return Tensor(std::make_shared<TensorImpl>(std::move(storage), dtype, shape, contiguous_strides(shape), 0));
}

Tensor Tensor::zeros(const Shape& shape, DType dtype, Device device) {
    return full(shape, 0.0, dtype, device);
}

Tensor Tensor::ones(const Shape& shape, DType dtype, Device device) {
    return full(shape, 1.0, dtype, device);
}

Tensor Tensor::full(const Shape& shape, double value, DType dtype, Device device) {
    Tensor result = empty(shape, dtype, device);
    result.fill_(value);
    return result;
}

Tensor Tensor::from_data(const void* src, DType src_dtype, Device src_device,
                         const Shape& shape, DType dtype, Device device) {
    Tensor result = empty(shape, dtype, device);
    const Layout layout = contiguous_layout(result.numel());
    copy_raw(result.mutable_data_ptr(), dtype, layout, device, src, src_dtype, layout, src_device);
    return result;
}

DType Tensor::dtype() const { return get(*this).dtype(); }
Device Tensor::device() const { return get(*this).device(); }
const Shape& Tensor::shape() const { return get(*this).shape(); }
const Strides& Tensor::strides() const { return get(*this).strides(); }
int64_t Tensor::offset() const { return get(*this).offset(); }
int64_t Tensor::ndim() const { return get(*this).ndim(); }
int64_t Tensor::numel() const { return get(*this).numel(); }
bool Tensor::is_contiguous() const { return get(*this).is_contiguous(); }

int64_t Tensor::size(int64_t dim) const {
    const TensorImpl& self = get(*this);
    const int64_t normalized = dim < 0 ? dim + self.ndim() : dim;
    if (normalized < 0 || normalized >= self.ndim()) throw std::out_of_range("dimension out of range");
    return self.shape()[normalized];
}

const void* Tensor::data_ptr() const { return get(*this).data_ptr(); }
void* Tensor::mutable_data_ptr() const { return get(*this).mutable_data_ptr(); }

Tensor Tensor::to(Device device) const { return to(device, dtype()); }
Tensor Tensor::to(DType dtype) const { return to(device(), dtype); }

Tensor Tensor::to(Device device, DType dtype) const {
    const TensorImpl& self = get(*this);
    if (device == self.device() && dtype == self.dtype()) return *this;
    Tensor result = copy_to(self, device, dtype);
    if (const AutogradMeta* meta = self.autograd_meta()) {
        AutogradMeta& out = result.impl()->ensure_autograd_meta();
        out.requires_grad = meta->requires_grad;
        if (meta->grad.defined()) out.grad = meta->grad.to(device, dtype);
    }
    return result;
}

Tensor Tensor::clone() const {
    const TensorImpl& self = get(*this);
    Tensor result = copy_to(self, self.device(), self.dtype());
    if (requires_grad()) result.impl()->ensure_autograd_meta().requires_grad = true;
    return result;
}

Tensor Tensor::contiguous() const {
    return is_contiguous() ? *this : clone();
}

Tensor Tensor::as_strided(const Shape& shape, const Strides& strides, int64_t offset) const {
    const TensorImpl& self = get(*this);
    return Tensor(std::make_shared<TensorImpl>(self.storage(), self.dtype(), shape, strides, offset));
}

const Tensor& Tensor::copy_(const Tensor& src) const {
    TensorImpl& self = get(*this);
    const TensorImpl& other = get(src);
    if (self.shape() != other.shape()) throw std::invalid_argument("copy_: shape mismatch");
    check_inplace(*this);
    const Tensor source = self.storage() == other.storage() ? src.clone() : src;
    const TensorImpl& from = *source.impl();
    copy_raw(self.mutable_data_ptr(), self.dtype(), make_layout(self), self.device(),
             from.data_ptr(), from.dtype(), make_layout(from), from.device());
    return *this;
}

const Tensor& Tensor::fill_(double value) const {
    TensorImpl& self = get(*this);
    check_inplace(*this);
    const Layout layout = make_layout(self);
    void* data = self.mutable_data_ptr();
    if (self.device().is_cpu()) {
        fill_host(data, self.dtype(), layout, value);
    } else {
        fill_device(data, self.dtype(), layout, value, self.device().index);
    }
    return *this;
}

bool Tensor::requires_grad() const {
    const AutogradMeta* meta = std::as_const(get(*this)).autograd_meta();
    return meta && meta->requires_grad;
}

const Tensor& Tensor::requires_grad_(bool requires_grad) const {
    get(*this).ensure_autograd_meta().requires_grad = requires_grad;
    return *this;
}

bool Tensor::is_leaf() const {
    const AutogradMeta* meta = std::as_const(get(*this)).autograd_meta();
    return !meta || !meta->grad_fn;
}

Tensor Tensor::grad() const {
    const AutogradMeta* meta = std::as_const(get(*this)).autograd_meta();
    return meta ? meta->grad : Tensor();
}

std::shared_ptr<GradFn> Tensor::grad_fn() const {
    const AutogradMeta* meta = std::as_const(get(*this)).autograd_meta();
    return meta ? meta->grad_fn : nullptr;
}

uint64_t Tensor::version() const { return get(*this).storage()->version(); }

void Tensor::zero_grad() const {
    if (AutogradMeta* meta = get(*this).autograd_meta()) meta->grad = Tensor();
}

Tensor Tensor::detach() const {
    return as_strided(shape(), strides(), offset());
}

bool GradMode::is_enabled() { return grad_enabled; }
void GradMode::set_enabled(bool enabled) { grad_enabled = enabled; }

NoGradGuard::NoGradGuard() : previous_(GradMode::is_enabled()) { GradMode::set_enabled(false); }
NoGradGuard::~NoGradGuard() { GradMode::set_enabled(previous_); }

}
