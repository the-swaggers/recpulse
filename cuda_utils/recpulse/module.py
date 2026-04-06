import recpulse_cuda as rp


class Module:
    def __init__(self):
        self.tracked = {}
        self._modules = {}
        self._training = True

    def track(self, name, tensor):
        self.tracked[name] = tensor
        return tensor

    def __setattr__(self, name, value):
        if name.startswith('_') or name == 'tracked':
            super().__setattr__(name, value)
            return

        if isinstance(value, Module):
            self._modules[name] = value
            for k, v in value.tracked.items():
                self.tracked[f"{name}.{k}"] = v
            for sub_name, sub_mod in value._modules.items():
                self._modules[f"{name}.{sub_name}"] = sub_mod

        super().__setattr__(name, value)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        self._intermediates = []
        for m in self._modules.values():
            m._intermediates = []
        result = self.forward(*args, **kwargs)
        return result

    def keep(self, tensor):
        self._intermediates.append(tensor)
        return tensor

    def learnable(self):
        return {k: v for k, v in self.tracked.items() if v.requires_grad}

    def frozen(self):
        return {k: v for k, v in self.tracked.items() if not v.requires_grad}

    def parameters(self):
        return list(self.learnable().values())

    def to(self, device=None, dtype=None):
        new_tracked = {}
        for name, tensor in self.tracked.items():
            new_t = tensor.to(device=device, dtype=dtype, inplace=True)
            new_tracked[name] = new_t
            parts = name.split('.')
            obj = self
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], new_t)
        self.tracked = new_tracked
        return self

    def train(self):
        self._training = True
        for m in self._modules.values():
            m.train()
        return self

    def eval(self):
        self._training = False
        for m in self._modules.values():
            m.eval()
        return self

    def zero_grad(self):
        for t in self.tracked.values():
            if t.requires_grad:
                t.zero_grad()


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        scale = (1.0 / in_features) ** 0.5
        w_raw = rp.randn([in_features, out_features])
        self.weight = w_raw.mul_scalar(scale)
        self.weight.requires_grad_(True)
        self.track("weight", self.weight)

        if bias:
            self.bias = rp.zeros([out_features])
            self.bias.requires_grad_(True)
            self.track("bias", self.bias)
        else:
            self.bias = None

    def forward(self, x):
        out = self.keep(x.op_matmul(self.weight))
        if self.bias is not None:
            out = self.keep(out.op_add(self.bias))
        return out


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(padding, int): padding = (padding, padding)
        if isinstance(dilation, int): dilation = (dilation, dilation)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        kH, kW = kernel_size
        fan_in = in_channels * kH * kW
        scale = (1.0 / fan_in) ** 0.5
        w_raw = rp.randn([out_channels * in_channels * kH * kW])
        w_scaled = w_raw.mul_scalar(scale)
        self._w_base = w_scaled
        self.weight = w_scaled.reshape([out_channels, in_channels, kH, kW])
        self.weight.requires_grad_(True)
        self.track("weight", self.weight)

        if bias:
            self.bias = rp.zeros([out_channels])
            self.bias.requires_grad_(True)
            self.track("bias", self.bias)
        else:
            self.bias = None

    def forward(self, x):
        return x.op_conv2d(
            self.weight, bias=self.bias,
            stride_h=self.stride[0], stride_w=self.stride[1],
            pad_h=self.padding[0], pad_w=self.padding[1],
            dilation_h=self.dilation[0], dilation_w=self.dilation[1]
        )


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        if stride is None: stride = kernel_size
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(padding, int): padding = (padding, padding)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return x.op_maxpool2d(
            self.kernel_size[0], self.kernel_size[1],
            stride_h=self.stride[0], stride_w=self.stride[1],
            pad_h=self.padding[0], pad_w=self.padding[1]
        )


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        if stride is None: stride = kernel_size
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(padding, int): padding = (padding, padding)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return x.op_avgpool2d(
            self.kernel_size[0], self.kernel_size[1],
            stride_h=self.stride[0], stride_w=self.stride[1],
            pad_h=self.padding[0], pad_w=self.padding[1]
        )


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        self.normalized_shape = list(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            total = 1
            for s in normalized_shape:
                total *= s
            self.weight = rp.ones([total])
            self.weight.requires_grad_(True)
            self.track("weight", self.weight)
            self.bias = rp.zeros([total])
            self.bias.requires_grad_(True)
            self.track("bias", self.bias)
        else:
            self.weight = None
            self.bias = None

    def forward(self, x):
        ndim_norm = len(self.normalized_shape)
        reduce_dims = list(range(x.ndim - ndim_norm, x.ndim))

        current = x
        for d in reversed(reduce_dims):
            mean = self.keep(current.op_mean_dim(d, keepdim=True))
            current = self.keep(current.op_sub(mean))
            sq = self.keep(current.op_square())
            var = self.keep(sq.op_mean_dim(d, keepdim=True))
            var_eps = self.keep(var.op_add_scalar(self.eps))
            std = self.keep(var_eps.op_sqrt())
            current = self.keep(current.op_div(std))

        if self.elementwise_affine:
            w = self.weight
            b = self.bias
            if len(self.normalized_shape) > 1:
                w = self.keep(w.reshape(self.normalized_shape))
                b = self.keep(b.reshape(self.normalized_shape))
            current = self.keep(current.op_mul(w))
            current = self.keep(current.op_add(b))

        return current


class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if affine:
            self.weight = rp.ones([num_features])
            self.weight.requires_grad_(True)
            self.track("weight", self.weight)
            self.bias = rp.zeros([num_features])
            self.bias.requires_grad_(True)
            self.track("bias", self.bias)
        else:
            self.weight = None
            self.bias = None

        self.running_mean = rp.zeros([num_features])
        self.track("running_mean", self.running_mean)
        self.running_var = rp.ones([num_features])
        self.track("running_var", self.running_var)

    def forward(self, x):
        if x.ndim != 4:
            return self._forward_flat(x)
        return self._forward_4d(x)

    def _forward_4d(self, x):
        N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

        if self._training:
            x_transposed = self.keep(x.permute([1, 0, 2, 3]))
            x_flat = self.keep(x_transposed.reshape([C, N * H * W]))

            mean = self.keep(x_flat.op_mean_dim(1, keepdim=True))
            diff = self.keep(x_flat.op_sub(mean))
            sq = self.keep(diff.op_square())
            var = self.keep(sq.op_mean_dim(1, keepdim=True))

            mean_1d = self.keep(mean.reshape([C]))
            var_1d = self.keep(var.reshape([C]))

            self._update_running_stats(mean_1d, var_1d, N * H * W)

            var_eps = self.keep(var.op_add_scalar(self.eps))
            std = self.keep(var_eps.op_sqrt())
            normed_flat = self.keep(diff.op_div(std))

            normed_transposed = self.keep(normed_flat.reshape([C, N, H, W]))
            normed = self.keep(normed_transposed.permute([1, 0, 2, 3]))
        else:
            mean_shape = [1, C, 1, 1]
            rm = self.keep(self.running_mean.reshape(mean_shape))
            rv = self.keep(self.running_var.reshape(mean_shape))

            diff = self.keep(x.op_sub(rm))
            rv_eps = self.keep(rv.op_add_scalar(self.eps))
            std = self.keep(rv_eps.op_sqrt())
            normed = self.keep(diff.op_div(std))

        if self.affine:
            w = self.keep(self.weight.reshape([1, C, 1, 1]))
            b = self.keep(self.bias.reshape([1, C, 1, 1]))
            normed = self.keep(normed.op_mul(w))
            normed = self.keep(normed.op_add(b))

        return normed

    def _forward_flat(self, x):
        C = x.shape[-1]

        if self._training:
            mean = self.keep(x.op_mean_dim(0, keepdim=True))
            diff = self.keep(x.op_sub(mean))
            sq = self.keep(diff.op_square())
            var = self.keep(sq.op_mean_dim(0, keepdim=True))

            mean_1d = self.keep(mean.reshape([C]))
            var_1d = self.keep(var.reshape([C]))

            self._update_running_stats(mean_1d, var_1d, x.shape[0])

            var_eps = self.keep(var.op_add_scalar(self.eps))
            std = self.keep(var_eps.op_sqrt())
            normed = self.keep(diff.op_div(std))
        else:
            diff = self.keep(x.op_sub(self.running_mean))
            rv_eps = self.keep(self.running_var.op_add_scalar(self.eps))
            std = self.keep(rv_eps.op_sqrt())
            normed = self.keep(diff.op_div(std))

        if self.affine:
            normed = self.keep(normed.op_mul(self.weight))
            normed = self.keep(normed.op_add(self.bias))

        return normed

    def _update_running_stats(self, batch_mean, batch_var, count):
        alpha = self.momentum
        bm = batch_mean.copy()
        bv = batch_var.copy()
        new_rm = self.running_mean.mul_scalar(1.0 - alpha).add(bm.mul_scalar(alpha))
        new_rv = self.running_var.mul_scalar(1.0 - alpha).add(bv.mul_scalar(alpha))
        self.running_mean = new_rm
        self.tracked["running_mean"] = new_rm
        self.running_var = new_rv
        self.tracked["running_var"] = new_rv
