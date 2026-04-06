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
